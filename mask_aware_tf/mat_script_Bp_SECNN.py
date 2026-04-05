# %%
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import subprocess
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split, TensorDataset
import csv, json, time
from sklearn.metrics import f1_score
from tqdm import tqdm  # Progress bar
import torch.optim as optim
import math
matplotlib.use('Agg')

from lwm1_1.input_preprocess import tokenizer
from lwm1_1.utils import prepare_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

#######SELECT MODEL##############################################
# choose one: 'vit' or 'pseudo'
mdl = int(input("choose model 0 for MAT-ViT, 1 for MAT-Pseudo: "))
model_choices = ['vit', 'pseudo']
model_choice = model_choices[mdl]  # Change index to select model

#######SELECT INPUT##############################################
# choose one: 'cls_emb', 'channel_emb', or 'raw'
input_types = ['cls_emb', 'channel_emb', 'raw']
selected_input_type = input_types[0]
################Select Tasks#####################################
tasks = ['LoS/NLoS Classification', 'Beam Prediction']
task = tasks[1] # Choose 0 for LoS/NLoS labels or 1 for beam prediction labels.
num_epochs = 30
batch_size = 32  # Set a value (adjust as needed)
print(
    "---------------------------- training Details ----------------------------\n"
    f"Model: MAT-{model_choice.upper()}-LWM (light)\n"
    f"epochs: {num_epochs}, "
    f"batch size: {batch_size}, "
    f"input type: {selected_input_type}\n"
    f"task: {task}"
)

# %%
# Define scenario names and select one (or more).
scenario_names = np.array([
    "city_0_newyork", "city_1_losangeles", "city_2_chicago", "city_3_houston",
    "city_4_phoenix", "city_5_philadelphia", "city_6_miami", "city_7_sandiego",
    "city_8_dallas", "city_9_sanfrancisco", "city_10_austin", "city_11_santaclara",
    "city_12_fortworth", "city_13_columbus", "city_15_indianapolis", "city_17_seattle",
    "city_18_denver", "city_19_oklahoma", "O1_3p5B"])
#################################################### Select the first scenario (index 0) – adjust as needed##################################################
scenario_idxs = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])[0:19]
selected_scenario_names = scenario_names[scenario_idxs]
print("selected scenarios: ")
for i in selected_scenario_names: print(i, end=", ")

# %%
n_beams = 64  # Set the number of beams for beam prediction task (adjust as needed)
task_type = ["classification", "regression"][0]
preprocessed_data, labels, raw_chs = tokenizer(
    selected_scenario_names,
    bs_idxs=[1],
    load_data=True,
    task=task,
    n_beams=n_beams,
    manual_data=None,
    mask=False)
print(preprocessed_data.shape)
print(labels.shape)

# %%
#%% LOAD THE MODEL
gpu_ids = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Model selection ---
if model_choice == 'vit':
    from mask_aware_tf.mat_vit_lwm import MATViTLWM
    model = MATViTLWM(gen_raw=True).to(device)
    model_name = "512_100mat_vit_lwm_weights.pth"
    results_file = "results_mat_vit.txt"
    photo_prefix = "vit"
elif model_choice == 'pseudo':
    from mask_aware_tf.mat_pseudo_lwm import MATPseudoLWM
    model = MATPseudoLWM(gen_raw=True).to(device)
    model_name = "512_100mat_pseudo_lwm_weights.pth"
    results_file = "results_mat_pseudo.txt"
    photo_prefix = "pseudo"
else:
    raise ValueError(f"Unknown model_choice: {model_choice}. Use 'vit' or 'pseudo'.")

state_dict = torch.load(f"{model_name}", map_location=device)
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
new_state_dict = {k.replace("module.", ""): v for k, v in new_state_dict.items()}
model.load_state_dict(new_state_dict, strict=True)

lwm_model = model
lwm_model.eval()
print(f"Model [{model_choice}] loaded successfully on {device}")

# %%
N = raw_chs.shape[0]
HW = raw_chs.shape[1] // 2        # half = H*W
H = W = int(HW ** 0.5)
channels_ri = torch.zeros(N, 2, H, W, dtype=torch.float32)
channels_ri[:, 0] = raw_chs[:, :HW].view(N, H, W)   # real
channels_ri[:, 1] = raw_chs[:, HW:].view(N, H, W)   # imag

def get_embeddings_mat(mat_model, channels_ri, input_type, model_choice, device, batch_size=64):
    """Extract embeddings from MAT models (vit or pseudo)."""
    loader = DataLoader(TensorDataset(channels_ri), batch_size=batch_size, shuffle=False)
    all_embs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            if input_type == "raw":
                if model_choice == 'pseudo':
                    from mask_aware_tf.mat_pseudo_lwm import channels_to_patches
                    patches = channels_to_patches(batch)  # (B, 128, 16)
                    all_embs.append(patches.cpu())
                else:
                    # vit: use proj_in to get initial features, then flatten
                    feat = mat_model.proj_in(batch)  # (B, 64, 32, 32)
                    B = feat.size(0)
                    feat_flat = feat.view(B, 64, -1).permute(0, 2, 1)  # (B, 1024, 64)
                    all_embs.append(feat_flat.cpu())
            else:
                cls_emb, channel_emb = mat_model(batch)
                if input_type == "cls_emb":
                    all_embs.append(cls_emb.cpu())      # (B, 64)
                elif input_type == "channel_emb":
                    all_embs.append(channel_emb.cpu())   # (B, 128, 64)

    return torch.cat(all_embs, dim=0).float()

with torch.no_grad():
    dataset = get_embeddings_mat(lwm_model, channels_ri, selected_input_type, model_choice, device, batch_size=batch_size)

torch.cuda.empty_cache()

# %%
# Initial log (Header)
message = (
    "---------------------------- training Details ----------------------------\n"
    f"Model: MAT-{model_choice.upper()}-LWM (light)\n"
    f"Dataset Size: {len(dataset)}, shape: {dataset.shape}\n"
    f"epochs: {num_epochs}, "
    f"batch size: {batch_size}, "
    f"input type: {selected_input_type}\n"
    f"task: {task}"
)
# Write header to file
with open(results_file, "a") as f:
    f.write("\n" + message)
print("\n\ninitiated", results_file, "with\n", message, '\n'*3)
print(selected_input_type, "dataset shape:", dataset.shape)

# %%
unique_labels, counts = torch.unique(labels, return_counts=True)
x_values = unique_labels.cpu().numpy()
y_values = counts.cpu().numpy()

plt.figure(figsize=(18, 7)) 
bars = plt.bar(x_values, y_values, color='skyblue', edgecolor='black', alpha=0.8)
plt.xticks(x_values, rotation=0, fontsize=9) 
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
             va='bottom', ha='center', fontsize=7, rotation=0)

plt.xlabel('Beam Index (Labels)', labelpad=15)
plt.ylabel('Frequency')
plt.title(f'Frequency Distribution of Beam Labels ($N = {len(labels)}$)')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# %%
#function to combine data and labels and split in the given train ratio ratio
def get_data_loaders(data_tensor, labels_tensor, batch_size, train_ratio):
    dataset = TensorDataset(data_tensor, labels_tensor)
    N = len(dataset)

    train_size = int(train_ratio * N)
    remaining = N - train_size
    val_size = remaining // 2
    test_size = remaining - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# %%
# Mapping for beam prediction input types — MAT models use d_model=64
if model_choice == 'pseudo':
    mapping = {
        'cls_emb': {'input_channels': 1, 'sequence_length': 64},      # CLS dim = 64
        'channel_emb': {'input_channels': 128, 'sequence_length': 64}, # (B, 128, 64)
        'raw': {'input_channels': 16, 'sequence_length': 128}          # (B, 128, 16) raw patches
    }
elif model_choice == 'vit':
    mapping = {
        'cls_emb': {'input_channels': 1, 'sequence_length': 64},      # CLS dim = 64
        'channel_emb': {'input_channels': 128, 'sequence_length': 64}, # (B, 128, 64)
        'raw': {'input_channels': 64, 'sequence_length': 1024}         # (B, 1024, 64) from proj_in
    }

input_type = selected_input_type
params = mapping.get(input_type, mapping[selected_input_type])
initial_lr = 0.001
num_classes = n_beams + 1
print(selected_input_type)

# %%
# ----------------------------------
# 1. SE LAYER
# ----------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# ----------------------------------
# 2. RESIDUAL BLOCK (WITH OPTIONAL SE)
# ----------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_se=False):
        super().__init__()

        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)

        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_c) if use_se else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply SE attention (if enabled)
        out = self.se(out)

        out += identity
        out = self.relu(out)

        return out

# ----------------------------------
# 3. MAIN MODEL
# ----------------------------------
class SEResNet1D(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes):
        super().__init__()

        # No early downsampling
        self.conv1 = nn.Conv1d(
            input_channels, 64,
            kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (No SE)
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # Stage 2 (SE starts)
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_se=True),
            ResidualBlock(128, 128, use_se=True)
        )

        # Stage 3 (SE)
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_se=True),
            ResidualBlock(256, 256, use_se=True)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)

        # Infer FC size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, sequence_length)
            dummy = self._forward_conv(dummy)
            flatten_size = dummy.view(1, -1).size(1)

        self.fc = nn.Linear(flatten_size, num_classes)

    def _forward_conv(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return x

    def forward(self, x):
        # Input shape: [B, L, C]
        x = x.transpose(1, 2)  # -> [B, C, L]

        x = self._forward_conv(x)
        x = x.flatten(1)
        x = self.dropout(x)

        return self.fc(x)

print("Final SE-ResNet1D Model Defined.")


# ----------------------------------
# 4. LABEL SMOOTHING LOSS
# ----------------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# %%
# Function to plot training metrics.
def plot_training_metrics(epochs, train_losses, val_losses, val_f1_scores, save_path=None):
    plt.figure(figsize=(12, 5))
    # Loss plot.
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    # F1 score plot.
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1_scores, label='Validation Weighted F1', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# %%
# Create photo directories
photo_dir = os.path.join(os.path.dirname(__file__), "photos", f"{photo_prefix}_{selected_input_type}")
os.makedirs(photo_dir, exist_ok=True)

# %%
# Define the split ratios to iterate over
split_ratios = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4]

for split_ratio in split_ratios:
    print(f"\n--- Starting training for split ratio: {split_ratio} ---")

    # Instantiate the model FRESH for every split ratio (train from scratch)
    beam_model = SEResNet1D(params['input_channels'], params['sequence_length'], num_classes).to(device)
    optimizer = Adam(beam_model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
        T_0=10,      # Restart every 10 epochs
        T_mult=2,    # Double the restart interval (10, 20, 40...)
        eta_min=1e-6 # Minimum LR
    )
    print("Advanced Optimizer and Scheduler Initialized.")
    
    # Get DataLoaders for the current split_ratio
    train_loader, val_loader, test_loader = get_data_loaders(dataset, labels, batch_size=batch_size, train_ratio=split_ratio)
    
    print(f"train: {len(train_loader)} | validate: {len(val_loader)} | test: {len(test_loader)}")
    
    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
    train_losses = []
    val_losses = []
    val_f1_scores = []
    epochs_list = []

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, num_epochs + 1):
        beam_model.train()
        running_loss = 0.0
        # Training with tqdm progress bar.
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
            data, target = data.to(device), target.to(device)
            # Adjust input shape based on type.
            if input_type == 'raw':
                data = data.view(data.size(0), params['sequence_length'], params['input_channels'])
            elif input_type == 'cls_emb':
                data = data.unsqueeze(2)
            optimizer.zero_grad()
            outputs = beam_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(beam_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Validation loop with tqdm.
        beam_model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_targets = []
        for data, target in tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False):
            data, target = data.to(device), target.to(device)
            if input_type == 'raw':
                data = data.view(data.size(0), params['sequence_length'], params['input_channels'])
            elif input_type == 'cls_emb':
                data = data.unsqueeze(2)
            outputs = beam_model(data)
            loss = criterion(outputs, target)
            val_running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        val_loss = val_running_loss / len(val_loader.dataset)
        val_f1 = f1_score(all_targets, all_preds, average='weighted')

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Weighted F1: {val_f1:.4f}")

    # -----------------------------
    # Test Loop (After Training)
    # -----------------------------
    beam_model.eval()
    test_running_loss = 0.0
    all_preds_test = []
    all_targets_test = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            if input_type == 'raw':
                data = data.view(data.size(0), params['sequence_length'], params['input_channels'])
            elif input_type == 'cls_emb':
                data = data.unsqueeze(2)
            outputs = beam_model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds_test.extend(predicted.cpu().numpy())
            all_targets_test.extend(target.cpu().numpy())
            
    accuracy = 100 * correct / total
    test_f1 = f1_score(all_targets_test, all_preds_test, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.2f}%, Test F1: {test_f1:.4f}")

    # -----------------------------
    # Save results to file
    # -----------------------------
    with open(results_file, "a") as f:
        f.write(
            f"\nSplit Ratio: {split_ratio} | "
            f"Test Accuracy: {accuracy:.2f}% | "
            f"Test F1: {test_f1:.4f}\n"
        )
    print(f"Results saved to {results_file}")

    # -----------------------------
    # Save plot
    # -----------------------------
    fig = plt.figure()
    plot_training_metrics(epochs_list, train_losses, val_losses, val_f1_scores)
    plot_path = os.path.join(photo_dir, f"{selected_input_type}_{split_ratio}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved as {plot_path}")


