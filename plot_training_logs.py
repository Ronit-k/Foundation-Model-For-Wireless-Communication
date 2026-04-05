import re
import argparse
import matplotlib.pyplot as plt
import os

def parse_log(file_path):
    epochs = []
    train_losses = []
    val_losses = []
    
    # Regex patterns
    # Example: Epoch 1/100
    epoch_pattern = re.compile(r'Epoch (\d+)/\d+')
    # Example: Training Loss: 3.951485
    train_loss_pattern = re.compile(r'Training Loss: ([\d.]+)')
    # Example: Validation Loss: 1.169559
    val_loss_pattern = re.compile(r'Validation Loss: ([\d.]+)')
    
    current_epoch = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Match Epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            
            # Match Training Loss
            train_match = train_loss_pattern.search(line)
            if train_match:
                train_val = float(train_match.group(1))
                # Only append if we found an epoch for it
                if current_epoch is not None:
                    # If this epoch was already partially filled or we're on a new one
                    if len(epochs) < current_epoch:
                        epochs.append(current_epoch)
                        train_losses.append(train_val)
                    else:
                        train_losses[-1] = train_val
                continue
                
            # Match Validation Loss
            val_match = val_loss_pattern.search(line)
            if val_match:
                val_val = float(val_match.group(1))
                if current_epoch is not None:
                    # Validation usually comes after training loss in each epoch block
                    # Ensure we have a corresponding train loss entry
                    if len(val_losses) < len(train_losses):
                        val_losses.append(val_val)
                    else:
                        val_losses[-1] = val_val
                continue
                
    # Basic sanity check: ensure all lists are the same length
    min_len = min(len(epochs), len(train_losses), len(val_losses))
    return epochs[:min_len], train_losses[:min_len], val_losses[:min_len]

def plot_logs(log_files, output_file, title):
    # Set non-interactive backend
    plt.switch_backend('Agg')
    plt.figure(figsize=(12, 8))
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} does not exist. Skipping.")
            continue
            
        label = os.path.basename(log_file).replace('.log', '')
        epochs, train_losses, val_losses = parse_log(log_file)
        
        if not epochs:
            print(f"Warning: No data found in {log_file}.")
            continue
            
        plt.plot(epochs, train_losses, label=f'{label} (Train)', linestyle='--')
        plt.plot(epochs, val_losses, label=f'{label} (Val)', linestyle='-')
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training progress from log files.')
    parser.add_argument('logs', nargs='+', help='Path(s) to log file(s)')
    parser.add_argument('--output', '-o', default='training_progress.png', help='Output plot file name (default: training_progress.png)')
    parser.add_argument('--title', '-t', default='Training and Validation Loss', help='Plot title')
    
    args = parser.parse_args()
    
    plot_logs(args.logs, args.output, args.title)
