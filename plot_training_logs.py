import re
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

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
                if current_epoch is not None:
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
                    if len(val_losses) < len(train_losses):
                        val_losses.append(val_val)
                    else:
                        val_losses[-1] = val_val
                continue
                
    min_len = min(len(epochs), len(train_losses), len(val_losses))
    return np.array(epochs[:min_len]), np.array(train_losses[:min_len]), np.array(val_losses[:min_len])

def plot_logs(log_files, output_file, title, use_log_scale=False):
    # Set non-interactive backend
    plt.switch_backend('Agg')
    
    # Try to use a nice style
    try:
        plt.style.use('seaborn-v0_8-muted')
    except:
        plt.style.use('ggplot')
        
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Predefined color cycle for consistent multi-file plotting
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for i, log_file in enumerate(log_files):
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} does not exist. Skipping.")
            continue
            
        label = os.path.basename(log_file).replace('.log', '').replace('_weights', '')
        epochs, train_losses, val_losses = parse_log(log_file)
        
        if len(epochs) == 0:
            print(f"Warning: No data found in {log_file}.")
            continue
            
        color = colors[i]
        
        # Plot curves
        ax.plot(epochs, train_losses, label=f'{label} (Train)', linestyle='--', alpha=0.6, color=color, linewidth=1.5)
        ax.plot(epochs, val_losses, label=f'{label} (Val)', linestyle='-', alpha=1.0, color=color, linewidth=2.5)
        
        # Fill between for awareness of convergence gap
        ax.fill_between(epochs, train_losses, val_losses, color=color, alpha=0.05)
        
        # Annotate minimum validation loss
        min_val_idx = np.argmin(val_losses)
        min_val = val_losses[min_val_idx]
        min_epoch = epochs[min_val_idx]
        
        ax.scatter(min_epoch, min_val, color=color, s=50, edgecolors='white', zorder=5)
        ax.annotate(f'min: {min_val:.4f}', 
                    xy=(min_epoch, min_val), 
                    xytext=(10, -10), 
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec=color))

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    if use_log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (Log Scale)', fontsize=12, fontweight='bold')
        
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    
    # Add minor grid for log scale specifically
    if use_log_scale:
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training progress from log files.')
    parser.add_argument('logs', nargs='+', help='Path(s) to log file(s)')
    parser.add_argument('--output', '-o', default='training_progress.png', help='Output plot file name (default: training_progress.png)')
    parser.add_argument('--title', '-t', default='Training and Validation Loss', help='Plot title')
    parser.add_argument('--log-scale', '-l', action='store_true', help='Use log scale for y-axis')
    
    args = parser.parse_args()
    
    plot_logs(args.logs, args.output, args.title, args.log_scale)
