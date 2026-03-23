import matplotlib.pyplot as plt

def plot_results():
    # --- Data Definition ---
    split_ratios = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4]

    # Configurations & F1 Scores
    data_cls = {
        'Base LWM1.0 + Base CNN':   [0.4231, 0.5108, 0.6642, 0.7200, 0.7609, 0.8031],
        'LWM1.0_CA + SECNN':        [0.5805, 0.6601, 0.7959, 0.8351, 0.8576, 0.8748],
        'Base LWM1.1 + Base CNN':[0.4003, 0.4046, 0.5679, 0.5758, 0.6878, 0.7314],
        'LWM1.1_CA + SECNN':     [0.4921, 0.5686, 0.7218, 0.7632, 0.7823, 0.8021]
    }
    data_emb = {
        'Base LWM1.0 + Base CNN':   [0.7815, 0.8361, 0.9082, 0.9296, 0.9394, 0.9524],
        'LWM1.0_CA + SECNN':        [0.8196, 0.8694, 0.9287, 0.9410, 0.9525, 0.9592],
        'Base LWM1.1 + Base CNN':[0.7119, 0.7905, 0.8866, 0.9105, 0.9302, 0.9454],
        'LWM1.1_CA + SECNN':     [0.7528, 0.8164, 0.9101, 0.9255, 0.9400, 0.9480]
    }

    data = data_emb
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'd']

    # --- Plotting Setup ---
    # Create 1 row, 2 columns (Side-by-side plots)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main Heading for the entire figure
    fig.suptitle('Using Channel_Embeddings for Beam Prediction Task', fontsize=16, fontweight='bold')

    # Loop to create both Linear (index 0) and Log (index 1) plots
    for i, ax in enumerate(axes):
        # Plot each line
        for (label, scores), color, marker in zip(data.items(), colors, markers):
            ax.plot(split_ratios, scores, marker=marker, color=color, linewidth=2, label=label)

        # Common Labels
        ax.set_xlabel('Split Ratio (Training Data Fraction)', fontsize=12)
        ax.set_ylabel('Test F1 Score', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Specifics for Linear vs Log
        if i == 0:
            ax.set_title('Linear Scale View', fontsize=14)
        else:
            ax.set_title('Logarithmic Scale View', fontsize=14)
            ax.set_xscale('log')
            # Set specific ticks for log scale to match the data points exactly
            ax.set_xticks(split_ratios)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    # --- Show Plot ---
    print("Displaying plots...")
    plt.show()

    # --- Save Prompt ---
    save_query = input("Do you want to save this plot as 'results_all4_cls.png'? (yes/no): ").strip().lower()

    if save_query in ['yes', 'y']:
        # Save the specific figure
        fig.savefig('channel_results_all4.png', dpi=300)
        print("Successfully saved to 'results_all4.png'")
    else:
        print("Plot not saved.")

if __name__ == "__main__":
    plot_results()
