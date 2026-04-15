import matplotlib.pyplot as plt
import seaborn as sns

def plot_results():
    # --- Aesthetic Setup ---
    # Apply a modern clean seaborn theme
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # --- Data Definition ---
    split_ratios = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4]

    # Configurations & F1 Scores
    data_cls = {
        # 'Base LWM1.0 + Base CNN':   [0.4231, 0.5108, 0.6642, 0.7200, 0.7609, 0.8031],
        'LWM1.0_CA + SECNN':        [0.5805, 0.6601, 0.7959, 0.8351, 0.8576, 0.8748],
        # 'Base LWM1.1 + Base CNN':[0.4003, 0.4046, 0.5679, 0.5758, 0.6878, 0.7314],
        'LWM1.1_CA + SECNN':     [0.4921, 0.5686, 0.7218, 0.7632, 0.7823, 0.8021],
        # 'mat_pseudo + Base CNN': [0.4598, 0.5761, 0.6821, 0.7720, 0.7983, 0.8188],
        'mat_pseudo + SECNN': [0.5771, 0.6509, 0.7790, 0.8067, 0.8218, 0.8370],
        'mat_pseudo_ca + SECNN': [0.6219, 0.6821, 0.7917, 0.8202, 0.8390, 0.8515] #put actual values
    }
    data_emb = {
        'Base LWM1.0 + Base CNN':   [0.7815, 0.8361, 0.9082, 0.9296, 0.9394, 0.9524],
        # 'LWM1.0_CA + SECNN':        [0.8196, 0.8694, 0.9287, 0.9410, 0.9525, 0.9592],
        'Base LWM1.1 + Base CNN':[0.7119, 0.7905, 0.8866, 0.9105, 0.9302, 0.9454],
        # 'LWM1.1_CA + SECNN':     [0.7528, 0.8164, 0.9101, 0.9255, 0.9400, 0.9480],
        'mat_pseudo + Base CNN': [0.7327, 0.8209, 0.9089, 0.9244, 0.9380, 0.9466],
        'mat_pseudo + SECNN': [0.8074, 0.8551, 0.9174, 0.9316, 0.9437, 0.9520],
        'mat_pseudo_ca + SECNN': [0.8074, 0.8551, 0.9174, 0.9316, 0.9437, 0.9520] #put actual values
    }

    choice = ['Channel','cls'][1]
    if (choice == 'cls'):
        data = data_cls
    else:
        data = data_emb
    
    # Modern professional color palette
    colors = ['#5b7c99', '#4b917d', '#F28E2B', '#E15759']  # Muted Indigo, Teal, Warm Orange, Coral Red
    markers = ['o', 's', '^', 'D']

    # --- Plotting Setup ---
    # Create 1 row, 2 columns with a refined soft background
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')  
    
    # Main Heading 
    fig.suptitle(f'Using {choice} Embeddings for Beam Prediction Task', 
                 fontsize=22, fontweight='bold', color='#2C3E50', y=0.96)

    for i, ax in enumerate(axes):
        ax.set_facecolor('#FFFFFF')
        
        # Plot each line
        for (label, scores), color, marker in zip(data.items(), colors, markers):
            # 1. Subtle drop shadow for depth
            ax.plot(split_ratios, scores, color='black', linewidth=4, alpha=0.1, 
                    zorder=1, label='_nolegend_')
            
            # 2. Main styled line
            ax.plot(split_ratios, scores, marker=marker, color=color, linewidth=2.8, 
                    markersize=10, markeredgecolor='white', markeredgewidth=1.8,
                    zorder=2, label=label)

        # Labels & Styling
        ax.set_xlabel('Split Ratio (Training Data Fraction)', fontsize=14, fontweight='bold', color='#34495E')
        ax.set_ylabel('Test F1 Score', fontsize=14, fontweight='bold', color='#34495E')
        
        # Smooth transparent grid
        ax.grid(True, linestyle='--', alpha=0.6, color='#BDC3C7')
        ax.tick_params(axis='both', colors='#34495E', labelsize=12)

        # Spine formatting (removing borders for clean look)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#BDC3C7')
            ax.spines[spine].set_linewidth(1.5)

        # Titles and Scaling
        if i == 0:
            ax.set_title('Linear Scale View', fontsize=16, pad=15, fontweight='bold', color='#2C3E50')
        else:
            ax.set_title('Logarithmic Scale View', fontsize=16, pad=15, fontweight='bold', color='#2C3E50')
            ax.set_xscale('log')
            ax.set_xticks(split_ratios)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # Enhanced stylized legend (only on the right plot to keep it clean)
        if i == 1:
            legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, 
                               borderpad=1.2, prop={'weight': 'bold', 'size': 12})
            legend.get_frame().set_facecolor('#FFFFFF')
            legend.get_frame().set_edgecolor('#BDC3C7')

    # Perfectly fit the geometries
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) 
    
    # --- Show Plot ---
    print("Displaying plots...")
    plt.show()

    # --- Save Prompt ---
    save_query = input(f"\nDo you want to save this plot as 'Mat_pseudo_{choice}_benchmark.png'? (yes/no): ").strip().lower()

    if save_query in ['yes', 'y']:
        # Ensure identical styling goes into the PNG file
        fig.savefig(f'Mat_pseudo_{choice}_benchmark.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        print("Successfully saved to 'Mat_pseudo_{choice}_benchmark.png'")
    else:
        print("Plot not saved.")

if __name__ == "__main__":
    plot_results()
