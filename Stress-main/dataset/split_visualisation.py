import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_class_distribution(data_dir='c:\\Users\\shaff\\Downloads\\Stress_Detect\\Stress-main\\processed_data'):
    """Create elegant bar charts showing class distributions."""
    # Load the data
    train_path = os.path.join(data_dir, 'train_data.npy')
    test_path = os.path.join(data_dir, 'test_data.npy')
    
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    # Extract labels and count
    train_labels = train_data[:, -1]
    test_labels = test_data[:, -1]
    
    train_no_stress = np.sum(train_labels == 0)
    train_stress = np.sum(train_labels == 1)
    test_no_stress = np.sum(test_labels == 0)
    test_stress = np.sum(test_labels == 1)
    
    # Create figure with closer subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.1)  # Reduce space between plots
    
    # Much slimmer bars
    bar_width = 0.2
    # Green color scheme - two different shades of green
    colors = ['#2ecc71', '#27ae60']  # Light green and dark green
    
    # Use custom positions for x axis to place bars MUCH closer together
    x_positions = [0.35, 0.65]  # Even closer than before
    
    # Plot for train set - slimmer bars with custom positions
    ax1.bar(x_positions, [train_no_stress, train_stress], 
            color=colors, width=bar_width, alpha=0.9)
    ax1.set_title('Training Set', fontsize=12)
    ax1.set_ylabel('Samples', fontsize=10)
    ax1.grid(False)  # Remove grid
    ax1.set_xlim(0, 1)  # Tighter x-axis limits
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(['No Stress', 'Stress'])
    
    # Plot for test set - slimmer bars with custom positions
    ax2.bar(x_positions, [test_no_stress, test_stress], 
            color=colors, width=bar_width, alpha=0.9)
    ax2.set_title('Validation Set', fontsize=12)
    ax2.grid(False)  # Remove grid
    ax2.set_xlim(0, 1)  # Tighter x-axis limits
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(['No Stress', 'Stress'])
    
    # Add counts on top of bars (simpler)
    for ax, counts in [(ax1, [train_no_stress, train_stress]), 
                       (ax2, [test_no_stress, test_stress])]:
        for i, (pos, count) in enumerate(zip(x_positions, counts)):
            ax.text(pos, count + 0.5, str(count), ha='center', fontsize=9)
    
    # Clean title with just essential info
    plt.suptitle('Dataset Distribution', fontsize=12, y=0.98)
    
    # Clean layout
    plt.tight_layout()
    
    # Save and show
    output_path = os.path.join(data_dir, 'class_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()
    
    return {
        'train': {'no_stress': train_no_stress, 'stress': train_stress},
        'test': {'no_stress': test_no_stress, 'stress': test_stress}
    }

if __name__ == "__main__":
    visualize_class_distribution()