import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = 'nc_voter_explanations_data.csv'

def plot_explanations_and_depth(file_path):
    """Reads data from a CSV and plots a grouped bar chart for Explanations and Depth."""
    try:
        # Load the data from the CSV file
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return

    # Data setup
    labels = df['attribute']
    bar_width = 0.35
    x = np.arange(len(labels))  # The label locations

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the two sets of bars, offsetting them by bar_width/2
    rects1 = ax.bar(x - bar_width/2, df['explanations'], bar_width, label='Explanations Count', color='teal')
    rects2 = ax.bar(x + bar_width/2, df['depth'], bar_width, label='Depth', color='orange')

    # Set custom x-axis tick labels, title, and legend
    ax.set_ylabel('Count / Depth Value', fontsize=12)
    ax.set_xlabel('Attribute Name', fontsize=12)
    ax.set_title('Explanations and Depth by Attribute', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Display the plot
    plt.show()

# Run the function
plot_explanations_and_depth(FILE_NAME)