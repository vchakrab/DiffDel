import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file_path = '/Users/adhariya/src/RTF25/rtf_core/Algorithms/algorithm1_tests_csv_files/all_dataset_edge_data.csv'
try:
    # Read the data from the CSV file
    df = pd.read_csv(csv_file_path)

    # --- Plot 1: Time vs. Dataset (Bar Graph) ---

    # Create a figure for the plot
    plt.figure(figsize = (10, 6))

    # Sort by time for a cleaner-looking graph
    df_sorted_time = df.sort_values(by = 'time')

    # Create the bar chart
    plt.bar(df_sorted_time['name'], df_sorted_time['time'], color = 'skyblue')

    # Set labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs. Dataset')
    plt.xticks(rotation = 45, ha = 'right')  # Rotate x-axis labels if they overlap

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('time_vs_dataset.png')
    print(f"Successfully saved 'time_vs_dataset.png'")

    # --- Plot 2: Other Metrics (Grouped Bar Graph) ---

    # Get the data for plotting
    labels = df['name']
    boundary_edges = df['boundary_edges']
    internal_edges = df['internal_edges']
    internal_cells = df['internal_cells']

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    # Create the figure and axes
    fig, ax = plt.subplots(figsize = (12, 7))

    # Create the bars for each metric, offsetting the x-position
    rects1 = ax.bar(x - width, boundary_edges, width, label = 'boundary_edges')
    rects2 = ax.bar(x, internal_edges, width, label = 'internal_edges')
    rects3 = ax.bar(x + width, internal_cells, width, label = 'internal_cells')

    # Add labels, title, and legend
    ax.set_ylabel('Count')
    ax.set_xlabel('Dataset')
    ax.set_title('Metrics by Dataset')
    ax.set_xticks(x)  # Set x-tick positions
    ax.set_xticklabels(labels)  # Set x-tick labels
    ax.legend()

    # Add labels on top of the bars
    ax.bar_label(rects1, padding = 3)
    ax.bar_label(rects2, padding = 3)
    ax.bar_label(rects3, padding = 3)

    # Adjust layout and save the figure
    fig.tight_layout()
    plt.savefig('metrics_vs_dataset.png')
    print(f"Successfully saved 'metrics_vs_dataset.png'")

    print("\nData successfully read and plotted.")
    print("Data Head:")
    print(df.head())


except FileNotFoundError:
    print(f"--- ERROR ---")
    print(f"The file '{csv_file_path}' was not found.")
    print("Please make sure the variable 'csv_file_path' is set correctly.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")