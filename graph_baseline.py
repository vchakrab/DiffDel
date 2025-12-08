import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --- Placeholder for the CSV file name ---
CSV_FILE_NAME = 'baseline_deletion_2_data.csv'

# Aggregated data that would be derived from the CSV
# NOTE: This data is hardcoded here for the visualization purpose,
# but a function to calculate it dynamically is shown below.
data = {
    'Dataset': ['hospital', 'ncvoter', 'tax'],
    'Total Time (s)': [7.9621, 24.3228, 1.1578, 30.6865],
    'Total Cells Deleted': [511, 3363, 1029, 451]
}

df_plot = pd.DataFrame(data)

# Sort the DataFrame by 'Total Time (s)'
df_plot = df_plot.sort_values(by = 'Total Time (s)', ascending = False)


# --- Visualization Code ---
def create_dual_axis_chart(df):
    """Generates the dual-axis bar chart from the aggregated DataFrame."""
    fig, ax1 = plt.subplots(figsize = (10, 6))

    # --- Setup for Primary Axis (Time) ---
    color_time = 'tab:blue'
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Total Time (s)', color = color_time)
    bars1 = ax1.bar(df['Dataset'], df['Total Time (s)'], color = color_time, alpha = 0.6,
                    width = 0.4, label = 'Total Time (s)')
    ax1.tick_params(axis = 'y', labelcolor = color_time)
    ax1.set_ylim(0, df['Total Time (s)'].max() * 1.1)

    # --- Setup for Secondary Axis (Cells Deleted) ---
    ax2 = ax1.twinx()  # Create a second axes sharing the same x-axis
    color_cells = 'tab:red'
    ax2.set_ylabel('Total Cells Deleted', color = color_cells)

    x_pos = np.arange(len(df['Dataset']))
    width = 0.4
    # Plot cells, offset to be next to time bars
    bars2 = ax2.bar(x_pos + width, df['Total Cells Deleted'], color = color_cells, alpha = 0.6,
                    width = 0.4, label = 'Total Cells Deleted')
    ax2.tick_params(axis = 'y', labelcolor = color_cells)
    ax2.set_ylim(0, df['Total Cells Deleted'].max() * 1.1)

    plt.title('Performance Metrics by Dataset: Time vs. Cells Deleted')

    # Add data labels
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.01 * ax1.get_ylim()[1],
                 f'{yval:.2f}s', ha = 'center', va = 'bottom', color = color_time, fontsize = 9)

    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.01 * ax2.get_ylim()[1],
                 f'{yval}', ha = 'center', va = 'bottom', color = color_cells, fontsize = 9)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc = 'upper right')

    # Adjust x-ticks to be centered between the grouped bars
    ax1.set_xticks(x_pos + width / 2)
    ax1.set_xticklabels(df['Dataset'])

    fig.tight_layout()
    plt.show()


# --- Example of Dynamic Data Aggregation (If your CSV is correctly formatted) ---
# Assuming your CSV has columns: 'Dataset', 'time', and 'cells'
def aggregate_data(file_path):
    """Loads a CSV and aggregates time and cells by Dataset."""
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please create it first.")
        return None

    # Group by the Dataset column (e.g., airport, hospital, etc.) and sum the time and cells
    df_agg = df_raw.groupby('Dataset')[['time', 'cells']].sum().reset_index()
    df_agg.columns = ['Dataset', 'Total Time (s)', 'Total Cells Deleted']
    return df_agg


# --- Main Execution ---
# You would uncomment the following lines and prepare your CSV file
# df_aggregated = aggregate_data(CSV_FILE_NAME)
# if df_aggregated is not None:
#     # Sort for better visualization
#     df_aggregated = df_aggregated.sort_values(by='Total Time (s)', ascending=False)
#     create_dual_axis_chart(df_aggregated)

# For now, we will use the hardcoded data to show the output chart structure:
create_dual_axis_chart(df_plot)