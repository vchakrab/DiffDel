import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_total_time_vs_threshold_sized_by_mean_constraint(file_path = 'airport_time_data_alg1.csv'):
    """
    Reads time, threshold, and constraint data, groups by threshold,
    calculates TOTAL TIME (sum) and MEAN CONSTRAINTS, and plots the result
    with dot size determined by the mean constraint value.
    Annotations show the unrounded mean constraint in GREEN and appear WELL UNDER the dot.
    """
    try:
        df = pd.read_csv(file_path)

        required_cols = ['threshold', 'time', 'constraints']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV file must contain columns: {required_cols}.")
            return

        # 1. Group the data: Calculate TOTAL TIME (sum) and MEAN CONSTRAINTS
        grouped_df = df.groupby('threshold').agg(
            total_time = ('time', 'sum'),  # <-- MODIFIED: Use sum for Total Time
            mean_constraints = ('constraints', 'mean')  # Use mean for sizing and labeling
        ).reset_index()

        # 2. Prepare data for plotting
        x_thresholds = grouped_df['threshold']
        y_total_time = grouped_df['total_time']  # <-- Use Total Time for Y-axis
        mean_constraints_values = grouped_df['mean_constraints']

        # 3. Scale the dot size based on mean_constraints_values (DRAMATIC scaling)
        min_size = 50
        scale_factor = 50
        base_multiplier = 20

        if mean_constraints_values.max() == mean_constraints_values.min():
            dot_sizes = np.full_like(mean_constraints_values, min_size)
        else:
            normalized_constraints = (mean_constraints_values - mean_constraints_values.min()) / \
                                     (mean_constraints_values.max() - mean_constraints_values.min())

            dot_sizes = min_size + normalized_constraints * scale_factor * base_multiplier

        # 4. Generate the plot
        plt.figure(figsize = (12, 7))

        # --- Plotting as a scatter plot with sized dots ---
        plt.scatter(
            x_thresholds,
            y_total_time,  # <-- Plotting Total Time
            s = dot_sizes,
            color = 'blue',
            marker = 'o',
            edgecolors = 'black',
            linewidth = 0.8,
            zorder = 5,
            label = 'Total Time (s)'
        )

        # 5. Add the unrounded mean constraint values as annotations
        for i in range(len(x_thresholds)):
            plt.annotate(
                f"{mean_constraints_values.iloc[i]:.3f}",
                (x_thresholds.iloc[i], y_total_time.iloc[i]),
                # <-- Annotating using Total Time Y-coordinate
                textcoords = "offset points",
                xytext = (0, -25),  # Offset to place text well under the dot
                ha = 'center',
                fontsize = 9,
                color = 'green',
                fontweight = 'bold'
            )

        # Set labels and title
        plt.xlabel('Threshold Value')
        plt.ylabel('Total Time (s)')  # <-- Updated Y-axis label
        plt.title('Total Time vs. Threshold (Dot Size Scaled by Mean Constraint)', fontsize = 14)

        # Set X-axis scale from 0.0 to 1.0
        plt.xlim(-0.05, 1.05)

        # Add grid
        plt.grid(True, linestyle = '--', alpha = 0.6)

        # Add a simple legend for the blue dots
        plt.legend(loc = 'upper right')

        # Save the plot
        output_file = 'threshold_total_time_scatter_final.png'
        plt.savefig(output_file)
        plt.close()

        print(f"Graph successfully generated and saved as '{output_file}'.")
        print("\nSummary of the grouped data used for plotting:")
        print(grouped_df)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Call the function to generate the plot
plot_total_time_vs_threshold_sized_by_mean_constraint()