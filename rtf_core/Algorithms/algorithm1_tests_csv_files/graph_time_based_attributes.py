import pandas as pd
import matplotlib.pyplot as plt

FILE_NAME = 'airport_explanations_data.csv'
def plot_time_by_attribute(file_path):
    """Reads data from a CSV, plots Time vs. Attribute, sorted by Time."""
    try:
        # Load the data from the CSV file
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return

    # Sort by 'time' for a visually ordered bar chart
    df_sorted = df.sort_values('time', ascending = False)

    # Create the plot
    plt.figure(figsize = (10, 6))
    plt.bar(df_sorted['attribute'], df_sorted['time'], color = 'skyblue')

    # Add labels and title
    plt.xlabel('Attribute Name', fontsize = 12)
    plt.ylabel('Time (s)', fontsize = 12)
    plt.title('Execution Time by Attribute (Sorted)', fontsize = 14)
    plt.xticks(rotation = 45, ha = 'right')  # Rotate names for better readability
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
    plt.tight_layout()

    # Display the plot
    plt.show()


# Run the function
plot_time_by_attribute(FILE_NAME)