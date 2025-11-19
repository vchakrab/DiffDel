import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


# Raw data provided by the user
data_str = """
ident,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
type,[93, 84, 79, 79, 70, 47, 37, 27, 9, 4, 0],[20, 18, 17, 17, 15, 10, 8, 6, 2, 1, 0]
name,[49, 45, 45, 45, 45, 33, 29, 25, 13, 4, 0],[12, 11, 11, 11, 11, 8, 7, 6, 3, 1, 0]
latitude_deg,[105, 100, 95, 95, 85, 57, 47, 33, 14, 4, 0],[22, 21, 20, 20, 18, 12, 10, 7, 3, 1, 0]
longitude_deg,[86, 77, 72, 72, 60, 47, 32, 27, 13, 4, 0],[19, 17, 16, 16, 13, 10, 7, 6, 3, 1, 0]
elevation_ft,[18, 18, 18, 18, 18, 13, 13, 13, 13, 4, 0],[4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 0]
iso_country,[66, 56, 56, 56, 48, 36, 32, 27, 13, 4, 0],[15, 13, 13, 13, 11, 8, 7, 6, 3, 1, 0]
iso_region,[77, 68, 63, 63, 58, 42, 32, 23, 5, 0, 0],[17, 15, 14, 14, 13, 9, 7, 5, 1, 0, 0]
municipality,[4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
scheduled_service,[43, 33, 33, 33, 29, 24, 14, 14, 4, 0, 0],[9, 7, 7, 7, 6, 5, 3, 3, 1, 0, 0]
"""

# Thresholds from 0.0 to 1.0 with 11 steps
thresholds = np.linspace(0.0, 1.0, 11)


# Function to safely evaluate the list strings
def parse_list_str(list_str):
    """Safely converts a string representation of a list into a Python list."""
    try:
        return ast.literal_eval(list_str.strip())
    except (ValueError, SyntaxError):
        return []


# Process the data
parsed_data = []
for line in data_str.strip().split('\n'):
    parts = line.split(',', 1)  # Split only on the first comma to separate attribute name

    if len(parts) < 2:
        continue

    attribute = parts[0].strip()
    list_part = parts[1].strip()

    # Corrected parsing logic: split at '],[' and reconstruct the list strings
    if '],[' in list_part:
        split_parts = list_part.split('],[', 1)

        # Reconstruct the first list string (Cell Count)
        cell_count_str = split_parts[0] + ']'

        # Reconstruct the second list string (Explanation Count)
        explanation_count_str = '[' + split_parts[1]

        cell_counts = parse_list_str(cell_count_str)
        explanation_counts = parse_list_str(explanation_count_str)

        # Ensure all lists have 11 elements
        if len(cell_counts) == 11 and len(explanation_counts) == 11:
            for i in range(11):
                parsed_data.append({
                    'Attribute': attribute,
                    'Threshold': thresholds[i],
                    'Cell_Count': cell_counts[i],
                    'Explanation_Count': explanation_counts[i]
                })

if not parsed_data:
    raise ValueError("Error: No data was successfully parsed. Please check the input format.")

df = pd.DataFrame(parsed_data)

# --- Plotting ---

# Significantly reduced figure size for a paper format
fig, ax = plt.subplots(
    figsize = (7, 4.5))  # Adjust these values as needed for your paper's requirements

# Scaling for dot size: s (area) = base_size + multiplier * Explanation_Count
# You might want to adjust BASE_SIZE and MULTIPLIER to make dots proportionally smaller
BASE_SIZE = 10  # Reduced base size
MULTIPLIER = 8  # Reduced multiplier
# Ensure the smallest dot (0 explanations) is still visible
if BASE_SIZE + MULTIPLIER * 0 < 1:
    BASE_SIZE = 1  # Minimum size for visibility

# Get unique attributes for coloring
attributes = df['Attribute'].unique()
colors = plt.cm.get_cmap('tab10', len(attributes))

for i, attribute in enumerate(attributes):
    subset = df[df['Attribute'] == attribute]
    sizes = BASE_SIZE + MULTIPLIER * subset['Explanation_Count']

    # Scatter plot for the current attribute
    ax.scatter(
        subset['Threshold'],
        subset['Cell_Count'],
        s = sizes,
        color = colors(i),
        label = attribute,
        alpha = 0.7,
        edgecolors = 'w',
        linewidth = 0.5
    )

# --- Legend Setup ---

# Generate size legend markers
max_explanation = df['Explanation_Count'].max()
unique_explanations = df['Explanation_Count'].replace(0, np.nan).dropna().unique()
size_legend_counts = [0]
if len(unique_explanations) > 0:
    if len(unique_explanations) >= 4:
        quantiles = np.quantile(unique_explanations, [0.25, 0.5, 0.75])
        size_legend_counts.extend([int(q) for q in quantiles])
    else:
        size_legend_counts.extend([int(x) for x in
                                   np.linspace(unique_explanations.min(), unique_explanations.max(),
                                               min(len(unique_explanations), 3))])

size_legend_counts.append(int(max_explanation))
size_legend_counts = sorted(list(set(size_legend_counts)))

legend_markers = []
legend_labels = []
for count in size_legend_counts:
    size = BASE_SIZE + MULTIPLIER * count
    legend_markers.append(
        ax.scatter([], [], s = size, color = 'gray', alpha = 0.7, edgecolors = 'w',
                   linewidth = 0.5))
    legend_labels.append(f'{int(count)} explanations')

# Adjust font size for legends and titles for better readability in a smaller graph
plt.rcParams.update({'font.size': 8})  # Smaller default font size

# 1. Attribute/Color legend
legend1 = ax.legend(title = 'Attribute', loc = 'upper left', bbox_to_anchor = (1.02, 1.0),
                    borderaxespad = 0., fontsize = 'small', title_fontsize = 'small')

# 2. Explanation Count/Size legend
# Placing size legend below color legend
legend2 = ax.legend(legend_markers, legend_labels, title = 'Number of Explanations\n(Dot Size)',
                    loc = 'upper left', bbox_to_anchor = (1.02, 0.4), frameon = False,
                    fontsize = 'small', title_fontsize = 'small')

# Add the first legend back manually
ax.add_artist(legend1)

# --- Axis and Title Setup ---
ax.set_xlabel('Threshold for Explanation Cutoff (0.0 to 1.0)', fontsize = 'small')
ax.set_ylabel('Actual Cell Count', fontsize = 'small')
ax.set_title('Cell Count vs. Explanation Cutoff Threshold by Attribute', fontsize = 'medium')

# Set x-axis ticks to match the thresholds
ax.set_xticks(thresholds)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, linestyle = '--', alpha = 0.6)

# Adjust tick label size
ax.tick_params(axis = 'both', which = 'major', labelsize = 7)

# Adjust layout to make room for the legends on the right
plt.tight_layout(rect = [0, 0, 0.8, 1])

# Save the plot with a higher DPI for better quality when smaller
plt.savefig('attribute_threshold_scatter_plot_paper_size.png', dpi = 300)

print("Plot saved as 'attribute_threshold_scatter_plot_paper_size.png'")