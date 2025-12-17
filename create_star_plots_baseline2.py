import csv
from math import pi, cos, sin
import sys
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

def parse_csv_for_baselines(file_path):
    """General purpose CSV parser for all baseline formats."""
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping: {file_path}")
        return data

    print(f"DEBUG: Starting parsing for file: {file_path}") # Added debug print

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        current_dataset = None
        header_map = {}
        for row in reader:
            if not row: continue
            if row[0].startswith('-----'):
                current_dataset = row[0].strip('-').lower()
                if current_dataset not in data:
                    data[current_dataset] = []
                header_map = {}
                print(f"DEBUG: Identified dataset: {current_dataset}") # Added debug print
                continue
            if not current_dataset: continue
            if not header_map:
                header = [h.strip() for h in row]
                header_map = {name: idx for idx, name in enumerate(header)}
                print(f"DEBUG: Header Map for {current_dataset}: {header_map}") # Added debug print
                continue
            
            try:
                time = float(row[header_map['total_time']])
                dep_key = next((k for k in ['num_paths', 'dependencies', 'num_constraints', 'num_explanations'] if k in header_map), None)
                dependencies = float(row[header_map[dep_key]]) if dep_key else 0.0
                cells_key = next((k for k in ['mask_size', 'cells_deleted'] if k in header_map), None)
                cells = int(row[header_map[cells_key]]) if cells_key else 0
                mem_key = next((k for k in ['memory_overhead_bytes', 'memory_bytes'] if k in header_map), None)
                memory = float(row[header_map[mem_key]]) if mem_key else 0.0
                leakage = float(row[header_map['leakage']]) if 'leakage' in header_map else 0.0
                paths_blocked = float(row[header_map['paths_blocked']]) if 'paths_blocked' in header_map else 0.0
                
                if 'exponential_deletion_data' in file_path:
                    cells += 1
                
                parsed_record = {
                    'time': time, 'dependencies': dependencies, 'cells': cells,
                    'space_overhead': memory, 'leakage': leakage,
                    'paths_blocked': paths_blocked
                }
                data[current_dataset].append(parsed_record)
                # Print every record for now - very verbose!
                # print(f"DEBUG: Appended record for {current_dataset}: {parsed_record}")

            except (ValueError, IndexError):
                print(f"DEBUG: Skipping malformed row for {current_dataset}: {row}") # Added debug print
                continue
    return data

def calculate_aggregated_metrics(dataset_data):
    if not dataset_data: return None
    return {
        'total_time': sum(d['time'] for d in dataset_data),
        'space_overhead_sum': sum(d['space_overhead'] for d in dataset_data),
        'cells_sum': sum(d['cells'] for d in dataset_data),
        'dependencies': sum(d['dependencies'] for d in dataset_data),
        'leakage': sum(d['leakage'] for d in dataset_data),
        'paths_blocked': sum(d['paths_blocked'] for d in dataset_data)
    }

def create_star_plot(ax, metrics_data, dataset_name, max_values_all):
    categories = ['Mask Size', 'Leakage', 'Total Time', 'Memory', 'Total Paths']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] # Close the loop for plotting

    colors = ['#009E73', '#56B4E9', '#800000', '#F0E442', '#D55E00'] # High-contrast palette
    markers = ['o', 's', '^', 'D', 'P']
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    ax.set_rorigin(-0.25) # Original setting
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Draw axis lines
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], color="lightgrey", linewidth=0.7, alpha=0.7, zorder=1)

    # --- FIX: Draw polygonal grid lines with LOGARITHMIC labels ---
    grid_levels = [0.25, 0.5, 0.75, 1.0]
    for r in grid_levels:
        points = np.array([(r, angle) for angle in angles])
        ax.plot(points[:, 1], points[:, 0], color="lightgrey", linewidth=0.6, alpha=0.6, zorder=1)
        
        # Add labels to the first axis for scale
        # Un-log the value: value = (max_value + 1)^r - 1
        max_val_for_label = max_values_all[0]
        label_val = (max_val_for_label + 1)**r - 1
        
        label_text = f"{label_val:.0f}"
        if label_val < 10 and label_val > 0:
            label_text = f"{label_val:.1f}"

        ax.text(pi / 2, r, label_text, ha='center', va='center', fontsize=7, color='grey', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
    # --- End FIX ---

    # Plot data
    for idx, (baseline_name, metrics) in enumerate(metrics_data.items()):
        if not metrics: continue
        raw_values = [
            metrics.get('cells_sum', 0),
            metrics.get('leakage', 0),
            metrics.get('total_time', 0),
            metrics.get('space_overhead_sum', 0),
            metrics.get('dependencies', 0)
        ]
        
        # --- FIX: Use logarithmic normalization ---
        normalized_values = [np.log10(val + 1) / np.log10(max_val + 1) if max_val > 0 else 0 
                             for val, max_val in zip(raw_values, max_values_all)]
        # --- End FIX ---

        plot_values = np.append(normalized_values, normalized_values[0])
        
        color, marker, linestyle = colors[idx], markers[idx], linestyles[idx]

        ax.plot(angles, plot_values, linestyle=linestyle, linewidth=1.5, label=baseline_name, color=color, 
                markersize=7, marker=marker, markeredgewidth=1.5, markeredgecolor='white', zorder=3)
        ax.fill(angles, plot_values, alpha=0.08, color=color, zorder=2)

    # Place category labels
    for i, angle in enumerate(angles[:-1]):
        ax.text(angle, 1.15, categories[i], ha='center', va='center', fontsize=12)

    ax.set_ylim(0, 1.0)
    for spine in ax.spines.values(): 
        spine.set_visible(False)
    if 'polar' in ax.spines: 
        ax.spines['polar'].set_visible(False)

    ax.set_title(dataset_name.strip().title(), size=14, y=1.25)

if __name__ == '__main__':
    baseline_files = {
        'Baseline 3': 'baseline_deletion_3_data_v10.csv',
        'Exponential Deletion': 'exponential_deletion_data.csv',
        'Greedy Gumbel': 'greedy_gumbel_data.csv',
    }
    datasets_to_plot = ['airport', 'hospital', 'ncvoter', 'tax']
    
    all_data = {name: parse_csv_for_baselines(path) for name, path in baseline_files.items()}
    
    final_metrics = {ds: {name: calculate_aggregated_metrics(all_data[name].get(ds, [])) 
                        for name in baseline_files} for ds in datasets_to_plot}

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')

    for ax, dataset_name in zip(axes, datasets_to_plot):
        max_values_map = {cat: [] for cat in ['cells_sum', 'leakage', 'total_time', 'space_overhead_sum', 'dependencies']}
        if dataset_name in final_metrics and final_metrics[dataset_name]:
            for metrics in final_metrics[dataset_name].values():
                if metrics:
                    for cat in max_values_map:
                        max_values_map[cat].append(metrics.get(cat, 0))

        max_values = [
            max(max_values_map['cells_sum']) * 1.2 if max_values_map['cells_sum'] else 1.0,
            max(max_values_map['leakage']) * 1.2 if max_values_map['leakage'] else 1.0,
            max(max_values_map['total_time']) * 1.2 if max_values_map['total_time'] else 1.0,
            max(max_values_map['space_overhead_sum']) * 1.2 if max_values_map['space_overhead_sum'] else 1.0,
            max(max_values_map['dependencies']) * 1.2 if max_values_map['dependencies'] else 1.0,
        ]

        if dataset_name in final_metrics and final_metrics[dataset_name]:
            create_star_plot(ax, final_metrics[dataset_name], dataset_name, max_values)
        else:
            ax.set_title(dataset_name.strip().title() + "\n(No Data)", size = 14, pad = 20)
            ax.set_xticks([]);
            ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

    first_ax_with_data = next((ax for ax in axes if ax.get_legend_handles_labels()[0]), None)

    if first_ax_with_data:
        handles, labels = first_ax_with_data.get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'upper center', ncol = 3, bbox_to_anchor = (0.5, 1.05), fontsize = 14)

    plt.tight_layout(rect = [0, 0, 1, 0.95])
    
    output_pdf = 'star_plots_B3_Exp_Gumbel.pdf'
    plt.savefig(output_pdf, bbox_inches = 'tight')
    plt.close(fig)
    
    print(f"Comparison star plot saved to {output_pdf}")