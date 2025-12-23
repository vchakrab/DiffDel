import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

def parse_tradeoff_data(file_path):
    """Parses a CSV file to extract instantiated cells, mask size, and path count."""
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping: {file_path}")
        return data
        
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
                continue
            if not current_dataset: continue
            if not header_map:
                header = [h.strip() for h in row]
                header_map = {name: idx for idx, name in enumerate(header)}
                continue
            
            try:
                inst_key = next((k for k in ['num_instantiated_cells', 'cells'] if k in header_map), None)
                mask_key = next((k for k in ['mask_size', 'cells_deleted'] if k in header_map), None)
                paths_key = next((k for k in ['num_paths', 'dependencies', 'num_constraints', 'num_explanations'] if k in header_map), None)

                instantiated = float(row[header_map[inst_key]]) if inst_key and header_map[inst_key] < len(row) else 0.0
                mask_size = int(row[header_map[mask_key]]) if mask_key and header_map[mask_key] < len(row) else 0
                paths = int(row[header_map[paths_key]]) if paths_key and header_map[paths_key] < len(row) else 0
                
                if 'exponential' in file_path.lower() or 'gumbel' in file_path.lower():
                    mask_size += 1

                data[current_dataset].append({
                    'instantiated': instantiated,
                    'mask_size': mask_size,
                    'paths': paths,
                })
            except (ValueError, IndexError, KeyError):
                continue
    return data

def create_tradeoff_plot(all_data, baselines, datasets):
    """Creates a scatter plot of every individual data point, colored by algorithm, shaped by dataset."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = {'Baseline 3 (DelMin)': '#009E73', 'Exponential Deletion': '#D55E00', 'Greedy Gumbel': '#56B4E9'}

    for baseline in baselines:
        for dataset in datasets:
            records = all_data.get(baseline, {}).get(dataset, [])
            if not records:
                continue

            x_vals = [r['instantiated'] for r in records]
            y_vals = [r['mask_size'] for r in records]
            sizes = [np.sqrt(r['paths']) * 2 + 15 for r in records] # Scale by sqrt(paths) for better visual range
            
            ax.scatter(x_vals, y_vals, s=sizes, color=colors[baseline], 
                       marker='o', alpha=0.6, edgecolors='black', linewidth=0.5,
                       label=f"{baseline} - {dataset.title()}" if dataset == datasets[0] else "") # Label only once per baseline

    ax.set_xlabel('Instantiated Cells', fontsize=14, fontweight='bold') # Removed Log Scale
    ax.set_ylabel('Mask Size (Cells Deleted)', fontsize=14, fontweight='bold')
    # ax.set_xscale('log') # Removed log scale
    ax.set_title('Deletion Efficiency: Instantiated Cells vs. Mask Size (All Runs)', fontsize=16, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # --- Create Custom Legend (only for colors of algorithms) ---
    legend_elements = []
    for baseline, color in colors.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=baseline, 
                                      markerfacecolor=color, markersize=12, markeredgecolor='black'))
    
    ax.legend(handles=legend_elements, title="Algorithms", fontsize=12, loc='upper left')

    return fig

def main():
    file_paths = {
        'Baseline 3 (DelMin)': 'baseline_deletion_3_data_v11.csv',
        'Exponential Deletion': 'exponential_deletion_data_v11.csv',
        'Greedy Gumbel': 'greedy_gumbel_data_v11.csv',
    }
    datasets = ['airport', 'hospital', 'ncvoter', 'tax']
    
    all_data = {name: parse_tradeoff_data(path) for name, path in file_paths.items()}
    
    if not any(all_data.values()):
        print("Error: No data was found. Check CSV file paths and content.")
        return

    fig = create_tradeoff_plot(all_data, list(file_paths.keys()), datasets)
    
    output_pdf = 'tradeoff_plot_all_points_final.pdf'
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Tradeoff plot saved to {output_pdf}")

if __name__ == '__main__':
    main()