import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def parse_csv_for_bar_plots(file_path):
    """Robust parser for all baseline and new algorithm CSV formats."""
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
                init_time = float(row[header_map.get('init_time', -1)]) if 'init_time' in header_map and header_map.get('init_time', -1) != -1 and header_map['init_time'] < len(row) else 0.0
                model_time = float(row[header_map.get('model_time', -1)]) if 'model_time' in header_map and header_map.get('model_time', -1) != -1 and header_map['model_time'] < len(row) else 0.0
                del_time = float(row[header_map.get('del_time', -1)]) if 'del_time' in header_map and header_map.get('del_time', -1) != -1 and header_map['del_time'] < len(row) else 0.0
                
                mem_key = next((k for k in ['memory_overhead_bytes', 'memory_bytes'] if k in header_map), None)
                memory = float(row[header_map[mem_key]]) if mem_key and header_map[mem_key] < len(row) else 0.0
                
                data[current_dataset].append({
                    'init_time': init_time, 'model_time': model_time,
                    'del_time': del_time, 'memory': memory,
                })
            except (ValueError, IndexError, KeyError) as e:
                print(f"Skipping row in {file_path} due to error: {e}. Row: {row}")
                continue
    return data

def calculate_sum_metrics(dataset_data):
    """Calculate summed metrics for a dataset."""
    if not dataset_data: return None
    return {
        'init_time_sum': sum(d['init_time'] for d in dataset_data),
        'model_time_sum': sum(d['model_time'] for d in dataset_data),
        'del_time_sum': sum(d['del_time'] for d in dataset_data),
        'memory_sum': sum(d['memory'] for d in dataset_data),
    }

def create_combined_bar_plot(all_metrics, datasets, baselines):
    """Create a single figure with a 1x4 grid of vertical stacked bar plots."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(24, 8), squeeze=False) # 1 row for horizontal layout
    axes = axes.flatten()
    fig.suptitle('Performance Comparison Across Datasets', fontsize=18, y=1.02)
    
    colors = {'Init Time': '#009E73', 'Model Time': '#56B4E9', 'Delete Time': '#800000', 'Memory': '#D55E00'} # High-contrast palette

    for i, dataset in enumerate(datasets):
        ax1 = axes[i]
        dataset_data = {b: all_metrics.get(b, {}).get(dataset, {}) for b in baselines}
        
        labels = [b.replace('_', ' ').title() for b in baselines]
        init_times = np.array([dataset_data[b].get('init_time_sum', 0) for b in baselines])
        model_times = np.array([dataset_data[b].get('model_time_sum', 0) for b in baselines])
        del_times = np.array([dataset_data[b].get('del_time_sum', 0) for b in baselines])
        memory_values_kb = np.array([dataset_data[b].get('memory_sum', 0) / 1024 for b in baselines])

        x = np.arange(len(labels))
        width = 0.35
        epsilon = 1e-6

        # --- Time Bars (Left Y-axis, Stacked) ---
        ax1.bar(x - width/2, init_times + epsilon, width, label='Init Time', color=colors['Init Time'])
        ax1.bar(x - width/2, model_times + epsilon, width, bottom=init_times + epsilon, label='Model Time', color=colors['Model Time'])
        ax1.bar(x - width/2, del_times + epsilon, width, bottom=init_times + model_times + epsilon, label='Delete Time', color=colors['Delete Time'])

        ax1.set_ylabel('Average Time (s) - Log Scale', fontsize=12, fontweight='bold')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
        ax1.set_title(dataset.title(), fontsize=14, fontweight='bold')

        # --- Memory Bars (Right Y-axis) ---
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, memory_values_kb + epsilon, width, label='Memory', color=colors['Memory'])
        ax2.set_ylabel('Average Memory (KB) - Log Scale', fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelsize=10)
        
        ax1.grid(axis='y', linestyle='--', alpha=0.7, which="both")

    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig

def main():
    file_paths = {
        'Baseline 3': 'baseline_deletion_3_data_v10.csv',
        'Exponential Deletion': 'exponential_deletion_data.csv',
        'Greedy Gumbel': 'greedy_gumbel_data.csv',
    }
    datasets = ['airport', 'hospital', 'ncvoter', 'tax']
    
    all_data = {name: parse_csv_for_bar_plots(path) for name, path in file_paths.items()}
    
    if not any(all_data.values()):
        print("No data found to plot. Check CSV file paths and content.")
        return

    all_metrics = {baseline: {ds: calculate_sum_metrics(all_data[baseline].get(ds, [])) 
                              for ds in datasets} 
                   for baseline in file_paths.keys()}

    if all_metrics:
        fig = create_combined_bar_plot(all_metrics, datasets, list(file_paths.keys()))
        output_file = 'bar_plot_combined_comparison_log.pdf'
        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        print(f"Saved combined bar plot: {output_file}")
        plt.close(fig)

if __name__ == '__main__':
    main()
