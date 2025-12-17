import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --- New, more robust parser adapted from the star plot script ---
def parse_csv_for_bar_chart(file_path):
    data = {}
    if not file_path or not os.path.exists(file_path):
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
                # Use .get() to handle missing columns gracefully
                init_time = float(row[header_map.get('init_time', -1)])
                model_time = float(row[header_map.get('model_time', -1)])
                del_time = float(row[header_map.get('del_time', -1)])
                
                mem_key = next((k for k in ['memory_overhead_bytes', 'memory_bytes'] if k in header_map), None)
                memory = float(row[header_map[mem_key]]) if mem_key else 0.0

                data[current_dataset].append({
                    'init_time': init_time,
                    'model_time': model_time,
                    'del_time': del_time,
                    'memory': memory
                })
            except (ValueError, IndexError):
                continue
    return data

# --- New aggregation function ---
def aggregate_bar_chart_data(all_data):
    aggregated = {}
    for baseline, baseline_data in all_data.items():
        for dataset, records in baseline_data.items():
            if not records: continue
            if dataset not in aggregated:
                aggregated[dataset] = {}
            
            aggregated[dataset][baseline] = {
                'avg_init_time': np.mean([r['init_time'] for r in records]),
                'avg_model_time': np.mean([r['model_time'] for r in records]),
                'avg_del_time': np.mean([r['del_time'] for r in records]),
                'avg_memory': np.mean([r['memory'] for r in records])
            }
    return aggregated

# --- New plotting function ---
def create_comparison_bar_chart(aggregated_data, baselines, datasets):
    # Colorblind-friendly palette: Blue, Green, Orange/Vermillion
    colors = ['#0072B2', '#009E73', '#D55E00']
    
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 6 * len(datasets)), squeeze=False)
    axes = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax1 = axes[i]
        dataset_data = aggregated_data.get(dataset, {})
        
        n_baselines = len(baselines)
        bar_width = 0.35
        index = np.arange(n_baselines)

        # --- Time Bars (Left Axis, Stacked) ---
        ax1.set_ylabel('Average Time (s) - Log Scale', fontweight='bold')
        ax1.set_yscale('log')
        
        init_times = [dataset_data.get(b, {}).get('avg_init_time', 0) for b in baselines]
        model_times = [dataset_data.get(b, {}).get('avg_model_time', 0) for b in baselines]
        del_times = [dataset_data.get(b, {}).get('avg_del_time', 0) for b in baselines]

        # Add a small epsilon to avoid log(0) issues
        epsilon = 1e-6
        p1 = ax1.bar(index - bar_width/2, [t + epsilon for t in init_times], bar_width, label='Init Time', color=colors[0], alpha=0.4)
        p2 = ax1.bar(index - bar_width/2, [t + epsilon for t in model_times], bar_width, bottom=[i + epsilon for i in init_times], label='Model Time', color=colors[0], alpha=0.6)
        p3 = ax1.bar(index - bar_width/2, [t + epsilon for t in del_times], bar_width, bottom=[i + m + epsilon for i, m in zip(init_times, model_times)], label='Deletion Time', color=colors[0], alpha=0.8)

        # --- Memory Bars (Right Axis) ---
        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Memory (Bytes) - Log Scale', fontweight='bold')
        ax2.set_yscale('log')
        
        memory_vals = [dataset_data.get(b, {}).get('avg_memory', 0) for b in baselines]
        p4 = ax2.bar(index + bar_width/2, [m + epsilon for m in memory_vals], bar_width, label='Memory', color=colors[2])
        
        # --- Formatting ---
        ax1.set_title(f'Comparison for {dataset.title()} Dataset', fontsize=14, fontweight='bold')
        ax1.set_xticks(index)
        ax1.set_xticklabels([b.replace('_', ' ').title() for b in baselines], rotation=0, ha='center')
        
        # Set y-axis limits to be slightly larger than max values
        max_time = max([i + m + d for i, m, d in zip(init_times, model_times, del_times)])
        max_mem = max(memory_vals)
        if max_time > 0: ax1.set_ylim(bottom=1e-5, top=max_time * 10)
        if max_mem > 0: ax2.set_ylim(bottom=1, top=max_mem * 10)
        
        ax1.grid(True, which="both", ls="--", alpha=0.5, axis='y')

    # Create a single legend for the whole figure
    handles = [p1, p2, p3, p4]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98), fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# --- New main function ---
def main():
    baseline_files = {
        'Baseline 3': 'baseline_deletion_3_data_v10.csv',
        'Exponential Deletion': 'exponential_deletion_data.csv',
        'Greedy Gumbel': 'greedy_gumbel_data.csv',
    }
    datasets_to_plot = ['airport', 'hospital', 'ncvoter', 'tax']
    
    all_data = {name: parse_csv_for_bar_chart(path) for name, path in baseline_files.items()}
    
    aggregated_data = aggregate_bar_chart_data(all_data)
    
    if not aggregated_data:
        print("Error: No data was aggregated. Check if CSV files exist and are valid.")
        return

    fig = create_comparison_bar_chart(aggregated_data, list(baseline_files.keys()), datasets_to_plot)
    
    output_pdf = 'bar_plot_comparison_log.pdf'
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comparison bar plot saved to {output_pdf}")

if __name__ == '__main__':
    import os
    main()