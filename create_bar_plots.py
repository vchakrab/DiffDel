import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.backends.backend_pdf import PdfPages

def parse_csv_for_bar_plots(file_path):
    """
    Parse CSV files, extracting component times and memory for bar plots.
    This version correctly looks for 'init_time', 'model_time', 'del_time'.
    """
    data = {}
    current_dataset = None
    header_map = {}

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            if row[0].startswith('-----'):
                current_dataset = row[0].strip('-')
                data[current_dataset] = []
                header_map = {}
                continue
            if not current_dataset: continue
            if not header_map:
                header = [h.strip() for h in row]
                header_map = {name: idx for idx, name in enumerate(header)}
                continue

            try:
                # Use the correct headers from v7 files
                init_time = float(row[header_map['init_time']]) if 'init_time' in header_map else 0
                model_time = float(row[header_map['model_time']]) if 'model_time' in header_map else 0
                del_time = float(row[header_map['del_time']]) if 'del_time' in header_map else 0
                memory = float(row[header_map['memory_bytes']]) if 'memory_bytes' in header_map else 0
                total_time = float(row[header_map['total_time']]) if 'total_time' in header_map else (init_time + model_time + del_time) # Fallback to sum if total_time not explicit

                data[current_dataset].append({
                    'init_time': init_time,
                    'model_time': model_time,
                    'del_time': del_time,
                    'memory': memory,
                    'total_time': total_time
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
        'total_time_sum': sum(d['total_time'] for d in dataset_data)
    }

def create_combined_bar_plot(all_metrics, datasets):
    """Create a single figure with a 1x4 grid of stacked bar plots."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    axes = axes.flatten()
    fig.suptitle('Performance Comparison Across Datasets', fontsize=18, y=1.02)

    for i, dataset_name in enumerate(datasets):
        ax1 = axes[i]
        metrics_data = {baseline: all_metrics[baseline].get(dataset_name, {}) for baseline in all_metrics}

        labels = list(metrics_data.keys())
        init_times = [m.get('init_time_sum', 0) for m in metrics_data.values()]
        model_times = [m.get('model_time_sum', 0) for m in metrics_data.values()]
        del_times = [m.get('del_time_sum', 0) for m in metrics_data.values()]
        memory_values = [m.get('memory_sum', 0) for m in metrics_data.values()]
        total_times_csv = [m.get('total_time_sum', 0) for m in metrics_data.values()]

        x = np.arange(len(labels))
        width = 0.35

        bottom_init_model = np.array(init_times) + np.array(model_times)
        bottom_all_components = bottom_init_model + np.array(del_times)
        
        overhead_times = np.maximum(0, np.array(total_times_csv) - bottom_all_components)
        
        # --- Time Bars (Left Y-axis) ---
        ax1.bar(x - width/2, init_times, width, label='Init Time', color='#2E86AB')
        ax1.bar(x - width/2, model_times, width, bottom=init_times, label='Model Time', color='#F18F01')
        ax1.bar(x - width/2, del_times, width, bottom=bottom_init_model, label='Delete Time', color='#A23B72')
        ax1.bar(x - width/2, overhead_times, width, bottom=bottom_all_components, label='Other Overhead', color='#CCCCCC')

        ax1.set_ylabel('Total Time (s)', fontsize=12)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
        ax1.set_title(dataset_name.strip().title(), fontsize=14)

        ax2 = ax1.twinx()
        ax2.bar(x + width/2, memory_values, width, label='Memory', color='#34A853')
        ax2.set_ylabel('Total Memory (Bytes)', fontsize=12)
        ax2.tick_params(axis='y', labelsize=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

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
        'Baseline 1': 'baseline_deletion_1_data_v7.csv',
        'Baseline 2': 'baseline_deletion_2_data_v7.csv',
        'Baseline 3': 'baseline_deletion_3_data_v7.csv'
    }

    all_data = {}
    for baseline_name, file_path in file_paths.items():
        try:
            data = parse_csv_for_bar_plots(file_path)
            all_data[baseline_name] = data
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}.")
            continue
    
    if not all_data:
        print("No data to plot.")
        return

    common_datasets = sorted(list(set.intersection(*[set(data.keys()) for data in all_data.values()])))
    
    all_metrics = {baseline: {} for baseline in file_paths.keys()}
    for dataset_name in common_datasets:
        for baseline_name in file_paths.keys():
            if dataset_name in all_data[baseline_name]:
                metrics = calculate_sum_metrics(all_data[baseline_name][dataset_name])
                if metrics:
                    all_metrics[baseline_name][dataset_name] = metrics

    if all_metrics:
        fig = create_combined_bar_plot(all_metrics, common_datasets)
        output_file = 'bar_plot_combined_comparison.pdf'
        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        print(f"Saved combined bar plot: {output_file}")
        plt.close(fig)

if __name__ == '__main__':
    main()
