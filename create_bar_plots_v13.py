#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Global matplotlib + LaTeX configuration (ALL TEXT IN })
# ------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.variant": "small-caps",
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": 13,
})

# ------------------------------------------------------------------
# CSV Parsing
# ------------------------------------------------------------------
def parse_csv_for_bar_plots(file_path):
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping: {file_path}")
        return data

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        current_dataset = None
        header_map = {}

        for row in reader:
            if not row:
                continue

            if row[0].startswith('-----'):
                current_dataset = row[0].strip('-').lower()
                data.setdefault(current_dataset, [])
                header_map = {}
                continue

            if not current_dataset:
                continue

            if not header_map:
                header_map = {h.strip(): i for i, h in enumerate(row)}
                continue

            try:
                init_time = float(row[header_map['init_time']]) if 'init_time' in header_map else 0.0
                model_time = float(row[header_map['model_time']]) if 'model_time' in header_map else 0.0
                del_time = float(row[header_map['del_time']]) if 'del_time' in header_map else 0.0

                mem_key = next((k for k in ['memory_overhead_bytes', 'memory_bytes'] if k in header_map), None)
                memory = float(row[header_map[mem_key]]) if mem_key else 0.0

                data[current_dataset].append({
                    'init_time': init_time,
                    'model_time': model_time,
                    'del_time': del_time,
                    'memory': memory
                })
            except Exception:
                continue

    return data

# ------------------------------------------------------------------
# Averaging
# ------------------------------------------------------------------
def calculate_average_metrics(dataset_data):
    if not dataset_data:
        return dict(init_time_avg=0, model_time_avg=0, del_time_avg=0, memory_avg=0)

    n = len(dataset_data)
    return {
        'init_time_avg': sum(d['init_time'] for d in dataset_data) / n,
        'model_time_avg': sum(d['model_time'] for d in dataset_data) / n,
        'del_time_avg': sum(d['del_time'] for d in dataset_data) / n,
        'memory_avg': sum(d['memory'] for d in dataset_data) / n,
    }

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def create_combined_bar_plot(all_metrics, datasets, baselines):
    fig, axes = plt.subplots(1, len(datasets), figsize=(20, 8))
    axes = axes.flatten()

    colors = {
        'Instantiation Time': '#56B4E9',
        'Modeling Time': '#D55E00',
        'Update to NULL Time': '#009E73',
        'Model Size': 'brown'
    }

    label_map = {
        "Greedy Gumbel": r"DelGum",
        "Exponential Deletion": r"DelExp",
        "Baseline 3": r"DelMin",
        # "2-Phase Deletion": r"Del2Ph",
    }

    # Fixed order of baselines for plotting the bars
    fixed_plot_order = ['Baseline 3', 'Exponential Deletion', 'Greedy Gumbel']

    for i, dataset in enumerate(datasets):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        # Reorder dataset_data according to fixed_plot_order
        dataset_data = {b: all_metrics[b].get(dataset, {}) for b in fixed_plot_order}

        labels = [label_map[b] for b in fixed_plot_order]

        # Pull metrics in the same fixed order
        init = np.array([dataset_data[b]['init_time_avg'] for b in fixed_plot_order])
        model = np.array([dataset_data[b]['model_time_avg'] for b in fixed_plot_order])
        delete = np.array([dataset_data[b]['del_time_avg'] for b in fixed_plot_order])
        memory = np.array([dataset_data[b]['memory_avg'] for b in fixed_plot_order]) / (1024 ** 2)

        x = np.arange(len(labels))
        w = 0.35
        eps = 1e-9

        # Plot bars in fixed order
        ax1.bar(x - w / 2, init + eps, w, bottom = 0, color = colors['Instantiation Time'],
                label = "Instantiation Time")
        ax1.bar(x - w / 2, model + eps, w, bottom = init + eps, color = colors['Modeling Time'],
                label = "Modeling Time")
        ax1.bar(x - w / 2, delete + eps, w, bottom = init + model + eps,
                color = colors['Update to NULL Time'], label = "Update to Null Time")

        ax2.bar(x + w / 2, memory + eps, w, color = colors['Model Size'], label = "Model Size")

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=0)

        ax1.set_ylabel(r"Average Time (s)")
        ax2.set_ylabel(r"Model Size (MB)")

        if dataset == "ncvoter":
            title = r"NCVoter"
        else:
            title = rf"{dataset.title()}"

        ax1.set_title(title)

        ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Legend
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    order = [
        r"Instantiation Time",
        r"Modeling Time",
        r"Update to Null Time",
        r"Model Size",
    ]

    ordered_handles = [handles[labels.index(l)] for l in order if l in labels]

    fig.legend(
        ordered_handles,
        order,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.99)
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    file_paths = {
        'Greedy Gumbel': 'delgum_data_standardized_v2.csv',
        'Exponential Deletion': 'delexp_data_standardized_v2.csv',
        'Baseline 3': 'delmin_data_standardized_v2.csv',
        # '2-Phase Deletion': '2phase_deletion_data_v12.csv',
    }

    datasets = ['airport', 'hospital', 'ncvoter', 'onlineretail', 'adult']

    all_data = {k: parse_csv_for_bar_plots(v) for k, v in file_paths.items()}
    all_metrics = {
        b: {ds: calculate_average_metrics(all_data[b].get(ds, [])) for ds in datasets}
        for b in file_paths
    }

    baselines_ordered = [
        'Greedy Gumbel',
        'Exponential Deletion',
        'Baseline 3',
        '2-Phase Deletion'
    ]

    fig = create_combined_bar_plot(all_metrics, datasets, baselines_ordered)
    fig.savefig("bar_plot_AllBaselines_NoTax_v13.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved bar_plot_AllBaselines_NoTax_v13.pdf")


if __name__ == "__main__":
    main()
