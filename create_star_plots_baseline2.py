import csv
from math import pi, cos, sin
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

def parse_csv_for_baseline(file_path):
    """Parse a single baseline CSV and return a dictionary keyed by dataset."""
    data = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        current_dataset = None
        header_map = {}
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
                if 'total_time' not in header_map: continue
                if 'num_constraints' in header_map:
                    header_map['dependencies'] = header_map['num_constraints']
                elif 'num_explanations' in header_map:
                    header_map['dependencies'] = header_map['num_explanations']
                continue
            try:
                dependencies_val = float(row[header_map['dependencies']]) if 'dependencies' in header_map else 0
                memory_val = float(row[header_map['memory_bytes']]) if 'memory_bytes' in header_map else 0
                depth_val = float(row[header_map['max_depth']]) if 'max_depth' in header_map else 1.0

                data[current_dataset].append({
                    'time': float(row[header_map['total_time']]),
                    'dependencies': dependencies_val,
                    'cells': int(row[header_map['cells_deleted']]),
                    'space_overhead': memory_val,
                    'depth': depth_val
                })
            except (ValueError, IndexError, KeyError):
                continue
    return data

def calculate_metrics(dataset_data):
    """Calculate aggregated metrics for a list of data entries."""
    if not dataset_data: return None
    return {
        'total_time': sum(d['time'] for d in dataset_data),
        'space_overhead_sum': sum(d['space_overhead'] for d in dataset_data),
        'cells_sum': sum(d['cells'] for d in dataset_data),
        'dependencies': sum(d['dependencies'] for d in dataset_data),
        'depth': sum(d['depth'] for d in dataset_data)
    }

def draw_polygon_grid(ax, angles, num_levels=5):
    """Draw concentric polygons as grid lines for a cartesian plot."""
    for i in range(1, num_levels + 1):
        radius = i / num_levels
        # Correctly calculate cartesian coordinates from angles
        x_coords = [radius * cos(angle - pi / 2) for angle in angles]
        y_coords = [radius * sin(angle - pi / 2) for angle in angles]
        # Append the first point to close the polygon
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        ax.plot(x_coords, y_coords, 'k-', linewidth=0.7, alpha=0.4, zorder=1)

def create_star_plot(ax, metrics_data, dataset_name, max_values_all):
    """
    Create a star plot on a given matplotlib axis, matching the original PNG style.
    """
    categories = ['Mask size', 'Leakage', 'Total time', 'Memory', 'Paths Blocked']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']

    ax.set_aspect('equal')

    # Draw the polygon grid
    draw_polygon_grid(ax, angles, num_levels=5)

    # Draw axis lines and scale labels
    for i, angle in enumerate(angles):
        # Axis lines
        ax.plot([0, 1.05 * cos(angle - pi/2)], [0, 1.05 * sin(angle - pi/2)], 'k-', linewidth=1.0, alpha=1.0, zorder=1)
        # Scale labels
        for tick in range(1, 4):
            radius = tick / 3
            tick_value = max_values_all[i] * radius
            label_text = f'{tick_value:.1f}' if tick_value < 10 else f'{tick_value:.0f}'
            ax.text(radius * cos(angle - pi/2) * 0.94, radius * sin(angle - pi/2) * 0.94, label_text, 
                    fontsize=9, ha='center', va='center', zorder=2,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # Plot data for each baseline
    for idx, (baseline_name, metrics) in enumerate(metrics_data.items()):
        if not metrics: continue
        raw_values = [
            metrics['cells_sum'], metrics['dependencies'], metrics['total_time'],
            metrics['space_overhead_sum'], metrics['depth']
        ]
        
        normalized_values = [val / max_val if max_val > 0 else 0 for val, max_val in zip(raw_values, max_values_all)]
        plot_values = normalized_values + normalized_values[:1]
        plot_angles = angles + angles[:1]
        
        x_coords = [r * cos(a - pi/2) for r, a in zip(plot_values, plot_angles)]
        y_coords = [r * sin(a - pi/2) for r, a in zip(plot_values, plot_angles)]
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        ax.plot(x_coords, y_coords, linestyle=linestyle, linewidth=1.5,
                label=baseline_name, color=color, markersize=7, marker=marker,
                markeredgewidth=1.5, markeredgecolor='white', markerfacecolor=color, zorder=3)
        ax.fill(x_coords, y_coords, alpha=0.08, color=color, zorder=2)

    # Category (Axis) Labels
    for i, (angle, cat) in enumerate(zip(angles, categories)):
        label_dist = 1.2
        x = label_dist * cos(angle - pi/2)
        y = label_dist * sin(angle - pi/2)
        ha = 'center' if abs(x) < 0.1 else 'left' if x > 0 else 'right'
        va = 'center' if abs(y) < 0.1 else 'bottom' if y > 0 else 'top'
        ax.text(x, y, cat, fontsize=12, ha=ha, va=va, zorder=4)

    # Cleanup
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(dataset_name.strip().title(), size=14, pad=20)


if __name__ == '__main__':
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is required. Please install it.")
        sys.exit(1)

    baseline_files = {
        'Baseline 1': 'baseline_deletion_1_data_v7.csv',
        'Baseline 2': 'baseline_deletion_2_data_v7.csv',
        'Baseline 3': 'baseline_deletion_3_data_v7.csv'
    }

    datasets_to_plot = ['airport', 'hospital', 'ncvoter', 'tax']
    
    all_baseline_data = {}
    for name, path in baseline_files.items():
        try:
            all_baseline_data[name] = parse_csv_for_baseline(path)
        except FileNotFoundError:
            print(f"Warning: Data file not found for {name} at {path}. Skipping.")
    
    if len(all_baseline_data) < 3:
        print("Error: Not all baseline data files were found. Exiting.")
        sys.exit(1)

    final_metrics = {dataset: {} for dataset in datasets_to_plot}
    for baseline_name, datasets in all_baseline_data.items():
        for dataset_name, data_entries in datasets.items():
            if dataset_name in datasets_to_plot:
                final_metrics[dataset_name][baseline_name] = calculate_metrics(data_entries)

    all_values_for_max = {cat: [] for cat in ['cells_sum', 'dependencies', 'total_time', 'space_overhead_sum', 'depth']}
    for dataset in datasets_to_plot:
        for baseline in final_metrics[dataset].values():
            if baseline:
                all_values_for_max['cells_sum'].append(baseline['cells_sum'])
                all_values_for_max['dependencies'].append(baseline['dependencies'])
                all_values_for_max['total_time'].append(baseline['total_time'])
                all_values_for_max['space_overhead_sum'].append(baseline['space_overhead_sum'])
                all_values_for_max['depth'].append(baseline['depth'])

    max_values = [
        max(all_values_for_max['cells_sum']) * 1.2 or 1.0,
        max(all_values_for_max['dependencies']) * 1.2 or 1.0,
        max(all_values_for_max['total_time']) * 1.2 or 1.0,
        max(all_values_for_max['space_overhead_sum']) * 1.2 or 1.0,
        max(all_values_for_max['depth']) * 1.2 or 1.0,
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.patch.set_facecolor('white')

    for ax, dataset_name in zip(axes, datasets_to_plot):
        create_star_plot(ax, final_metrics[dataset_name], dataset_name, max_values)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98), fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    output_pdf = 'star_plots_comparison_horizontal.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)
    
    print(f"All star plots saved to {output_pdf}")