import csv
from math import pi, cos, sin


try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def parse_csv_file(file_path):
    """
    Parse the CSV file with section headers using the NEW column format:
    attribute,time,dependencies,cells,depth,space_overhead(B)
    """
    data = {}
    current_dataset = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header
            if line.startswith('-----') and line.endswith('-----'):
                current_dataset = line.strip('-')
                data[current_dataset] = []
            elif current_dataset and ',' in line:
                # Skip header row for the NEW format
                if line.startswith('attribute,time,dependencies,cells,depth,space_overhead(B)'):
                    continue

                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        attribute = parts[0]
                        time = float(parts[1])
                        dependencies = float(parts[2])
                        cells = int(parts[3])
                        depth = float(parts[4])
                        space_overhead = float(parts[5])

                        data[current_dataset].append({
                            'attribute': attribute,
                            'time': time,
                            'dependencies': dependencies,
                            'cells': cells,
                            'depth': depth,
                            'space_overhead': space_overhead
                        })
                    except (ValueError, IndexError):
                        continue

    return data


def calculate_metrics(dataset_data):
    """Calculate aggregated metrics (averages) for a dataset"""
    if not dataset_data:
        return None

    attribute = dataset_data[0]['attribute']

    total_times = [d['time'] for d in dataset_data]
    space_overheads = [d['space_overhead'] for d in dataset_data]
    cells_list = [d['cells'] for d in dataset_data]
    dependencies_list = [d['dependencies'] for d in dataset_data]
    depth_list = [d['depth'] for d in dataset_data]

    space_overhead_sum = sum(space_overheads)
    if space_overhead_sum < 0:
        space_overhead_sum = 0.0

    metrics = {
        'attribute': attribute,
        'total_time': sum(total_times) / len(total_times),
        'space_overhead_avg': sum(space_overheads) / len(space_overheads),
        'space_overhead_sum': space_overhead_sum,
        'cells_avg': sum(cells_list) / len(cells_list),
        'dependencies': sum(dependencies_list) / len(dependencies_list),
        'depth': sum(depth_list) / len(depth_list)
    }

    return metrics


def draw_polygon_grid(ax, angles, num_levels = 5):
    """Draw concentric polygons as grid lines"""
    for i in range(1, num_levels + 1):
        radius = i / num_levels
        x_coords = [radius * cos(angle - pi / 2) for angle in angles]
        y_coords = [radius * sin(angle - pi / 2) for angle in angles]
        ax.plot(x_coords, y_coords, 'k-', linewidth = 0.5, alpha = 0.15, zorder = 1)


def create_star_plot(metrics_data, dataset_name):
    """
    Create a star plot in P2E2 paper style - clean, minimal, publication-ready.
    """
    # Define the 5 axes
    categories = ['Total Time', 'Space Overhead', 'Cells', 'Dependencies', 'Depth']
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Collect all raw values from all baselines
    all_raw_values = {cat: [] for cat in categories}
    metrics_list = []

    for baseline_name, metrics in metrics_data.items():
        raw_values = [
            metrics['total_time'],
            metrics['space_overhead_sum'],
            metrics['cells_avg'],
            metrics['dependencies'],
            metrics['depth']
        ]
        metrics_list.append((baseline_name, raw_values))

        for i, cat in enumerate(categories):
            all_raw_values[cat].append(raw_values[i])

    # Calculate max value for each axis
    max_values = []
    for cat in categories:
        max_val_for_axis = max(all_raw_values[cat]) if all_raw_values[cat] else 0.1
        max_values.append(max(max_val_for_axis * 1.2, 0.1))

    # Create plot with P2E2 style
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.set_aspect('equal')

    # Draw subtle polygon grid
    draw_polygon_grid(ax, angles, num_levels = 5)

    # Draw axis lines - thin and subtle
    for i, angle in enumerate(angles[:-1]):
        x_line = [0, 1.05 * cos(angle - pi / 2)]
        y_line = [0, 1.05 * sin(angle - pi / 2)]
        ax.plot(x_line, y_line, 'k-', linewidth = 0.8, alpha = 0.4, zorder = 1)

        # Add scale labels along each axis
        num_ticks = 3
        for tick in range(1, num_ticks + 1):
            tick_radius = tick / num_ticks
            tick_value = max_values[i] * tick_radius

            tick_x = tick_radius * cos(angle - pi / 2) * 0.88
            tick_y = tick_radius * sin(angle - pi / 2) * 0.88

            # Format label
            if tick_value >= 100:
                label_text = f'{tick_value:.0f}'
            elif tick_value >= 10:
                label_text = f'{tick_value:.1f}'
            else:
                label_text = f'{tick_value:.2f}'

            ax.text(tick_x, tick_y, label_text,
                    fontsize = 8, ha = 'center', va = 'center',
                    bbox = dict(boxstyle = 'round,pad=0.2', facecolor = 'white',
                                edgecolor = 'none', alpha = 0.7),
                    zorder = 2)

    # P2E2-style colors - professional and distinct
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']

    # Plot each baseline - thin lines, clear markers
    for idx, (baseline_name, raw_values) in enumerate(metrics_list):
        normalized_values = [v / m for v, m in zip(raw_values, max_values)]
        normalized_values_plot = normalized_values + [normalized_values[0]]

        x_coords = [r * cos(angle - pi / 2) for r, angle in zip(normalized_values_plot, angles)]
        y_coords = [r * sin(angle - pi / 2) for r, angle in zip(normalized_values_plot, angles)]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        # Thin lines with clear markers
        ax.plot(x_coords, y_coords, linestyle = linestyle, linewidth = 1.5,
                label = baseline_name, color = color, markersize = 7,
                marker = marker, markeredgewidth = 1.5, markeredgecolor = 'white',
                markerfacecolor = color, zorder = 3)

        # Subtle fill
        ax.fill(x_coords, y_coords, alpha = 0.08, color = color, zorder = 2)

        # Value labels at points - small and unobtrusive
        for i, (x, y, raw_val) in enumerate(zip(x_coords[:-1], y_coords[:-1], raw_values)):
            offset = 0.15 + (idx * 0.08)
            angle = angles[i] - pi / 2
            x_offset = x + offset * cos(angle)
            y_offset = y + offset * sin(angle)

            label_text = f'{raw_val:.3g}'

            ax.text(x_offset, y_offset, label_text,
                    fontsize = 8, ha = 'center', va = 'center',
                    color = color,
                    bbox = dict(boxstyle = 'round,pad=0.2', facecolor = 'white',
                                edgecolor = color, linewidth = 0.8, alpha = 0.85),
                    zorder = 5)

    # Add category labels - clean, no bold
    for i, (angle, cat) in enumerate(zip(angles[:-1], categories)):
        label_distance = 1.20
        x_label = label_distance * cos(angle - pi / 2)
        y_label = label_distance * sin(angle - pi / 2)

        if abs(x_label) < 0.1:
            ha = 'center'
        elif x_label > 0:
            ha = 'left'
        else:
            ha = 'right'

        if abs(y_label) < 0.1:
            va = 'center'
        elif y_label > 0:
            va = 'bottom'
        else:
            va = 'top'

        ax.text(x_label, y_label, cat, fontsize = 12,
                ha = ha, va = va, zorder = 4)

    # Set axis limits
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title at top
    title = f'{dataset_name.strip().title()} Deletion Analysis'
    plt.suptitle(title, size = 16, y = 0.99)

    # Legend below title
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.12),
              ncol = len(metrics_list), frameon = True, fontsize = 11,
              edgecolor = 'gray', fancybox = False, shadow = False)

    # Clean white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    return fig


def main():
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required to generate star plots.")
        print("Please install it with: pip install matplotlib")
        return

    file_paths = {
        'Baseline 1': 'baseline_deletion_1_data_v2.csv',
        'Baseline 2': 'baseline_deletion_2_data_v2.csv'
    }

    all_data = {}

    for baseline_name, file_path in file_paths.items():
        print(f"Parsing {file_path} for {baseline_name}...")
        try:
            data = parse_csv_file(file_path)
            all_data[baseline_name] = data
            print(f"Found datasets in {baseline_name}: {list(data.keys())}")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Skipping {baseline_name}.")
            continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

    if not all_data:
        print("No data loaded. Exiting.")
        return

    first_baseline_name = list(all_data.keys())[0]
    datasets_to_plot = set(all_data[first_baseline_name].keys())

    for baseline_name, data in all_data.items():
        if baseline_name != first_baseline_name:
            datasets_to_plot.intersection_update(set(data.keys()))

    print(f"\nCommon datasets found for plotting: {sorted(list(datasets_to_plot))}")

    for dataset_name in sorted(list(datasets_to_plot)):
        print(f"\nProcessing {dataset_name}...")

        metrics_to_plot = {}
        for baseline_name in all_data.keys():
            dataset_data = all_data[baseline_name].get(dataset_name)
            if dataset_data:
                metrics = calculate_metrics(dataset_data)
                if metrics:
                    metrics_to_plot[baseline_name] = metrics
                    print(f"  {baseline_name} - Total Time (avg): {metrics['total_time']:.4f}")

        if len(metrics_to_plot) == len(file_paths):
            fig = create_star_plot(metrics_to_plot, dataset_name)

            output_file = f'star_plot_{dataset_name}_comparison.png'
            fig.savefig(output_file, dpi = 300, bbox_inches = 'tight')
            print(f"  Saved comparison plot: {output_file}")
            plt.close(fig)
        else:
            print(f"  Skipping {dataset_name}: Could not find data for all baselines.")

    print("\nAll comparison star plots created successfully!")


if __name__ == '__main__':
    main()