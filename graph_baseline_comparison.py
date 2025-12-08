import csv
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib and/or numpy not available. Install with: pip install matplotlib numpy")


def parse_csv_file(file_path):
    """Parse the CSV file with section headers (-----dataset-----)"""
    data = []
    current_dataset = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header
            if line.startswith('-----') and line.endswith('-----'):
                current_dataset = line.strip('-')
            elif current_dataset and ',' in line:
                # Skip header row
                if line.startswith('attribute,time,row,cells'):
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        attribute = parts[0]
                        time = float(parts[1])
                        row = int(parts[2])
                        cells = int(parts[3])
                        
                        data.append({
                            'dataset': current_dataset,
                            'attribute': attribute,
                            'time': time,
                            'row': row,
                            'cells': cells
                        })
                    except (ValueError, IndexError):
                        continue
    
    return data


def aggregate_data(data_list):
    """Aggregate data by dataset"""
    if not data_list:
        return {}
    
    aggregated = defaultdict(lambda: {'Total Time (s)': 0, 'Total Cells Deleted': 0, 'Number of Operations': 0})
    
    for record in data_list:
        dataset = record['dataset']
        aggregated[dataset]['Total Time (s)'] += record['time']
        aggregated[dataset]['Total Cells Deleted'] += record['cells']
        aggregated[dataset]['Number of Operations'] += 1
    
    # Convert to list of dicts with Dataset key
    result = []
    for dataset, values in aggregated.items():
        result.append({
            'Dataset': dataset,
            'Total Time (s)': values['Total Time (s)'],
            'Total Cells Deleted': values['Total Cells Deleted'],
            'Number of Operations': values['Number of Operations']
        })
    
    return result


def create_dual_axis_comparison_chart(data1, data2, label1='Baseline 1', label2='Baseline 2'):
    """Create a dual y-axis bar chart comparing two datasets"""
    
    # Convert list of dicts to dictionaries keyed by Dataset
    data1_dict = {item['Dataset']: item for item in data1}
    data2_dict = {item['Dataset']: item for item in data2}
    
    # Get all unique datasets in a specific order
    all_datasets = sorted(set(list(data1_dict.keys()) + list(data2_dict.keys())))
    # Ensure airport comes first if it exists
    if 'airport' in all_datasets:
        all_datasets.remove('airport')
        all_datasets.insert(0, 'airport')
    
    num_datasets = len(all_datasets)
    width = 0.35
    bar_spacing = 0.1  # Space between cells and time bars within same dataset
    dataset_spacing = 0.3  # Space between datasets
    baseline_gap = 1.5  # Big gap between Baseline 1 and Baseline 2
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(16, 7))
    
    # Setup for Primary Axis (Time)
    color_b1 = '#1f77b4'  # Blue for Baseline 1
    color_b2 = '#ff7f0e'  # Orange for Baseline 2
    color_cells = '#8B4513'  # Brown for cells
    color_time = '#4169E1'  # Blue for time
    
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Time (s)', color=color_time, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_time)
    
    # Setup for Secondary Axis (Cells)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Cells Deleted', color=color_cells, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_cells)
    
    # Build positions: Baseline 1 first, then gap, then Baseline 2
    x_positions = []
    dataset_labels = []
    dataset_centers = []
    
    # Baseline 1 positions (left side)
    b1_start = 0
    for i, ds in enumerate(all_datasets):
        # Cells bar position (left), Time bar position (right) for each dataset
        cells_pos = b1_start + i * (2 * width + bar_spacing + dataset_spacing)
        time_pos = cells_pos + width + bar_spacing
        x_positions.append({
            'dataset': ds,
            'baseline': 1,
            'cells_pos': cells_pos,
            'time_pos': time_pos
        })
        dataset_centers.append((cells_pos + time_pos) / 2)
        dataset_labels.append(f'{ds}\n({label1})')
    
    # Baseline 2 positions (right side)
    b2_start = x_positions[-1]['time_pos'] + width + baseline_gap
    for i, ds in enumerate(all_datasets):
        cells_pos = b2_start + i * (2 * width + bar_spacing + dataset_spacing)
        time_pos = cells_pos + width + bar_spacing
        x_positions.append({
            'dataset': ds,
            'baseline': 2,
            'cells_pos': cells_pos,
            'time_pos': time_pos
        })
        dataset_centers.append((cells_pos + time_pos) / 2)
        dataset_labels.append(f'{ds}\n({label2})')
    
    # Extract values and plot
    bars1_cells = []
    bars1_time = []
    bars2_cells = []
    bars2_time = []
    
    for pos_info in x_positions:
        ds = pos_info['dataset']
        baseline = pos_info['baseline']
        
        if baseline == 1:
            cells_val = data1_dict.get(ds, {}).get('Total Cells Deleted', 0)
            time_val = data1_dict.get(ds, {}).get('Total Time (s)', 0)
            
            if cells_val > 0:
                bar = ax2.bar(pos_info['cells_pos'], cells_val, width,
                             label=f'{label1} - Cells' if len(bars1_cells) == 0 else '',
                             color=color_cells, alpha=0.8, edgecolor='black', linewidth=1.5)
                bars1_cells.append(bar[0])
            
            if time_val > 0:
                bar = ax1.bar(pos_info['time_pos'], time_val, width,
                             label=f'{label1} - Time' if len(bars1_time) == 0 else '',
                             color=color_time, alpha=0.8, edgecolor='black', linewidth=1.5)
                bars1_time.append(bar[0])
        else:  # baseline == 2
            cells_val = data2_dict.get(ds, {}).get('Total Cells Deleted', 0)
            time_val = data2_dict.get(ds, {}).get('Total Time (s)', 0)
            
            if cells_val > 0:
                bar = ax2.bar(pos_info['cells_pos'], cells_val, width,
                             label=f'{label2} - Cells' if len(bars2_cells) == 0 else '',
                             color=color_cells, alpha=0.8, edgecolor='black', linewidth=1.5)
                bars2_cells.append(bar[0])
            
            if time_val > 0:
                bar = ax1.bar(pos_info['time_pos'], time_val, width,
                             label=f'{label2} - Time' if len(bars2_time) == 0 else '',
                             color=color_time, alpha=0.8, edgecolor='black', linewidth=1.5)
                bars2_time.append(bar[0])
    
    # Add vertical divider line in the middle
    divider_x = (x_positions[num_datasets - 1]['time_pos'] + x_positions[num_datasets]['cells_pos']) / 2
    ax1.axvline(x=divider_x, color='black', linestyle='-', linewidth=2, alpha=0.7, zorder=0)
    ax1.text(divider_x, ax1.get_ylim()[1] * 0.95, '|', ha='center', fontsize=20, 
             fontweight='bold', color='black')
    
    # Set x-axis labels
    ax1.set_xticks(dataset_centers)
    ax1.set_xticklabels(dataset_labels, rotation=0, ha='center', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', pad=12)
    
    # Calculate max values for y-axis limits
    all_times = []
    all_cells = []
    for pos_info in x_positions:
        ds = pos_info['dataset']
        if pos_info['baseline'] == 1:
            all_times.append(data1_dict.get(ds, {}).get('Total Time (s)', 0))
            all_cells.append(data1_dict.get(ds, {}).get('Total Cells Deleted', 0))
        else:
            all_times.append(data2_dict.get(ds, {}).get('Total Time (s)', 0))
            all_cells.append(data2_dict.get(ds, {}).get('Total Cells Deleted', 0))
    
    max_time = max(all_times) if all_times else 10
    max_cells = max(all_cells) if all_cells else 10
    
    ax1.set_ylim(0, max_time * 1.15 if max_time > 0 else 10)
    ax2.set_ylim(0, max_cells * 1.15 if max_cells > 0 else 10)
    
    # Add value labels on all bars
    def add_value_labels(bar_list, ax, format_func=lambda x: f'{x:.2f}'):
        for bar in bar_list:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       format_func(height),
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Get all bar objects for labeling
    all_bars_time = []
    all_bars_cells = []
    for pos_info in x_positions:
        ds = pos_info['dataset']
        if pos_info['baseline'] == 1:
            time_val = data1_dict.get(ds, {}).get('Total Time (s)', 0)
            cells_val = data1_dict.get(ds, {}).get('Total Cells Deleted', 0)
        else:
            time_val = data2_dict.get(ds, {}).get('Total Time (s)', 0)
            cells_val = data2_dict.get(ds, {}).get('Total Cells Deleted', 0)
        
        if time_val > 0:
            # Find the bar at this position
            for bar in bars1_time + bars2_time:
                if abs(bar.get_x() - pos_info['time_pos']) < 0.01:
                    all_bars_time.append(bar)
                    break
        
        if cells_val > 0:
            for bar in bars1_cells + bars2_cells:
                if abs(bar.get_x() - pos_info['cells_pos']) < 0.01:
                    all_bars_cells.append(bar)
                    break
    
    # Add labels by iterating through axes children
    for bar in ax1.patches:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for bar in ax2.patches:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add title
    plt.title('Baseline Deletion Comparison: Time vs. Cells Deleted', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add baseline labels at top - positioned to the sides
    b1_left = x_positions[0]['cells_pos'] - 0.5
    b2_right = x_positions[-1]['time_pos'] + 0.5
    ax1.text(b1_left, ax1.get_ylim()[1] * 1.05, label1, ha='left', 
             fontsize=12, fontweight='bold', color='black')
    ax1.text(b2_right, ax1.get_ylim()[1] * 1.05, label2, ha='right', 
             fontsize=12, fontweight='bold', color='black')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    ax1.legend(all_lines, all_labels, loc='upper left', 
               framealpha=0.95, fontsize=10, ncol=2)
    
    # Adjust layout to prevent label cutoff with extra padding
    fig.tight_layout(pad=2.0)
    
    # Add extra margin around the plot for spacing and labels
    plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.15)
    
    return fig


def main():
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required to generate the chart.")
        print("Please install it with: pip install matplotlib numpy")
        print("\nHowever, I can still show you the parsed data:\n")
        # Still parse and show data even without matplotlib
        show_data_only = True
    else:
        show_data_only = False
    
    # Parse both CSV files
    print("Parsing baseline_deletion_1_data.csv...")
    data1_raw = parse_csv_file('baseline_deletion_1_data.csv')
    print(f"Found {len(data1_raw)} records in file 1")
    
    print("Parsing baseline_deletion_2_data.csv...")
    data2_raw = parse_csv_file('baseline_deletion_2_data.csv')
    print(f"Found {len(data2_raw)} records in file 2")
    
    # Aggregate data by dataset
    data1_agg = aggregate_data(data1_raw)
    data2_agg = aggregate_data(data2_raw)
    
    print("\nAggregated data for Baseline 1:")
    for item in data1_agg:
        print(f"  {item['Dataset']}: Time={item['Total Time (s)']:.2f}s, Cells={item['Total Cells Deleted']}")
    
    print("\nAggregated data for Baseline 2:")
    for item in data2_agg:
        print(f"  {item['Dataset']}: Time={item['Total Time (s)']:.2f}s, Cells={item['Total Cells Deleted']}")
    
    if show_data_only:
        return
    
    # Create comparison chart
    if data1_agg or data2_agg:
        fig = create_dual_axis_comparison_chart(data1_agg, data2_agg, 
                                                label1='Baseline 1', 
                                                label2='Baseline 2')
        
        # Save the figure
        output_file = 'baseline_comparison_chart.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {output_file}")
        
        # Display the chart
        plt.show()
    else:
        print("Error: No data found in the CSV files.")


if __name__ == '__main__':
    main()

