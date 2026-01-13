#!/usr/bin/env python3
"""
Runtime figure generation script for DiffDel VLDB paper.

Generates a 5-panel figure with:
- Left group (3 thin bars): Runtime decomposition (Init/Model/Update) with left y-axis (ms)
- Right group (3 brown bars): Zone visualization showing instantiated cells,
  with dark brown top portion indicating mask size
- Dual y-axes: Left for time (ms), right for cells (brown)
- Independent scales per dataset

Usage:
    python generate_runtime_figure.py --delmin PATH --del2ph PATH --delgum PATH --output PATH

Input CSV files:
    - delmin: DelMin results CSV
    - del2ph: Del2Ph (Exp) results CSV
    - delgum: DelGum results CSV

Output:
    - fig_runtime_faceted.pdf
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
})

PHASE_COLORS = {'init': '#3498db', 'model': '#e74c3c', 'update': '#9b59b6'}
ZONE_COLOR_LIGHT = '#D2B48C'  # Light tan for instantiated cells
ZONE_COLOR_DARK = '#8B4513'  # Saddle brown for mask

DATASET_ORDER = ['Airport', 'Hospital', 'Adult', 'Flight', 'Tax']
METHOD_NAMES = {'DelMin': 'Min', 'Del2Ph': 'Exp', 'DelGum': 'Gum'}


# =============================================================================
# UTILITIES
# =============================================================================

def to_small_caps(text):
    """Convert text to Unicode small caps for matplotlib labels."""
    small_caps_map = {
        'a': 'ᴀ', 'b': 'ʙ', 'c': 'ᴄ', 'd': 'ᴅ', 'e': 'ᴇ', 'f': 'ғ', 'g': 'ɢ',
        'h': 'ʜ', 'i': 'ɪ', 'j': 'ᴊ', 'k': 'ᴋ', 'l': 'ʟ', 'm': 'ᴍ', 'n': 'ɴ',
        'o': 'ᴏ', 'p': 'ᴘ', 'q': 'ǫ', 'r': 'ʀ', 's': 's', 't': 'ᴛ', 'u': 'ᴜ',
        'v': 'ᴠ', 'w': 'ᴡ', 'x': 'x', 'y': 'ʏ', 'z': 'ᴢ'
    }
    return ''.join(small_caps_map.get(c.lower(), c) for c in text)


def load_data(delmin_path, del2ph_path, delgum_path):
    """
    Load and merge CSV files from the three deletion mechanisms.

    For Del2Ph and DelGum, expects columns:
        method, dataset, target_attribute, total_time, init_time,
        model_time, del_time, leakage, baseline_leakage_empty_mask,
        utility, total_paths, mask_size, model_size, num_instantiated_cells

    For Del2Ph: filters out model initialization outliers (P95 threshold)
    """
    # Column names for new format (del2ph, delgum)
    new_columns = ['method', 'dataset', 'target_attribute', 'total_time', 'init_time',
                   'model_time', 'del_time', 'leakage', 'baseline_leakage_empty_mask',
                   'utility', 'total_paths', 'mask_size', 'model_size', 'num_instantiated_cells']

    # Load DelMin (has header)
    delmin = pd.read_csv(delmin_path)
    delmin['method'] = 'DelMin'

    # Load Del2Ph (skip malformed header row)
    del2ph = pd.read_csv(del2ph_path, header = None, skiprows = 1, names = new_columns)
    del2ph['method'] = 'Del2Ph'
    del2ph = del2ph.rename(columns = {'del_time': 'update_time'})

    # Load DelGum (skip malformed header row)
    delgum = pd.read_csv(delgum_path, header = None, skiprows = 1, names = new_columns)
    delgum['method'] = 'DelGum'
    delgum = delgum.rename(columns = {'del_time': 'update_time'})

    # Filter Del2Ph outliers (model initialization runs)
    del2ph_filtered = []
    for dataset in del2ph['dataset'].unique():
        ds_data = del2ph[del2ph['dataset'] == dataset]
        threshold = ds_data['total_time'].quantile(0.95)
        filtered = ds_data[ds_data['total_time'] <= threshold]
        del2ph_filtered.append(filtered)
        print(
            f"Del2Ph {dataset}: {len(ds_data)} -> {len(filtered)} rows (filtered initialization outliers)")
    del2ph = pd.concat(del2ph_filtered, ignore_index = True)

    # Merge all
    df = pd.concat([delmin, del2ph, delgum], ignore_index = True)
    df['dataset'] = df['dataset'].str.capitalize()

    # Derived columns (convert to ms)
    df['init_time_ms'] = df['init_time'] * 1000
    df['model_time_ms'] = df['model_time'] * 1000
    df['update_time_ms'] = df['update_time'] * 1000

    return df


def generate_runtime_figure(df, output_path):
    """
    Generate runtime decomposition figure with 5 panels (one per dataset).

    Left group: 3 thin stacked bars for runtime phases (Init/Model/Update)
    Right group: 3 brown bars showing instantiated cells with mask overlay
    """
    # Aggregate
    summary = df.groupby(['dataset', 'method']).agg({
        'init_time_ms': 'mean',
        'model_time_ms': 'mean',
        'update_time_ms': 'mean',
        'mask_size': 'mean',
        'num_instantiated_cells': 'mean',
    }).reset_index()

    methods = ['DelMin', 'Del2Ph', 'DelGum']

    fig, axes = plt.subplots(1, 5, figsize = (12, 2.8))

    bar_width = 0.12

    for ax, dataset in zip(axes, DATASET_ORDER):
        ds_data = summary[summary['dataset'] == dataset].set_index('method').reindex(methods)

        # Bar positions
        runtime_positions = np.array([0, 0.15, 0.30])
        zone_positions = np.array([0.50, 0.65, 0.80])

        # Get data
        init = ds_data['init_time_ms'].values
        model = ds_data['model_time_ms'].values
        update = ds_data['update_time_ms'].values
        mask_sizes = ds_data['mask_size'].values + 1  # +1 for target cell
        instantiated = ds_data['num_instantiated_cells'].values

        # Plot runtime bars (stacked)
        ax.bar(runtime_positions, init, bar_width,
               color = PHASE_COLORS['init'], edgecolor = 'black', linewidth = 0.5)
        ax.bar(runtime_positions, model, bar_width, bottom = init,
               color = PHASE_COLORS['model'], edgecolor = 'black', linewidth = 0.5)
        ax.bar(runtime_positions, update, bar_width, bottom = init + model,
               color = PHASE_COLORS['update'], edgecolor = 'black', linewidth = 0.5)

        ax.set_ylabel('Time (ms)', fontsize = 7, color = 'black')
        ax.tick_params(axis = 'y', labelsize = 6, colors = 'black')

        # Create right y-axis for zone bars
        ax2 = ax.twinx()

        # Plot zone bars: full height = instantiated, top shaded = mask
        for i, (pos, mask_sz, inst) in enumerate(zip(zone_positions, mask_sizes, instantiated)):
            # Full bar: instantiated cells (light)
            ax2.bar(pos, inst, bar_width,
                    color = ZONE_COLOR_LIGHT, edgecolor = 'black', linewidth = 0.5)
            # Overlay top portion: mask (dark)
            ax2.bar(pos, mask_sz, bar_width, bottom = (inst - mask_sz),
                    color = ZONE_COLOR_DARK, edgecolor = 'black', linewidth = 0.5)

        ax2.set_ylabel('Cells', fontsize = 7, color = '#8B4513')
        ax2.tick_params(axis = 'y', labelsize = 6, colors = '#8B4513')
        max_inst = max(instantiated)
        ax2.set_ylim(0, max_inst * 1.15)

        # X-axis labels
        all_positions = list(runtime_positions) + list(zone_positions)
        sc_labels = [to_small_caps(METHOD_NAMES[m]) for m in methods]
        all_labels = sc_labels + sc_labels

        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, fontsize = 6)
        ax.set_xlim(-0.1, 0.95)

        ax.set_title(dataset, fontsize = 9, fontweight = 'bold')

        # Vertical separator between runtime and zone groups
        ax.axvline(x = 0.40, color = 'gray', linestyle = ':', alpha = 0.5, linewidth = 0.8)

    # Legend
    phase_patches = [
        Patch(facecolor = PHASE_COLORS['init'], label = 'Init', edgecolor = 'black',
              linewidth = 0.5),
        Patch(facecolor = PHASE_COLORS['model'], label = 'Model', edgecolor = 'black',
              linewidth = 0.5),
        Patch(facecolor = PHASE_COLORS['update'], label = 'Update', edgecolor = 'black',
              linewidth = 0.5),
        Patch(facecolor = ZONE_COLOR_LIGHT, label = 'Instantiated', edgecolor = 'black',
              linewidth = 0.5),
        Patch(facecolor = ZONE_COLOR_DARK, label = 'Mask', edgecolor = 'black', linewidth = 0.5),
    ]

    fig.legend(handles = phase_patches, loc = 'lower center', ncol = 5, fontsize = 7,
               bbox_to_anchor = (0.5, -0.02), frameon = False, columnspacing = 1.5)

    plt.tight_layout(rect = [0, 0.08, 1, 1])

    output_file = Path(output_path) / 'fig_runtime_faceted.pdf'
    plt.savefig(output_file, bbox_inches = 'tight')
    plt.close()

    print(f"\nSaved: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description = 'Generate runtime figure for DiffDel VLDB paper.'
    )
    parser.add_argument('--delmin', type = str, required = True,
                        help = 'Path to DelMin CSV file')
    parser.add_argument('--del2ph', type = str, required = True,
                        help = 'Path to Del2Ph (Exp) CSV file')
    parser.add_argument('--delgum', type = str, required = True,
                        help = 'Path to DelGum CSV file')
    parser.add_argument('--output', type = str, default = '.',
                        help = 'Output directory (default: current directory)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents = True, exist_ok = True)

    print("=" * 60)
    print("DiffDel VLDB Paper - Runtime Figure Generation")
    print("=" * 60)
    print(f"DelMin: {args.delmin}")
    print(f"Del2Ph: {args.del2ph}")
    print(f"DelGum: {args.delgum}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading data...")
    df = load_data(args.delmin, args.del2ph, args.delgum)
    print(f"Loaded {len(df)} total rows")
    print()

    # Generate figure
    print("Generating figure...")
    generate_runtime_figure(df, output_dir)

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
