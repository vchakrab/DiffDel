#!/usr/bin/env python3
"""
Runtime figure generation script for DiffDel VLDB paper.

Generates a 5-panel figure with:
- Left group (3 thin bars): Runtime decomposition (Init/Model/Update) with left y-axis (ms)
- Right group (3 brown bars): Zone visualization showing instantiated cells,
  with dark brown top portion indicating mask size
- Dual y-axes: Left for time (ms), right for cells (brown)
- Independent scales per dataset

No argparse. Just edit DATA_DIR + filenames below and run:
    python generate_runtime_figure.py

Output:
    - <OUTPUT_DIR>/fig_runtime_faceted.pdf
"""

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
ZONE_COLOR_DARK  = '#8B4513'  # Saddle brown for mask

DATASET_ORDER = ['Airport', 'Hospital', 'Adult', 'Flight', 'Tax']
METHOD_NAMES = {'DelMin': 'Min', 'Del2Ph': 'Exp', 'DelGum': 'Gum'}

# =============================================================================
# EDIT THESE PATHS
# =============================================================================

# Folder where the CSVs live
DATA_DIR = Path(".")   # change to Path("/absolute/path/to/your/csvs") if needed

# Output directory for the PDF
OUTPUT_DIR = Path(".")  # change if you want somewhere else

# Your filenames (use exactly what you have)
DELMIN_CSV = "min_20260115-085305.csv"
DEL2PH_CSV = "2ph_20260115-181356.csv"
DELGUM_CSV = "gum_20260115-181307.csv"

# =============================================================================
# UTILITIES
# =============================================================================

def to_small_caps(text: str) -> str:
    """Convert text to Unicode small caps for matplotlib labels."""
    small_caps_map = {
        'a': 'ᴀ', 'b': 'ʙ', 'c': 'ᴄ', 'd': 'ᴅ', 'e': 'ᴇ', 'f': 'ғ', 'g': 'ɢ',
        'h': 'ʜ', 'i': 'ɪ', 'j': 'ᴊ', 'k': 'ᴋ', 'l': 'ʟ', 'm': 'ᴍ', 'n': 'ɴ',
        'o': 'ᴏ', 'p': 'ᴘ', 'q': 'ǫ', 'r': 'ʀ', 's': 's', 't': 'ᴛ', 'u': 'ᴜ',
        'v': 'ᴠ', 'w': 'ᴡ', 'x': 'x', 'y': 'ʏ', 'z': 'ᴢ'
    }
    return ''.join(small_caps_map.get(c.lower(), c) for c in text)

def load_data(delmin_path: Path, del2ph_path: Path, delgum_path: Path) -> pd.DataFrame:
    """
    Robust loader that normalizes DelMin/Del2Ph/DelGum into a common schema:
      dataset, method, init_time, model_time, update_time, mask_size, num_instantiated_cells

    Del2Ph/DelGum: header=None, skiprows=1, fixed column layout.
    DelMin: auto-detect/rename common column variants.
    """

    # -----------------------
    # Del2Ph / DelGum schema
    # -----------------------
    new_columns = [
        'method', 'dataset', 'target_attribute', 'total_time', 'init_time',
        'model_time', 'del_time', 'leakage', 'baseline_leakage_empty_mask',
        'utility', 'total_paths', 'mask_size', 'model_size', 'num_instantiated_cells'
    ]

    # ---- DelMin (robust) ----
    delmin = pd.read_csv(delmin_path)

    # normalize column names (strip spaces)
    delmin.columns = [str(c).strip() for c in delmin.columns]

    # If DelMin doesn't have 'method', add it
    delmin['method'] = 'DelMin'

    # Auto-rename likely DelMin columns into the canonical names used by plotting.
    # Add/adjust aliases here if your delmin csv uses different names.
    aliases = {
        # dataset
        'Dataset': 'dataset',
        'ds': 'dataset',
        'data_set': 'dataset',

        # times
        'del_time': 'update_time',
        'deletion_time': 'update_time',
        'delete_time': 'update_time',
        'mask_time': 'update_time',
        'update': 'update_time',

        # instantiated cells / model size
        'instantiated_cells': 'num_instantiated_cells',
        'num_inst_cells': 'num_instantiated_cells',
        'instantiated': 'num_instantiated_cells',
        'model_size': 'num_instantiated_cells',   # sometimes DelMin logs this as model_size

        # mask size
        'masked_cells': 'mask_size',
        'num_masked_cells': 'mask_size',
        'mask_cells': 'mask_size',
        'mask': 'mask_size',

        # init/model time variants
        'init': 'init_time',
        'model': 'model_time',
        'build_time': 'model_time',
    }
    for src, dst in aliases.items():
        if src in delmin.columns and dst not in delmin.columns:
            delmin = delmin.rename(columns={src: dst})

    # If DelMin doesn't have update_time but has del_time, map it
    if 'update_time' not in delmin.columns and 'del_time' in delmin.columns:
        delmin = delmin.rename(columns={'del_time': 'update_time'})

    # If DelMin doesn't have num_instantiated_cells but has model_size, map it
    if 'num_instantiated_cells' not in delmin.columns and 'model_size' in delmin.columns:
        delmin = delmin.rename(columns={'model_size': 'num_instantiated_cells'})

    # If DelMin has total_time but no decomposition, we can't plot phases.
    # Try to synthesize missing parts as 0 so it still appears.
    if 'init_time' not in delmin.columns:
        delmin['init_time'] = 0.0
    if 'model_time' not in delmin.columns:
        delmin['model_time'] = 0.0
    if 'update_time' not in delmin.columns:
        # last resort: if only total_time exists
        if 'total_time' in delmin.columns:
            delmin['update_time'] = delmin['total_time']
        else:
            delmin['update_time'] = 0.0

    # If DelMin is missing instantiated cells or mask size, try to keep it but it may be blank on the right bars
    if 'mask_size' not in delmin.columns:
        delmin['mask_size'] = np.nan
    if 'num_instantiated_cells' not in delmin.columns:
        delmin['num_instantiated_cells'] = np.nan

    # Ensure dataset column exists
    if 'dataset' not in delmin.columns:
        raise ValueError(
            f"DelMin file {delmin_path} has no 'dataset' column (or alias). "
            f"Columns present: {list(delmin.columns)}"
        )

    # ---- Del2Ph ----
    del2ph = pd.read_csv(del2ph_path, header=None, skiprows=1, names=new_columns)
    del2ph['method'] = 'Del2Ph'
    del2ph = del2ph.rename(columns={'del_time': 'update_time'})

    # ---- DelGum ----
    delgum = pd.read_csv(delgum_path, header=None, skiprows=1, names=new_columns)
    delgum['method'] = 'DelGum'
    delgum = delgum.rename(columns={'del_time': 'update_time'})

    # NOTE: No outlier filtering here. This script should match generate_plots.py.

    # Merge all
    df = pd.concat([delmin, del2ph, delgum], ignore_index=True)

    # Normalize dataset formatting
    df['dataset'] = df['dataset'].astype(str).str.strip().str.capitalize()

    # Make sure numeric columns are numeric
    for col in ['init_time', 'model_time', 'update_time', 'mask_size', 'num_instantiated_cells']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert to ms (assumes seconds in inputs)
    df['init_time_ms'] = df['init_time'] * 1000.0
    df['model_time_ms'] = df['model_time'] * 1000.0
    df['update_time_ms'] = df['update_time'] * 1000.0

    # IMPORTANT: "time" is ALWAYS defined as init + model + del/update (not total_time)
    # so it matches generate_plots.py exactly.
    df['time_ms'] = df['init_time_ms'] + df['model_time_ms'] + df['update_time_ms']
    # Canonical time used everywhere: init + model + del
    df['time_ms'] = df['init_time_ms'] + df['model_time_ms'] + df['update_time_ms']

    # IMPORTANT: do NOT drop rows just because mask/instantiated are NaN
    # (otherwise DelMin disappears if it doesn't have those columns).
    df = df.dropna(subset=['dataset', 'method', 'init_time_ms', 'model_time_ms', 'update_time_ms'])

    # Debug print: confirm each method survived
    print("\nRows per method after normalization:")
    print(df['method'].value_counts(dropna=False).to_string())

    return df

def generate_runtime_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate runtime decomposition figure with 5 panels (one per dataset).

    Left group: 3 thin stacked bars for runtime phases (Init/Model/Update)
    Right group: 3 brown bars showing instantiated cells with mask overlay
    """
    summary = df.groupby(['dataset', 'method']).agg({
        'init_time_ms': 'mean',
        'model_time_ms': 'mean',
        'update_time_ms': 'mean',
        'mask_size': 'mean',
        'num_instantiated_cells': 'mean',
    }).reset_index()

    methods = ['DelMin', 'Del2Ph', 'DelGum']

    fig, axes = plt.subplots(1, 5, figsize=(12, 2.8))
    bar_width = 0.12

    for ax, dataset in zip(axes, DATASET_ORDER):
        ds_data = summary[summary['dataset'] == dataset].set_index('method').reindex(methods)

        # If a dataset is missing entirely, keep the panel but show "No data"
        if ds_data[['init_time_ms','model_time_ms','update_time_ms','mask_size','num_instantiated_cells']].isna().all().all():
            ax.set_title(dataset, fontsize=9, fontweight='bold')
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        runtime_positions = np.array([0.00, 0.15, 0.30])
        zone_positions    = np.array([0.50, 0.65, 0.80])

        init  = ds_data['init_time_ms'].values
        model = ds_data['model_time_ms'].values
        update = ds_data['update_time_ms'].values

        mask_sizes = ds_data['mask_size'].values  # +1 for target cell
        instantiated = ds_data['num_instantiated_cells'].values

        # Runtime stacked bars (left axis)
        ax.bar(runtime_positions, init, bar_width,
               color=PHASE_COLORS['init'], edgecolor='black', linewidth=0.5)
        ax.bar(runtime_positions, model, bar_width, bottom=init,
               color=PHASE_COLORS['model'], edgecolor='black', linewidth=0.5)
        ax.bar(runtime_positions, update, bar_width, bottom=init + model,
               color=PHASE_COLORS['update'], edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Time (ms)', fontsize=7, color='black')
        ax.tick_params(axis='y', labelsize=6, colors='black')

        # Zone bars (right axis)
        ax2 = ax.twinx()

        for pos, mask_sz, inst in zip(zone_positions, mask_sizes, instantiated):
            if np.isnan(inst) or inst <= 0:
                continue

            # Clamp overlay so we never draw negative bottoms if mask_sz > inst
            m = float(mask_sz)
            I = float(inst)
            m = min(m, I)

            ax2.bar(pos, I, bar_width,
                    color=ZONE_COLOR_LIGHT, edgecolor='black', linewidth=0.5)
            ax2.bar(pos, m, bar_width, bottom=(I - m),
                    color=ZONE_COLOR_DARK, edgecolor='black', linewidth=0.5)

        ax2.set_ylabel('Cells', fontsize=7, color=ZONE_COLOR_DARK)
        ax2.tick_params(axis='y', labelsize=6, colors=ZONE_COLOR_DARK)

        max_inst = np.nanmax(instantiated) if np.isfinite(np.nanmax(instantiated)) else 1.0
        ax2.set_ylim(0, max_inst * 1.15)

        # X labels (method names repeated for runtime + zone)
        sc_labels = [to_small_caps(METHOD_NAMES[m]) for m in methods]
        all_positions = list(runtime_positions) + list(zone_positions)
        all_labels = sc_labels + sc_labels

        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, fontsize=6)
        ax.set_xlim(-0.1, 0.95)

        ax.set_title(dataset, fontsize=9, fontweight='bold')

        # Separator between groups
        ax.axvline(x=0.40, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    # Legend
    phase_patches = [
        Patch(facecolor=PHASE_COLORS['init'],   label='Init',   edgecolor='black', linewidth=0.5),
        Patch(facecolor=PHASE_COLORS['model'],  label='Model',  edgecolor='black', linewidth=0.5),
        Patch(facecolor=PHASE_COLORS['update'], label='Update', edgecolor='black', linewidth=0.5),
        Patch(facecolor=ZONE_COLOR_LIGHT,       label='Instantiated', edgecolor='black', linewidth=0.5),
        Patch(facecolor=ZONE_COLOR_DARK,        label='Mask', edgecolor='black', linewidth=0.5),
    ]

    fig.legend(handles=phase_patches, loc='lower center', ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.02), frameon=False, columnspacing=1.5)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'fig_runtime_faceted.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {output_file}")

# =============================================================================
# MAIN (NO ARGPARSE)
# =============================================================================

def main():
    delmin_path = DATA_DIR / DELMIN_CSV
    del2ph_path = DATA_DIR / DEL2PH_CSV
    delgum_path = DATA_DIR / DELGUM_CSV

    print("=" * 60)
    print("DiffDel VLDB Paper - Runtime Figure Generation (no argparse)")
    print("=" * 60)
    print(f"DATA_DIR : {DATA_DIR.resolve()}")
    print(f"DelMin   : {delmin_path}")
    print(f"Del2Ph   : {del2ph_path}")
    print(f"DelGum   : {delgum_path}")
    print(f"OUTPUT   : {OUTPUT_DIR.resolve()}")
    print()

    for p in [delmin_path, del2ph_path, delgum_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    print("Loading data...")
    df = load_data(delmin_path, del2ph_path, delgum_path)
    print(f"Loaded {len(df)} total rows\n")

    print("Generating figure...")
    generate_runtime_figure(df, OUTPUT_DIR)

    print("\nDone!")

if __name__ == '__main__':
    main()
