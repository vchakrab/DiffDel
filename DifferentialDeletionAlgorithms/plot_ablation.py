#!/usr/bin/env python3
"""
Ablation Study Figure for VLDB Paper
Generates a single-row figure with 5 panels:
  (a) Exp leakage heatmap (λ sweep)
  (b) Gum leakage heatmap (λ sweep)
  (c) Mask size vs λ (both mechanisms)
  (d) Pareto frontier (Exp only)  [FIXED: true Pareto frontier + float-robust λ markers]
  (e) ε-Convergence: Std(leakage) vs ε

Usage:
    python plot_ablation.py --data_dir ./data --output ./fig_ablation.pdf

Data files expected:
    Lambda sweep: del2ph_1.csv ... del2ph_10.csv, delgum_1.csv ... delgum_10.csv
    Epsilon sweep: edel2ph_1.csv ... edel2ph_300.csv, edelgum_1.csv ... edelgum_300.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    'axes.linewidth': 0.4,
    'grid.linewidth': 0.2,
    'lines.linewidth': 1.0,
    'text.usetex': False,
})

# Small caps style names (Unicode approximation)
SC_EXP = 'Exᴘ'
SC_GUM = 'Gᴜᴍ'
SC_MIN = 'Mɪɴ'

# Dataset configuration
DATASETS = ['airport', 'hospital', 'adult', 'flight', 'tax']
SAMPLES_PER_DATASET = 100

DATASET_LABELS = {
    'airport': 'Airport',
    'hospital': 'Hospital',
    'adult': 'Adult',
    'flight': 'Flight',
    'tax': 'Tax',
}

# Colorblind-friendly palette
DATASET_COLORS = {
    'airport': '#E69F00',    # Orange
    'hospital': '#56B4E9',   # Sky blue
    'adult': '#009E73',      # Teal/green
    'flight': '#CC79A7',     # Pink/magenta
    'tax': '#D55E00',        # Vermillion/red-orange
}

# DelMin baseline mask sizes per dataset
DELMIN_MASK_SIZES = {
    'airport': 6,
    'hospital': 10,
    'adult': 12,
    'flight': 13,
    'tax': 3,
}

# =============================================================================
# Data Loading
# =============================================================================

def _assign_dataset_by_index(n_rows: int) -> list[str]:
    """Assign dataset labels based on row index blocks of SAMPLES_PER_DATASET."""
    datasets = []
    for idx in range(n_rows):
        dataset_idx = idx // SAMPLES_PER_DATASET
        if dataset_idx < len(DATASETS):
            datasets.append(DATASETS[dataset_idx])
        else:
            datasets.append('unknown')
    return datasets


def load_lambda_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load lambda sweep data from del2ph_*.csv and delgum_*.csv files.
    Each file contains 500 rows (100 samples × 5 datasets) (as assumed).
    """
    data: dict[str, list[pd.DataFrame]] = {'exp': [], 'gum': []}
    mechanism_map = {'del2ph': 'exp', 'delgum': 'gum'}

    for file_prefix, mech_key in mechanism_map.items():
        for i in range(1, 11):  # λ = 0.1 to 1.0
            filepath = data_dir / f"{file_prefix}_{i}.csv"
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath)
            if len(df) == 0:
                continue

            # Extract lambda from file or column
            lam = df['lambda'].iloc[0] if 'lambda' in df.columns else i / 10.0
            df['lambda'] = pd.to_numeric(lam)
            df['mechanism'] = mech_key

            # Coerce expected numeric cols if present
            for col in ['leakage', 'mask_size']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Assign dataset labels based on row index
            df['dataset'] = _assign_dataset_by_index(len(df))
            data[mech_key].append(df)

    out = {}
    for m in ['exp', 'gum']:
        if data[m]:
            out[m] = pd.concat(data[m], ignore_index=True)
    return out


def load_epsilon_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load epsilon sweep data from edel2ph_*.csv and edelgum_*.csv files.
    """
    data: dict[str, list[pd.DataFrame]] = {'exp': [], 'gum': []}
    mechanism_map = {'edel2ph': 'exp', 'edelgum': 'gum'}
    eps_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]

    for file_prefix, mech_key in mechanism_map.items():
        for eps in eps_values:
            filepath = data_dir / f"{file_prefix}_{eps}.csv"
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath)
            if len(df) == 0:
                continue

            # Handle duplicated header row if present
            if 'epsilon' in df.columns and isinstance(df.iloc[0]['epsilon'], str) and df.iloc[0]['epsilon'] == 'epsilon':
                df = df.iloc[1:].copy()

            # Ensure expected columns exist and are numeric
            if 'epsilon' in df.columns:
                df['epsilon'] = pd.to_numeric(df['epsilon'], errors='coerce')
            else:
                df['epsilon'] = eps

            for col in ['leakage', 'mask_size']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['mechanism'] = mech_key
            df['dataset'] = _assign_dataset_by_index(len(df))
            data[mech_key].append(df)

    out = {}
    for m in ['exp', 'gum']:
        if data[m]:
            out[m] = pd.concat(data[m], ignore_index=True)
    return out

# =============================================================================
# Statistics Computation
# =============================================================================

def compute_lambda_stats(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute mean and std statistics grouped by lambda and dataset."""
    stats = {}
    for mechanism, df in data.items():
        # Only keep known datasets
        df = df[df['dataset'].isin(DATASETS)].copy()

        grouped = df.groupby(['lambda', 'dataset']).agg({
            'leakage': ['mean', 'std', 'count'],
            'mask_size': ['mean', 'std'],
        }).reset_index()

        grouped.columns = [
            'lambda', 'dataset',
            'leakage_mean', 'leakage_std', 'leakage_count',
            'mask_size_mean', 'mask_size_std'
        ]
        stats[mechanism] = grouped
    return stats


def compute_epsilon_stats(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute mean and std statistics grouped by epsilon and dataset."""
    stats = {}
    for mechanism, df in data.items():
        # Only keep known datasets
        df = df[df['dataset'].isin(DATASETS)].copy()

        grouped = df.groupby(['epsilon', 'dataset']).agg({
            'leakage': ['mean', 'std', 'count'],
            'mask_size': ['mean', 'std'],
        }).reset_index()

        grouped.columns = [
            'epsilon', 'dataset',
            'leakage_mean', 'leakage_std', 'leakage_count',
            'mask_size_mean', 'mask_size_std'
        ]
        stats[mechanism] = grouped
    return stats

# =============================================================================
# Pareto helper
# =============================================================================

def pareto_frontier_min_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Indices of Pareto frontier points for minimizing both x and y.
    Returns indices in increasing x order.

    Implementation: sort by x asc, keep points that strictly improve best y so far.
    """
    order = np.argsort(x)
    best_y = np.inf
    keep = []
    for idx in order:
        if np.isfinite(x[idx]) and np.isfinite(y[idx]) and (y[idx] < best_y - 1e-12):
            keep.append(idx)
            best_y = y[idx]
    return np.array(keep, dtype=int)

# =============================================================================
# Plotting
# =============================================================================

def plot_ablation_figure(lambda_stats: dict[str, pd.DataFrame],
                        epsilon_stats: dict[str, pd.DataFrame],
                        output_path: Path) -> None:
    """Generate the single-row ablation figure with 5 panels."""
    fig = plt.figure(figsize=(12, 2.4))

    # Grid layout: legend row + plot row
    # Column widths: [heatmap, labels, heatmap, mask, pareto, epsilon]
    gs = fig.add_gridspec(
        2, 6,
        height_ratios=[0.08, 1],
        width_ratios=[1, 0.01, 1, 1.2, 1.2, 1.2],
        hspace=0.35, wspace=0.35,
        left=0.02, right=0.99, top=0.85, bottom=0.18
    )

    # =========================================================================
    # Legend (centred over panels c, d, e)
    # =========================================================================
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')

    dataset_handles = [
        mpatches.Patch(color=DATASET_COLORS[d], label=DATASET_LABELS[d])
        for d in DATASETS
    ]
    mech_handles = [
        Line2D([0], [0], color='dimgray', linestyle='-', linewidth=1.5, label=SC_EXP),
        Line2D([0], [0], color='dimgray', linestyle='--', linewidth=1.5, label=SC_GUM),
        Line2D([0], [0], color='black', linestyle=':', linewidth=1.2, label=SC_MIN),
    ]

    all_handles = dataset_handles + mech_handles
    ax_legend.legend(
        handles=all_handles,
        loc='center', ncol=8, frameon=False, fontsize=7,
        handlelength=1.8, handletextpad=0.4, columnspacing=1.0,
        bbox_to_anchor=(0.72, 0.5)
    )

    # =========================================================================
    # Create subplot axes
    # =========================================================================
    ax_exp_heat = fig.add_subplot(gs[1, 0])
    ax_labels = fig.add_subplot(gs[1, 1])
    ax_gum_heat = fig.add_subplot(gs[1, 2])
    ax_mask = fig.add_subplot(gs[1, 3])
    ax_pareto = fig.add_subplot(gs[1, 4])  # (remove duplicate bug)
    ax_eps_std = fig.add_subplot(gs[1, 5])

    # =========================================================================
    # (a) Exp leakage heatmap
    # =========================================================================
    if 'exp' in lambda_stats:
        df = lambda_stats['exp']
        pivot = df.pivot(index='dataset', columns='lambda', values='leakage_mean')
        pivot = pivot.reindex(DATASETS)
        pivot.index = [DATASET_LABELS[d] for d in pivot.index]

        sns.heatmap(
            pivot, ax=ax_exp_heat, cmap='Greys', vmin=0, vmax=1,
            annot=False, cbar=False,
            linewidths=0.3, linecolor='white'
        )

        ax_exp_heat.set_xlabel(r'$\lambda$', fontsize=7)
        ax_exp_heat.set_ylabel('')
        ax_exp_heat.set_yticklabels([])
        ax_exp_heat.yaxis.tick_right()

        # Ensure stable tick labels even if float columns are messy
        cols = list(pivot.columns)
        ax_exp_heat.set_xticklabels([f'{float(x):.1f}' for x in cols], rotation=0, fontsize=5)

    ax_exp_heat.text(
        0.5, -0.32, f'(a) {SC_EXP}: Leakage',
        transform=ax_exp_heat.transAxes, ha='center', fontsize=7
    )

    # =========================================================================
    # Shared dataset labels (between heatmaps)
    # =========================================================================
    ax_labels.axis('off')
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(0, 1)
    for i, dataset in enumerate(DATASETS):
        y_pos = 1 - (i + 0.5) / len(DATASETS)
        ax_labels.text(
            0.5, y_pos, DATASET_LABELS[dataset],
            ha='center', va='center', fontsize=5
        )

    # =========================================================================
    # (b) Gum leakage heatmap
    # =========================================================================
    if 'gum' in lambda_stats:
        df = lambda_stats['gum']
        pivot = df.pivot(index='dataset', columns='lambda', values='leakage_mean')
        pivot = pivot.reindex(DATASETS)
        pivot.index = [DATASET_LABELS[d] for d in pivot.index]

        sns.heatmap(
            pivot, ax=ax_gum_heat, cmap='Greys', vmin=0, vmax=1,
            annot=False, cbar=False,
            linewidths=0.3, linecolor='white'
        )

        ax_gum_heat.set_xlabel(r'$\lambda$', fontsize=7)
        ax_gum_heat.set_ylabel('')
        ax_gum_heat.set_yticklabels([])

        cols = list(pivot.columns)
        ax_gum_heat.set_xticklabels([f'{float(x):.1f}' for x in cols], rotation=0, fontsize=5)

    ax_gum_heat.text(
        0.5, -0.32, f'(b) {SC_GUM}: Leakage',
        transform=ax_gum_heat.transAxes, ha='center', fontsize=7
    )

    # =========================================================================
    # Colorbar for heatmaps
    # =========================================================================
    cbar_ax = fig.add_axes([0.02, 0.92, 0.20, 0.02])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap='Greys', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_label(r'$\mathcal{L}$', fontsize=6, labelpad=1)

    # =========================================================================
    # (c) Mask size vs λ (both mechanisms)
    # =========================================================================
    for mechanism in ['exp', 'gum']:
        if mechanism not in lambda_stats:
            continue
        df = lambda_stats[mechanism]
        linestyle = '-' if mechanism == 'exp' else '--'

        for dataset in DATASETS:
            df_ds = df[df['dataset'] == dataset].sort_values('lambda')
            ax_mask.plot(
                df_ds['lambda'], df_ds['mask_size_mean'],
                color=DATASET_COLORS[dataset],
                linestyle=linestyle,
                linewidth=1.0
            )

    # DelMin reference lines
    for dataset in DATASETS:
        ax_mask.axhline(
            y=DELMIN_MASK_SIZES[dataset],
            color=DATASET_COLORS[dataset],
            linestyle=':', linewidth=0.8, alpha=0.7
        )

    ax_mask.set_xlabel(r'$\lambda$', fontsize=7)
    ax_mask.set_ylabel(r'$|M|$', fontsize=7, labelpad=2)
    ax_mask.grid(True, alpha=0.3)
    ax_mask.set_xlim(0.05, 1.05)
    ax_mask.tick_params(axis='both', labelsize=6)

    ax_mask.text(
        0.5, -0.32, '(c) Mask Size',
        transform=ax_mask.transAxes, ha='center', fontsize=7
    )

    # =========================================================================
    # (d) Pareto frontier (Exp only) -- FIXED (true Pareto over all λ)
    # =========================================================================
    key_lambdas = np.array([0.3, 0.5, 0.7, 0.9], dtype=float)

    if 'exp' in lambda_stats:
        df = lambda_stats['exp'].copy()

        for dataset in DATASETS:
            df_ds = df[df['dataset'] == dataset].copy()
            if df_ds.empty:
                continue

            x = df_ds['mask_size_mean'].to_numpy(dtype=float)
            y = df_ds['leakage_mean'].to_numpy(dtype=float)

            keep_idx = pareto_frontier_min_xy(x, y)
            x_f = x[keep_idx]
            y_f = y[keep_idx]

            ax_pareto.plot(
                x_f, y_f,
                color=DATASET_COLORS[dataset],
                linestyle='-',
                linewidth=1.0,
                marker='o',
                markersize=3
            )

            # Overlay markers for selected λ values (float-robust)
            lam = df_ds['lambda'].to_numpy(dtype=float)
            for kl in key_lambdas:
                m = np.isclose(lam, kl, atol=1e-9, rtol=0)
                if np.any(m):
                    ax_pareto.scatter(
                        df_ds.loc[m, 'mask_size_mean'],
                        df_ds.loc[m, 'leakage_mean'],
                        color=DATASET_COLORS[dataset],
                        s=18,
                        zorder=5
                    )

    ax_pareto.set_xlabel(r'$|M|$', fontsize=7)
    ax_pareto.set_ylabel(r'$\mathcal{L}$', fontsize=7, labelpad=1)
    ax_pareto.grid(True, alpha=0.3)
    ax_pareto.set_ylim(0, 1.05)
    ax_pareto.tick_params(axis='both', labelsize=6)

    ax_pareto.text(
        0.5, -0.32, '(d) Pareto Frontier',
        transform=ax_pareto.transAxes, ha='center', fontsize=7
    )

    # =========================================================================
    # (e) ε-Convergence: Std(leakage) vs ε
    # =========================================================================
    for mechanism in ['exp', 'gum']:
        if mechanism not in epsilon_stats:
            continue
        df = epsilon_stats[mechanism]
        linestyle = '-' if mechanism == 'exp' else '--'

        for dataset in DATASETS:
            df_ds = df[df['dataset'] == dataset].sort_values('epsilon')
            ax_eps_std.plot(
                df_ds['epsilon'], df_ds['leakage_std'],
                color=DATASET_COLORS[dataset],
                linestyle=linestyle,
                linewidth=1.0
            )

    ax_eps_std.set_xlabel(r'$\varepsilon$', fontsize=7)
    ax_eps_std.set_ylabel(r'Std($\mathcal{L}$)', fontsize=7, labelpad=2)
    ax_eps_std.set_xscale('log')
    ax_eps_std.set_xlim(0.8, 400)
    ax_eps_std.set_ylim(0, 0.55)
    ax_eps_std.grid(True, alpha=0.3)
    ax_eps_std.tick_params(axis='both', labelsize=6)

    ax_eps_std.text(
        0.5, -0.32, r'(e) $\varepsilon$-Convergence',
        transform=ax_eps_std.transAxes, ha='center', fontsize=7
    )

    # =========================================================================
    # Save figure
    # =========================================================================
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"Saved: {output_path}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate ablation study figure for VLDB paper')
    parser.add_argument('--data_dir', type=str, default='./data2',
                        help='Directory containing CSV data files')
    parser.add_argument('--output', type=str, default='./fig_ablation.pdf',
                        help='Output PDF path')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    print("Loading lambda data...")
    lambda_data = load_lambda_data(data_dir)
    lambda_stats = compute_lambda_stats(lambda_data)

    print("Loading epsilon data...")
    epsilon_data = load_epsilon_data(data_dir)
    epsilon_stats = compute_epsilon_stats(epsilon_data)

    print("Generating ablation figure...")
    plot_ablation_figure(lambda_stats, epsilon_stats, output_path)

    print("\nDone!")

if __name__ == '__main__':
    main()
