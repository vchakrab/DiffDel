#!/usr/bin/env python3
"""
Ablation Study Figure for VLDB Paper (2 rows) — FINAL

ALL PANELS ARE LINE GRAPHS (NO HEATMAPS).

TOP ROW (λ sweep):
  (a) Exp: Leakage vs λ (5 dataset lines)
  (b) Gum: Leakage vs λ (5 dataset lines)
  (c) Leakage vs λ (Exp+Gum overlaid; more distinguishable via markers)
  (d) Mask improvement vs λ  [% vs DelMin mask]
  (e) Utility vs λ

BOTTOM ROW (ε sweep) — LOG SCALE X ON ALL PANELS:
  (f) Exp: Leakage vs ε (5 dataset lines)            [LOG X]
  (g) Gum: Leakage vs ε (5 dataset lines)            [LOG X]
  (h) Leakage vs ε (Exp+Gum overlaid; markers)       [LOG X]
  (i) Mask improvement vs ε  [% vs DelMin mask]      [LOG X]
  (j) Utility vs ε                                   [LOG X]

Usage:
    python plot_ablation.py --data_dir ./data --output ./fig_ablation.pdf

Data files expected:
  Lambda sweep:   del2ph_0.csv ... delgum_10.csv
  Epsilon sweep:  edel2ph_<eps>.csv and edelgum_<eps>.csv where <eps> in:
      [0.05, 0.1, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1, 2, 8, 16, 32]

"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

SC_EXP = 'Exᴘ'
SC_GUM = 'Gᴜᴍ'
SC_MIN = 'Mɪɴ'

DATASETS = ['airport', 'hospital', 'adult', 'flight', 'tax']
SAMPLES_PER_DATASET = 100

DATASET_LABELS = {
    'airport': 'Airport',
    'hospital': 'Hospital',
    'adult': 'Adult',
    'flight': 'Flight',
    'tax': 'Tax',
}

DATASET_COLORS = {
    'airport': '#E69F00',
    'hospital': '#56B4E9',
    'adult': '#009E73',
    'flight': '#CC79A7',
    'tax': '#D55E00',
}

DELMIN_MASK_SIZES = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}

UTILITY_COL_CANDIDATES = [
    "utility",
    "utility_hinge",
    "hinge_utility",
    "utility_log",
    "log_utility",
    "util",
]

EPS_TICKS_PLOT = [0.05, 0.1, 0.2, 0.5, 1, 2, 8, 32]
EPS_VALUES = [0.05, 0.1, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1, 2, 8, 16, 32]# Exp/Gum distinguishability (combined plots)
MECH_STYLE = {
    "exp": {"linestyle": "-",  "marker": "o", "markersize": 2.8},
    "gum": {"linestyle": "--", "marker": "s", "markersize": 2.8},
}

# Single-mechanism leakage panels: clean 5 dataset lines
SINGLE_MECH_STYLE = {
    "exp": {"linestyle": "-",  "marker": None},
    "gum": {"linestyle": "--", "marker": None},
}

# =============================================================================
# Data Loading
# =============================================================================

def _assign_dataset_by_index(n_rows: int) -> List[str]:
    datasets = []
    for idx in range(n_rows):
        dataset_idx = idx // SAMPLES_PER_DATASET
        datasets.append(DATASETS[dataset_idx] if dataset_idx < len(DATASETS) else "unknown")
    return datasets


def _pick_utility_column(df: pd.DataFrame) -> Optional[str]:
    for c in UTILITY_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def load_lambda_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data: Dict[str, List[pd.DataFrame]] = {'exp': [], 'gum': []}
    mechanism_map = {'del2ph': 'exp', 'delgum': 'gum'}

    for file_prefix, mech_key in mechanism_map.items():
        for i in range(0, 11):
            filepath = data_dir / f"{file_prefix}_{i}.csv"
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath)
            if len(df) == 0:
                continue

            if 'lambda' in df.columns:
                lam = pd.to_numeric(df['lambda'].iloc[0], errors='coerce')
                if not np.isfinite(lam):
                    lam = float(i) / 10.0
            else:
                lam = float(i) / 10.0

            df = df.copy()
            df['lambda'] = float(lam)
            df['mechanism'] = mech_key

            for col in ['leakage', 'mask_size']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            util_col = _pick_utility_column(df)
            if util_col is not None:
                df[util_col] = pd.to_numeric(df[util_col], errors='coerce')
                df['utility'] = df[util_col]
            else:
                df['utility'] = np.nan

            df['dataset'] = _assign_dataset_by_index(len(df))

            expected_rows = len(DATASETS) * SAMPLES_PER_DATASET
            if len(df) != expected_rows:
                print(f"WARNING: {filepath.name} has {len(df)} rows; expected {expected_rows}. "
                      f"Dataset-by-index mapping may be wrong.")

            data[mech_key].append(df)

    out: Dict[str, pd.DataFrame] = {}
    for m in ['exp', 'gum']:
        if data[m]:
            out[m] = pd.concat(data[m], ignore_index=True)
    return out


def load_epsilon_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data: Dict[str, List[pd.DataFrame]] = {'exp': [], 'gum': []}
    mechanism_map = {'edel2ph': 'exp', 'edelgum': 'gum'}

    for file_prefix, mech_key in mechanism_map.items():
        for eps in EPS_VALUES:
            filepath = data_dir / f"{file_prefix}_{eps}.csv"
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath)
            if len(df) == 0:
                continue

            # Handle duplicated header row if present
            if 'epsilon' in df.columns and isinstance(df.iloc[0]['epsilon'], str) and df.iloc[0]['epsilon'] == 'epsilon':
                df = df.iloc[1:].copy()

            df = df.copy()
            if 'epsilon' in df.columns:
                df['epsilon'] = pd.to_numeric(df['epsilon'], errors='coerce')
                if not np.isfinite(df['epsilon'].iloc[0]):
                    df['epsilon'] = float(eps)
            else:
                df['epsilon'] = float(eps)

            for col in ['leakage', 'mask_size']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            util_col = _pick_utility_column(df)
            if util_col is not None:
                df[util_col] = pd.to_numeric(df[util_col], errors='coerce')
                df['utility'] = df[util_col]
            else:
                df['utility'] = np.nan

            df['mechanism'] = mech_key
            df['dataset'] = _assign_dataset_by_index(len(df))

            expected_rows = len(DATASETS) * SAMPLES_PER_DATASET
            if len(df) != expected_rows:
                print(f"WARNING: {filepath.name} has {len(df)} rows; expected {expected_rows}. "
                      f"Dataset-by-index mapping may be wrong.")

            data[mech_key].append(df)

    out: Dict[str, pd.DataFrame] = {}
    for m in ['exp', 'gum']:
        if data[m]:
            out[m] = pd.concat(data[m], ignore_index=True)
    return out

# =============================================================================
# Statistics
# =============================================================================

def compute_stats(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    df = df[df['dataset'].isin(DATASETS)].copy()

    agg = {
        'leakage': ['mean', 'std', 'count'],
        'mask_size': ['mean', 'std'],
        'utility': ['mean', 'std'],
    }
    grouped = df.groupby([xcol, 'dataset']).agg(agg).reset_index()

    # flatten columns
    new_cols = []
    for c in grouped.columns:
        if isinstance(c, tuple):
            new_cols.append(f"{c[0]}_{c[1]}" if c[1] else c[0])
        else:
            new_cols.append(c)
    grouped.columns = new_cols

    # normalize count col name
    for c in list(grouped.columns):
        if c.startswith("leakage_") and c.endswith("count") and c != "leakage_count":
            grouped.rename(columns={c: "leakage_count"}, inplace=True)

    for c in ["utility_mean", "utility_std", "leakage_count"]:
        if c not in grouped.columns:
            grouped[c] = np.nan

    return grouped


def compute_lambda_stats(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {mech: compute_stats(df, "lambda") for mech, df in data.items()}


def compute_epsilon_stats(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {mech: compute_stats(df, "epsilon") for mech, df in data.items()}

# =============================================================================
# Plotting helpers
# =============================================================================

def compute_mask_improvement(stats: Dict[str, pd.DataFrame]) -> None:
    """
    Adds mask_impr_mean (%) to each mechanism stats DF:
      100 * (DelMin - mask) / DelMin   (positive = better than DelMin)
    """
    for mech in ['exp', 'gum']:
        if mech not in stats or stats[mech].empty:
            continue
        dfm = stats[mech].copy()

        def _impr(row):
            base = float(DELMIN_MASK_SIZES[row['dataset']])
            m = row['mask_size_mean']
            if not np.isfinite(m) or base <= 0:
                return np.nan
            return 100.0 * (base - float(m)) / base

        dfm['mask_impr_mean'] = dfm.apply(_impr, axis=1)
        stats[mech] = dfm


def _apply_eps_ticks(ax):
    ax.set_xticks(EPS_TICKS_PLOT)
    ax.set_xticklabels([f"{x:g}" for x in EPS_TICKS_PLOT], fontsize=6, rotation=25, ha='right')
    ax.tick_params(axis='x', which='minor', bottom=False)  # hide minor tick marks


def plot_single_mech_leakage_lines(ax, stats_df: pd.DataFrame, xcol: str, mech: str,
                                  xscale: Optional[str] = None,
                                  xlim: Optional[tuple] = None,
                                  ylim: Optional[tuple] = None,
                                  set_eps_ticks: bool = False):
    """
    5 lines = 5 datasets, one mechanism only.
    """
    if stats_df is None or stats_df.empty:
        return

    st = SINGLE_MECH_STYLE[mech]

    for ds in DATASETS:
        df_ds = stats_df[stats_df['dataset'] == ds].sort_values(xcol)
        if df_ds.empty:
            continue
        ax.plot(
            df_ds[xcol].to_numpy(dtype=float),
            df_ds['leakage_mean'].to_numpy(dtype=float),
            color=DATASET_COLORS[ds],
            linestyle=st["linestyle"],
            linewidth=1.0,
            alpha=0.95
        )

    ax.set_xlabel(r'$\lambda$' if xcol == "lambda" else r'$\varepsilon$', fontsize=7)
    ax.set_ylabel(r'$\mathcal{L}$', fontsize=7, labelpad=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=6)

    if xscale:
        ax.set_xscale(xscale)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if set_eps_ticks and xcol == "epsilon":
        _apply_eps_ticks(ax)


def plot_metric_vs_x(ax, stats: Dict[str, pd.DataFrame], xcol: str, ycol: str,
                     ylabel: str, panel_label: str,
                     xscale: Optional[str] = None,
                     xlim: Optional[tuple] = None,
                     ylim: Optional[tuple] = None,
                     y0_line: Optional[float] = None,
                     set_eps_ticks: bool = False):
    """
    Combined Exp + Gum plot with distinguishable styles (linestyle + marker).
    """
    for mech in ['exp', 'gum']:
        if mech not in stats or stats[mech].empty:
            continue
        dfm = stats[mech]
        st = MECH_STYLE[mech]

        for ds in DATASETS:
            df_ds = dfm[dfm['dataset'] == ds].sort_values(xcol)
            if df_ds.empty or ycol not in df_ds.columns:
                continue
            ax.plot(
                df_ds[xcol].to_numpy(dtype=float),
                df_ds[ycol].to_numpy(dtype=float),
                color=DATASET_COLORS[ds],
                linestyle=st["linestyle"],
                marker=st["marker"],
                markersize=st["markersize"],
                markevery=1,
                linewidth=1.0,
                alpha=0.95,
            )

    if y0_line is not None:
        ax.axhline(y=y0_line, color='black', linestyle=':', linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r'$\lambda$' if xcol == "lambda" else r'$\varepsilon$', fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=6)

    if xscale:
        ax.set_xscale(xscale)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if set_eps_ticks and xcol == "epsilon":
        _apply_eps_ticks(ax)

    ax.text(0.5, -0.32, panel_label, transform=ax.transAxes, ha='center', fontsize=7)

# =============================================================================
# Main Figure
# =============================================================================

def plot_ablation_figure(lambda_stats: Dict[str, pd.DataFrame],
                         epsilon_stats: Dict[str, pd.DataFrame],
                         output_path: Path) -> None:
    """
    Legend row + 2 plot rows (lambda on top, epsilon on bottom).
    Each plot row: [exp_leak_lines | spacer | gum_leak_lines | leakage_combined | mask_impr | utility]
    """
    fig = plt.figure(figsize=(12, 4.8))

    gs = fig.add_gridspec(
        3, 6,
        height_ratios=[0.08, 1, 1],
        width_ratios=[1, 0.01, 1, 1.2, 1.2, 1.2],
        hspace=0.55, wspace=0.35,
        left=0.02, right=0.99, top=0.90, bottom=0.10
    )

    # Legend
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')

    dataset_handles = [
        mpatches.Patch(color=DATASET_COLORS[d], label=DATASET_LABELS[d])
        for d in DATASETS
    ]
    mech_handles = [
        Line2D([0], [0], color='dimgray', linestyle='-',  marker='o', markersize=4, linewidth=1.5, label=SC_EXP),
        Line2D([0], [0], color='dimgray', linestyle='--', marker='s', markersize=4, linewidth=1.5, label=SC_GUM),
        Line2D([0], [0], color='black',   linestyle=':',  linewidth=1.2, label=SC_MIN),
    ]

    ax_legend.legend(
        handles=dataset_handles + mech_handles,
        loc='center', ncol=8, frameon=False, fontsize=7,
        handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
        bbox_to_anchor=(0.72, 0.5)
    )

    # Add mask improvement columns
    compute_mask_improvement(lambda_stats)
    compute_mask_improvement(epsilon_stats)

    # ---------------------------
    # Row 1 (lambda)
    # ---------------------------
    ax_exp_l  = fig.add_subplot(gs[1, 0])
    ax_sp_l   = fig.add_subplot(gs[1, 1])
    ax_gum_l  = fig.add_subplot(gs[1, 2])
    ax_leak_l = fig.add_subplot(gs[1, 3])
    ax_impr_l = fig.add_subplot(gs[1, 4])
    ax_util_l = fig.add_subplot(gs[1, 5])

    ax_sp_l.axis('off')

    if 'exp' in lambda_stats:
        plot_single_mech_leakage_lines(ax_exp_l, lambda_stats['exp'], "lambda", "exp",
                                       xlim=(0.0, 1.0), ylim=(0.0, 1.05))
    ax_exp_l.text(0.5, -0.32, f'(a) {SC_EXP}: Leakage vs $\\lambda$', transform=ax_exp_l.transAxes,
                  ha='center', fontsize=7)

    if 'gum' in lambda_stats:
        plot_single_mech_leakage_lines(ax_gum_l, lambda_stats['gum'], "lambda", "gum",
                                       xlim=(0.0, 1.0), ylim=(0.0, 1.05))
    ax_gum_l.text(0.5, -0.32, f'(b) {SC_GUM}: Leakage vs $\\lambda$', transform=ax_gum_l.transAxes,
                  ha='center', fontsize=7)

    plot_metric_vs_x(
        ax_leak_l, lambda_stats, "lambda", "leakage_mean",
        ylabel=r'$\mathcal{L}$',
        panel_label=r'(c) Leakage vs $\lambda$',
        xlim=(0.0, 1.0), ylim=(0.0, 1.05)
    )

    plot_metric_vs_x(
        ax_impr_l, lambda_stats, "lambda", "mask_impr_mean",
        ylabel=r'$\%\Delta |M|$ vs ' + SC_MIN,
        panel_label=r'(d) Mask Improvement vs $\lambda$',
        xlim=(0.0, 1.0), y0_line=0.0
    )

    any_util_l = any(
        mech in lambda_stats and not lambda_stats[mech].empty and
        np.isfinite(lambda_stats[mech]['utility_mean'].to_numpy(dtype=float)).any()
        for mech in ['exp', 'gum']
    )
    if not any_util_l:
        print("WARNING: No usable utility column found for LAMBDA sweep. "
              f"Expected one of {UTILITY_COL_CANDIDATES}.")

    plot_metric_vs_x(
        ax_util_l, lambda_stats, "lambda", "utility_mean",
        ylabel=r'$\mathcal{U}$',
        panel_label=r'(e) Utility vs $\lambda$',
        xlim=(0.0, 1.0)
    )

    # ---------------------------
    # Row 2 (epsilon) — LOG SCALE ON ALL PANELS
    # ---------------------------
    ax_exp_e  = fig.add_subplot(gs[2, 0])
    ax_sp_e   = fig.add_subplot(gs[2, 1])
    ax_gum_e  = fig.add_subplot(gs[2, 2])
    ax_leak_e = fig.add_subplot(gs[2, 3])
    ax_impr_e = fig.add_subplot(gs[2, 4])
    ax_util_e = fig.add_subplot(gs[2, 5])

    ax_sp_e.axis('off')

    if 'exp' in epsilon_stats:
        plot_single_mech_leakage_lines(ax_exp_e, epsilon_stats['exp'], "epsilon", "exp",
                                       xscale='log', xlim=(0.05, 32), ylim=(0.0, 1.05),
                                       set_eps_ticks=True)
    ax_exp_e.text(0.5, -0.32, f'(f) {SC_EXP}: Leakage vs $\\varepsilon$', transform=ax_exp_e.transAxes,
                  ha='center', fontsize=7)

    if 'gum' in epsilon_stats:
        plot_single_mech_leakage_lines(ax_gum_e, epsilon_stats['gum'], "epsilon", "gum",
                                       xscale='log', xlim=(0.05, 32), ylim=(0.0, 1.05),
                                       set_eps_ticks=True)
    ax_gum_e.text(0.5, -0.32, f'(g) {SC_GUM}: Leakage vs $\\varepsilon$', transform=ax_gum_e.transAxes,
                  ha='center', fontsize=7)

    plot_metric_vs_x(
        ax_leak_e, epsilon_stats, "epsilon", "leakage_mean",
        ylabel=r'$\mathcal{L}$',
        panel_label=r'(h) Leakage vs $\varepsilon$',
        xscale='log', xlim=(0.05, 32), ylim=(0.0, 1.05),
        set_eps_ticks=True
    )

    plot_metric_vs_x(
        ax_impr_e, epsilon_stats, "epsilon", "mask_impr_mean",
        ylabel=r'$\%\Delta |M|$ vs ' + SC_MIN,
        panel_label=r'(i) Mask Improvement vs $\varepsilon$',
        xscale='log', xlim=(0.05, 32), y0_line=0.0,
        set_eps_ticks=True
    )

    any_util_e = any(
        mech in epsilon_stats and not epsilon_stats[mech].empty and
        np.isfinite(epsilon_stats[mech]['utility_mean'].to_numpy(dtype=float)).any()
        for mech in ['exp', 'gum']
    )
    if not any_util_e:
        print("WARNING: No usable utility column found for EPSILON sweep. "
              f"Expected one of {UTILITY_COL_CANDIDATES}.")

    plot_metric_vs_x(
        ax_util_e, epsilon_stats, "epsilon", "utility_mean",
        ylabel=r'$\mathcal{U}$',
        panel_label=r'(j) Utility vs $\varepsilon$',
        xscale='log', xlim=(0.05, 32),
        set_eps_ticks=True
    )

    # Save
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"Saved: {output_path}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate 2-row ablation figure (lambda top, epsilon bottom)')
    parser.add_argument('--data_dir', type=str, default='./data3',
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
    eps_data = load_epsilon_data(data_dir)
    epsilon_stats = compute_epsilon_stats(eps_data)

    print("Generating ablation figure...")
    plot_ablation_figure(lambda_stats, epsilon_stats, output_path)

    print("\nDone!")

if __name__ == '__main__':
    main()
