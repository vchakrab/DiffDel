#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from pathlib import Path

# =============================================================================
# STYLE
# =============================================================================
FS = 12

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "legend.fontsize": FS,
    "legend.title_fontsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

plt.rcParams["mathtext.fontset"] = "stix"

# =============================================================================
# CONFIG
# =============================================================================
MIN_MASK = {'airport': 5, 'hospital': 9, 'flight': 11}
DATASETS = ['airport', 'hospital', 'flight']

PASTEL_COLORS = ['#FFF8F0', '#F0F7EC', '#D4EDDA', '#A8D8B9',
                 '#7BC89C', '#4DAF7A', '#2D8B57']
PASTEL_CMAP = LinearSegmentedColormap.from_list(
    'pastel_green', PASTEL_COLORS, N=256
)

MECH_COLOR = {'Exp': '#1f77b4', 'Gum': '#ff7f0e'}

TAU_CONTOURS = [0.15, 0.32, 0.52, 0.73]
TAU_COLORS = {
    0.15: '#d62728',
    0.32: '#0072B2',
    0.52: '#9467bd',
    0.73: '#17BECF'
}

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data():

    BASE_DIR = Path(__file__).resolve().parent
    EXP_DIR = BASE_DIR / "exp"
    GUM_DIR = BASE_DIR / "gumbel"

    required_cols = ["dataset", "mask_size", "leakage", "epsilon_m", "L0"]

    def load_folder(folder_path, mechanism_name):
        frames = []
        for csv_file in folder_path.rglob("*.csv"):
            df = pd.read_csv(csv_file)
            if not all(c in df.columns for c in required_cols):
                continue
            df = df[df["epsilon_m"] > 0].copy()
            df["mechanism"] = mechanism_name
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    exp = load_folder(EXP_DIR, "Exp")
    gum = load_folder(GUM_DIR, "Gum")

    df = pd.concat([exp, gum], ignore_index=True)
    df["min_mask"] = df["dataset"].map(MIN_MASK)

    return df

# =============================================================================
# TAU FUNCTION
# =============================================================================
def tau_fn(eps_m, L0):
    a = np.exp(eps_m) * L0
    return a / ((1 - L0) + a)

# =============================================================================
# SECTION HEADERS (Centered over 3-plot blocks)
# =============================================================================
# =============================================================================
# SECTION HEADERS (Centered + Underlined)
# =============================================================================
def add_section_headers(fig):
    fig.text(0.333, 1.4, "Exp",
             ha='center', va='center',
             fontsize=FS+2, fontweight='bold')

    fig.text(0.666, 1.4, "Gumbel",
             ha='center', va='center',
             fontsize=FS+2, fontweight='bold')

    # Underlines
    fig.lines.extend([
        plt.Line2D([0.285, 0.381], [0.928, 0.928],
                   transform=fig.transFigure,
                   color='black', linewidth=1.5),
        plt.Line2D([0.618, 0.714], [0.928, 0.928],
                   transform=fig.transFigure,
                   color='black', linewidth=1.5)
    ])
def plot_heatmap(df):

    eps_vals = sorted(df['epsilon_m'].unique())
    L0_vals = sorted(df['L0'].unique())
    L0_plot = L0_vals[:-1]

    df["improvement"] = (
        100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
    )

    agg = df.groupby(
        ['dataset', 'mechanism', 'epsilon_m', 'L0']
    )['improvement'].mean().reset_index()

    fig = plt.figure(figsize=(26, 3))

    gs = GridSpec(
        1, 7,
        width_ratios=[1,1,1,1,1,1,0.06],
        wspace=0.22
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    norm = Normalize(vmin=0, vmax=75)

    ordered = (
        [(ds, 'Exp') for ds in DATASETS] +
        [(ds, 'Gum') for ds in DATASETS]
    )

    for i, (ds, mech) in enumerate(ordered):

        ax = axes[i]

        # ---- Updated Title ----
        if mech == 'Exp':
            ax.set_title(f"{ds.capitalize()} ({mech})",
                     pad=8,
                     fontweight='bold')
        else:
            ax.set_title(f"{ds.capitalize()} ({mech})",
                         pad = 8,
                         fontweight = 'bold')

        sub = agg[(agg.dataset==ds)&(agg.mechanism==mech)]

        pivot = sub.pivot_table(
            index='L0',
            columns='epsilon_m',
            values='improvement'
        ).reindex(index=L0_plot, columns=eps_vals)

        im = ax.imshow(
            pivot.values,
            cmap=PASTEL_CMAP,
            norm=norm,
            origin='lower',
            aspect='auto'
        )

        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels(eps_vals, rotation=45)
        ax.set_yticks(range(len(L0_plot)))
        ax.set_yticklabels(L0_plot)

        if i == 0:
            ax.set_ylabel(r"Re-inference Leakage Threshold $L_0$")

        ax.set_xlabel(r"Masking-Privacy Budget $\epsilon_m$")

        # ---- τ Contours ----
        for tau in TAU_CONTOURS:

            x_coords = []
            y_coords = []

            for ei, em in enumerate(eps_vals):
                l0 = tau / (np.exp(em)*(1-tau) + tau)

                if L0_plot[0] <= l0 <= L0_plot[-1]:
                    yp = np.interp(
                        l0,
                        L0_plot,
                        np.arange(len(L0_plot))
                    )
                    x_coords.append(ei)
                    y_coords.append(yp)

            if len(x_coords) >= 2:
                ax.plot(
                    x_coords,
                    y_coords,
                    linestyle='--',
                    linewidth=2.5,
                    color=TAU_COLORS[tau],
                    zorder=5,
                    clip_on=False
                )

    # ---- Colorbar ----
    cbar_ax = fig.add_subplot(gs[0,6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Mask Improvement (%)")

    # ---- Standard Matplotlib Legend on Top ----
    handles = [
        Line2D([0], [0],
               color=TAU_COLORS[t],
               linestyle='--',
               linewidth=2.5,
               label=rf"$\tau={t}$")
        for t in TAU_CONTOURS
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(TAU_CONTOURS),
        frameon=True
    )

    plt.subplots_adjust(top=0.78)

    plt.savefig("heatmap_improvement.pdf")
    plt.close()

def plot_budget(df):

    df["improvement"] = (
        100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
    )

    agg = df.groupby(
        ['dataset','mechanism','epsilon_m','L0']
    )['improvement'].mean().reset_index()

    agg['tau'] = agg.apply(
        lambda r: tau_fn(r['epsilon_m'], r['L0']),
        axis=1
    )

    fig = plt.figure(figsize=(26, 3))
    gs = GridSpec(1, 6, wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]

    ordered = (
        [(ds, 'Exp') for ds in DATASETS] +
        [(ds, 'Gum') for ds in DATASETS]
    )

    for i, (ds, mech) in enumerate(ordered):

        ax = axes[i]
        ax.set_title(ds.capitalize(), pad=10, fontweight='bold')

        sub = agg[(agg.dataset==ds)&(agg.mechanism==mech)]

        best, worst = [], []
        tol = 0.04

        for t in TAU_CONTOURS:
            close = sub[np.abs(sub['tau']-t)<=tol]
            if len(close) < 2:
                best.append(0)
                worst.append(0)
                continue
            best.append(close['improvement'].max())
            worst.append(close['improvement'].min())

        x = np.arange(len(TAU_CONTOURS))

        ax.bar(x, worst, 0.35,
               color='#d62728',
               label='Worst split')

        ax.bar(x+0.35, best, 0.35,
               color='#2ca02c',
               label='Best split')

        ax.set_xticks(x+0.175)
        ax.set_xticklabels([f"τ≈{t}" for t in TAU_CONTOURS])

        if i == 0:
            ax.set_ylabel("Mask Improvement (%)")

        ax.set_ylim(0,100)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles = handles,
        loc = "center left",
        bbox_to_anchor = (0.012, 0.5),
        frameon = True,
        ncol = 1,
        borderaxespad = 0.0,
        handlelength = 1.2,
        handletextpad = 0.4,
        borderpad = 0.3
    )

    add_section_headers(fig)
    plt.subplots_adjust(top=0.80)

    plt.savefig("budget_split_bars.pdf")
    plt.close()

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    df = load_data()
    plot_heatmap(df)
    plot_budget(df)
    print("Heatmap and Budget generated.")