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
# STYLE (tight kept)
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
    "savefig.bbox": "tight",   # KEPT
    "axes.grid": True,
    "grid.alpha": 0.3,
})

plt.rcParams["mathtext.fontset"] = "stix"

# =============================================================================
# CONFIG
# =============================================================================
MIN_MASK = {'airport': 5, 'hospital': 9, 'adult': 9, 'flight': 11, 'tax': 3}
DATASETS = ['airport', 'hospital', 'adult', 'flight', 'tax']

# ORIGINAL HEATMAP COLORS (unchanged)
PASTEL_COLORS = ['#FFF8F0', '#F0F7EC', '#D4EDDA', '#A8D8B9',
                 '#7BC89C', '#4DAF7A', '#2D8B57']
PASTEL_CMAP = LinearSegmentedColormap.from_list(
    'pastel_green', PASTEL_COLORS, N=256
)

MECH_COLOR = {'Exp': '#1f77b4', 'Gum': '#ff7f0e'}

TAU_CONTOURS = [0.15, 0.32, 0.52, 0.73]
TAU_COLORS = {
    0.15: '#d62728',   # red (keep)
    0.32: '#0072B2',   # strong blue (pops against green)
    0.52: '#9467bd',   # purple (keep)
    0.73: '#17BECF'    # black (high contrast, clean)
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
# TAU
# =============================================================================
def tau_fn(eps_m, L0):
    a = np.exp(eps_m) * L0
    return a / ((1 - L0) + a)

# # =============================================================================
# # HEATMAP (GridSpec version – tight compatible)
# # =============================================================================
# def plot_heatmap(df):
#
#     eps_vals = sorted(df['epsilon_m'].unique())
#     L0_vals = sorted(df['L0'].unique())
#
#     df["improvement"] = (
#         100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
#     )
#
#     agg = df.groupby(
#         ['dataset', 'mechanism', 'epsilon_m', 'L0']
#     )['improvement'].mean().reset_index()
#
#     fig = plt.figure(figsize=(26, 7))
#
#     gs = GridSpec(
#         2, 6,
#         width_ratios=[1,1,1,1,1,0.06],
#         wspace=0.20,
#         hspace=0.17
#     )
#
#     axes = np.empty((2,5), dtype=object)
#
#     for r in range(2):
#         for c in range(5):
#             axes[r,c] = fig.add_subplot(gs[r,c])
#
#     norm = Normalize(vmin=0, vmax=75)
#
#     for col, ds in enumerate(DATASETS):
#         for row, mech in enumerate(['Exp','Gum']):
#             ax = axes[row, col]
#             ax.tick_params(axis='y', left=True, labelleft=True)
#             if row == 0:
#                 ax.set_title(ds.capitalize(), pad = 6, fontweight = 'bold')
#             sub = agg[(agg.dataset==ds)&(agg.mechanism==mech)]
#             pivot = sub.pivot_table(
#                 index='L0', columns='epsilon_m',
#                 values='improvement'
#             ).reindex(index=L0_vals[:-1], columns=eps_vals)
#
#             im = ax.imshow(
#                 pivot.values,
#                 cmap=PASTEL_CMAP,
#                 norm=norm,
#                 origin='lower',
#                 aspect='auto'
#             )
#
#             ax.set_xticks(range(len(eps_vals)))
#             ax.set_xticklabels(eps_vals, rotation=45)
#             ax.set_yticks(range(len(L0_vals) -1))
#             ax.set_yticklabels(L0_vals[:-1])
#
#             if col==0:
#                 ax.set_ylabel(f"{mech}\nRe-inference Leakage Threshold $L_0$")
#
#             if row==1:
#                 ax.set_xlabel(r"Masking-Privacy Budget $\epsilon_m$")
#
#             # -------- FIXED τ CONTOURS --------
#             L0_plot = L0_vals[:-1]
#
#             for tau in TAU_CONTOURS:
#
#                 x_coords = []
#                 y_coords = []
#
#                 for ei, em in enumerate(eps_vals):
#
#                     l0 = tau / (np.exp(em) * (1 - tau) + tau)
#
#                     if L0_plot[0] <= l0 <= L0_plot[-1]:
#                         yp = np.interp(
#                             l0,
#                             L0_plot,
#                             np.arange(len(L0_plot))
#                         )
#
#                         x_coords.append(ei)
#                         y_coords.append(yp)
#
#                 if len(x_coords) >= 2:
#                     ax.plot(
#                         x_coords,
#                         y_coords,
#                         linestyle = '--',
#                         linewidth = 2.5,
#                         color = TAU_COLORS[tau],
#                         zorder = 5,
#                         clip_on = False
#                     )
#     # Colorbar inside GridSpec
#     cbar_ax = fig.add_subplot(gs[:,5])
#     cbar = fig.colorbar(im, cax=cbar_ax)
#     cbar.set_label("Mask Improvement (%)")
#
#     # τ legend
#     handles = [
#         Line2D([0],[0],
#                color=TAU_COLORS[t],
#                ls='--',
#                lw=2,
#                label=f"$\\tau={t}$")
#         for t in TAU_CONTOURS
#     ]
#
#     fig.legend(
#         handles=handles,
#         labels=[h.get_label() for h in handles],
#         loc="upper center",
#         ncol=len(handles),
#         frameon=True,
#         bbox_to_anchor=(0.5,0.98)
#     )
#
#     plt.savefig("heatmap_improvement.pdf")
#     plt.close()
#
# # =============================================================================
# # BUDGET BARS
# # =============================================================================
# def plot_budget(df):
#
#     # Compute improvement first
#     df["improvement"] = (
#         100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
#     )
#
#     agg = df.groupby(
#         ['dataset','mechanism','epsilon_m','L0']
#     )['improvement'].mean().reset_index()
#
#     agg['tau'] = agg.apply(
#         lambda r: tau_fn(r['epsilon_m'], r['L0']),
#         axis=1
#     )
#
#     tau_targets = TAU_CONTOURS
#     tol = 0.04
#
#     fig, axes = plt.subplots(2,5,
#                              figsize=(26,7),
#                              sharey='row')
#
#     for row, mech in enumerate(['Exp','Gum']):
#         for col, ds in enumerate(DATASETS):
#
#             ax = axes[row,col]
#             ax.tick_params(axis='y', left=True, labelleft=True)
#
#             if row == 0:
#                 ax.set_title(ds.capitalize(),
#                              pad=6,
#                              fontweight='bold')
#
#             sub = agg[(agg.dataset==ds)&
#                       (agg.mechanism==mech)]
#
#             best, worst = [], []
#
#             for t in tau_targets:
#                 close = sub[np.abs(sub['tau']-t)<=tol]
#
#                 if len(close) < 2:
#                     best.append(0)
#                     worst.append(0)
#                     continue
#
#                 best.append(close['improvement'].max())
#                 worst.append(close['improvement'].min())
#
#             x = np.arange(len(tau_targets))
#
#             ax.bar(x, worst, 0.35,
#                    color='#d62728',
#                    label='Worst split')
#
#             ax.bar(x+0.35, best, 0.35,
#                    color='#2ca02c',
#                    label='Best split')
#
#             ax.set_xticks(x+0.175)
#             ax.set_xticklabels([f"τ≈{t}" for t in tau_targets])
#
#             # 🔥 Y label changed
#             if col == 0:
#                 ax.set_ylabel(f"{mech}\nMask Improvement (%)")
#
#             ax.set_ylim(0, 100)
#
#     handles, labels = axes[0,0].get_legend_handles_labels()
#
#     fig.legend(handles=handles,
#                labels=labels,
#                loc='upper center',
#                frameon=True,
#                bbox_to_anchor=(0.5,0.97),
#                ncol=2)
#
#     plt.savefig("budget_split_bars.pdf")
#     plt.close()

# =============================================================================
# PARETO
# =============================================================================
# =============================================================================
# PARETO (Baseline-normalized % version)
# =============================================================================
# =============================================================================
# PARETO (Baseline-normalized % version — fixed scaling + green diamond)
# =============================================================================
def plot_pareto(df):

    EM_VALS = [0.1, 1.0]

    K_SIZE = {
        "airport": 5,
        "hospital": 9,
        "adult": 9,
        "flight": 11,
        "tax": 3,
    }

    TARGET_ONLY_LEAK = {
        "airport": 0.49830908053361767,
        "hospital": 0.6623992294064142,
        "adult": 0.49122554744404323,
        "flight": 0.9826408586840785,
        "tax": 0.5548418449270073,
    }

    agg = df[df.epsilon_m.isin(EM_VALS)].groupby(
        ['dataset', 'mechanism', 'epsilon_m', 'L0']
    ).agg(
        mean_mask=('mask_size', 'mean'),
        mean_leak=('leakage', 'mean')
    ).reset_index()

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 2.75))

    for col, ds in enumerate(DATASETS):

        ax = axes[col]
        ax.tick_params(axis='y', left=True, labelleft=True)
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", pad=2, fontweight='bold', fontsize=FS + 2)

        baseline = K_SIZE[ds]

        # -------------------------------------------------
        # Curves
        # -------------------------------------------------
        for mech in ['Exp', 'Gum']:
            for em in EM_VALS:

                sub = agg[
                    (agg.dataset == ds) &
                    (agg.mechanism == mech) &
                    (agg.epsilon_m == em)
                ].sort_values('L0')

                if len(sub) == 0:
                    continue

                mask_pct = 100 * sub['mean_mask'] / baseline

                ax.plot(
                    sub['mean_leak'],
                    mask_pct,
                    marker='o',
                    linestyle='--' if em == 0.1 else '-',
                    color=MECH_COLOR[mech],
                    label=f"{mech}, ε={em}"
                )

        # -------------------------------------------------
        # 🔴 Baseline  M_MIN (M_det)
        # -------------------------------------------------
        ax.scatter(
            0,
            100,
            marker='*',
            s=240,
            color='red',
            edgecolor='black',
            zorder=6,
            label=r"$M_{\mathrm{MIN}}\,(M_{\mathrm{det}})$" if col == 0 else None
        )

        # -------------------------------------------------
        # 🟢 Target-only  M = empty set
        # -------------------------------------------------
        ax.scatter(
            TARGET_ONLY_LEAK[ds],
            0,
            marker='D',
            s=110,
            color='#2ca02c',
            edgecolor='black',
            zorder=6,
            label=r"$M = \emptyset$" if col == 0 else None
        )
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        #ax.set_xlabel(r"Expected Leakage ($\mathbb{E}[\mathcal{L}]$)", fontsize=FS + 2)

        if col == 0:
            ax.set_ylabel("Mask Size (% of Baseline)", fontsize=FS + 3)

        ax.set_ylim(-8, 124)


    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles=handles,
        labels=labels,
        loc='upper center',
        frameon=True,
        bbox_to_anchor=(0.5, 1.1),
        ncol=6
    )
    # Universal x-axis label
    fig.supxlabel(r"Expected Leakage ($\mathbb{E}[\mathcal{L}]$)", y = -0.06, fontsize = FS + 2)

    #plt.subplots_adjust(top = 0.75, bottom = 0.18)
    plt.savefig("pareto_frontier.pdf")
    plt.close()
# =============================================================================
# BUDGET TABLE (Numerical version of bar plot)
# =============================================================================
def generate_budget_table(df):

    print("Generating budget split table (tau = 0.22)...")

    # Same improvement computation
    df["improvement"] = (
        100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
    )

    agg = df.groupby(
        ['dataset','mechanism','epsilon_m','L0']
    )['improvement'].mean().reset_index()

    # Same tau computation
    agg['tau'] = agg.apply(
        lambda r: tau_fn(r['epsilon_m'], r['L0']),
        axis=1
    )

    TARGET_TAU = 0.22
    tol = 0.04

    rows = []

    for mech in ['Exp','Gum']:
        for ds in DATASETS:

            sub = agg[(agg.dataset==ds) &
                      (agg.mechanism==mech)]

            close = sub[np.abs(sub['tau'] - TARGET_TAU) <= tol]

            if len(close) < 2:
                continue

            # ---- Best Split ----
            best_row = close.loc[
                close['improvement'].idxmax()
            ]

            # ---- Worst Split ----
            worst_row = close.loc[
                close['improvement'].idxmin()
            ]

            rows.append({
                "Dataset": ds,
                "Mechanism": mech,
                "Total Budget (tau)": round(TARGET_TAU, 3),

                "Best ε_m": best_row['epsilon_m'],
                "Best L0": best_row['L0'],
                "Best Mask Improvement (%)": round(best_row['improvement'], 2),

                "Worst ε_m": worst_row['epsilon_m'],
                "Worst L0": worst_row['L0'],
                "Worst Mask Improvement (%)": round(worst_row['improvement'], 2),
            })

    table_df = pd.DataFrame(rows)

    table_df.to_csv("budget_split_table_tau_022.csv", index=False)

    print("Saved: budget_split_table_tau_022.csv")
# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    df = load_data()
   # plot_heatmap(df)
    #plot_budget(df)
    generate_budget_table(df)
    plot_pareto(df)
    print("All figures generated.")