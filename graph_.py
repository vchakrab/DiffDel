#!/usr/bin/env python3
"""
generate_new_plots.py
Creates 4 separate PDFs (1×5 layout each):

1) fig_exp_mask.pdf
2) fig_gum_mask.pdf
3) fig_exp_leakage.pdf
4) fig_gum_leakage.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# =============================================================================
# STYLE (UNCHANGED)
# =============================================================================
FS = 16

plt.rcParams.update({
    "font.family":           "STIXGeneral",
    "font.size":             FS,
    "axes.labelsize":        FS,
    "axes.titlesize":        FS,
    "legend.fontsize":       FS,
    "legend.title_fontsize": FS,
    "xtick.labelsize":       FS,
    "ytick.labelsize":       FS,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "axes.grid":             True,
    "grid.alpha":            0.3,
})
plt.rcParams["mathtext.fontset"] = "stix"

DATASETS_5 = ["airport", "hospital", "adult", "flight", "tax"]
MIN_MASK = {"airport": 5, "hospital": 9, "adult": 9, "flight": 11, "tax": 3}

# =============================================================================
# DATA LOADER
# =============================================================================
def load_curves_data(datasets) -> pd.DataFrame:
    records = []
    for method in ["exp", "gum"]:
        for dataset in datasets:
            path = DATA_DIR / method / dataset / "full_data.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df["epsilon_m"] != 0]
            if df.empty:
                continue
            delmin = MIN_MASK[dataset]
            for (eps, L0), grp in df.groupby(["epsilon_m","L0"]):
                n = len(grp)
                mean_mask = grp["mask_size"].mean()
                mean_leak = grp["leakage"].mean()
                std_mask = grp["mask_size"].std()
                std_leak = grp["leakage"].std()
                ci_mask = 1.96 * std_mask / np.sqrt(n)
                ci_leak = 1.96 * std_leak / np.sqrt(n)
                records.append({
                    "method": method,
                    "dataset": dataset,
                    "epsilon_m": eps,
                    "L0": L0,
                    "improvement": 100 * abs(delmin - mean_mask) / delmin,
                    "ci_improvement": 100 * ci_mask / delmin,
                    "mean_leakage": mean_leak,
                    "ci_leakage": ci_leak,
                })
    return pd.DataFrame(records)

# =============================================================================
# MASK PLOT (1×5)
# =============================================================================
def plot_mask(df, mech):
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(wspace=0.18,
                        left=0.07, right=0.97, top=0.82, bottom=0.15)

    subset_all = df[df["method"] == mech]

    for col, ds in enumerate(DATASETS_5):
        ax = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(curve["L0"], curve["improvement"],
                    marker="o",
                    label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"],
                            curve["improvement"] - curve["ci_improvement"],
                            curve["improvement"] + curve["ci_improvement"],
                            alpha=0.18)

        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=FS)
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xlim(0.0,0.9)
        ax.set_ylim(0,100)
        ax.set_yticks([0,25,50,75,100])

        if col == 0:
            ax.set_ylabel("Mask Size Improvement (%)")
        else:
            ax.set_ylabel(None)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center",
               ncol=len(handles),
               frameon=True,
               bbox_to_anchor=(0.5,1.05))

    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=FS)

    return fig

# =============================================================================
# LEAKAGE PLOT (1×5)
# =============================================================================
def plot_leakage(df, mech):
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(wspace=0.18,
                        left=0.07, right=0.97, top=0.82, bottom=0.15)

    subset_all = df[df["method"] == mech]

    for col, ds in enumerate(DATASETS_5):
        ax = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean = 100 * curve["mean_leakage"]
            lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean, marker="o")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)

        ax.plot([0,1],[0,100], linestyle="--", linewidth=1)

        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=FS)
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xlim(0.0,0.9)
        ax.set_ylim(0,80)
        ax.set_yticks([0,20,40,60,80])

        if col == 0:
            ax.set_ylabel("Achieved Re-inference Leakage (%)", fontsize=FS - 1)
        else:
            ax.set_ylabel(None)

    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=FS)

    return fig

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    df = load_curves_data(DATASETS_5)

    if not df.empty:
        plot_mask(df, "exp").savefig("fig_exp_mask.pdf")
        plt.close()

        plot_mask(df, "gum").savefig("fig_gum_mask.pdf")
        plt.close()

        plot_leakage(df, "exp").savefig("fig_exp_leakage.pdf")
        plt.close()

        plot_leakage(df, "gum").savefig("fig_gum_leakage.pdf")
        plt.close()

    print("Done.")