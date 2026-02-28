#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["airport", "hospital", "flight"]
METHODS = ["exp", "gumbel"]

delmin_masks = {
    "airport": 5,
    "hospital": 9,
    "flight": 11
}

# =============================================================================
# LOAD DATA (BOTH METHODS TOGETHER)
# =============================================================================
def load_all_methods():

    records = []

    for method in METHODS:

        for dataset in DATASETS:

            path = os.path.join(BASE_PATH, method, dataset, "full_data.csv")

            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            df = df[df["epsilon_m"] != 0]

            if df.empty:
                continue

            for (eps, L0), group in df.groupby(["epsilon_m", "L0"]):

                n = len(group)

                mean_mask = group["mask_size"].mean()
                std_mask = group["mask_size"].std()

                mean_leakage = group["leakage"].mean()
                std_leakage = group["leakage"].std()

                ci_mask = 1.96 * std_mask / np.sqrt(n)
                ci_leakage = 1.96 * std_leakage / np.sqrt(n)

                delmin = delmin_masks[dataset]
                improvement = 100 * abs(delmin - mean_mask) / delmin
                ci_improvement = 100 * ci_mask / delmin

                records.append({
                    "method": method,
                    "dataset": dataset,
                    "epsilon_m": eps,
                    "L0": L0,
                    "improvement": improvement,
                    "ci_improvement": ci_improvement,
                    "mean_leakage": mean_leakage,
                    "ci_leakage": ci_leakage
                })

    return pd.DataFrame(records)

# =============================================================================
# MASK IMPROVEMENT (6 PANELS)
# =============================================================================
def plot_mask():

    df = load_all_methods()

    fig, axes = plt.subplots(1, 6, figsize=(26, 3), sharey=False)

    ordered = (
        [(ds, "exp") for ds in DATASETS] +
        [(ds, "gumbel") for ds in DATASETS]
    )

    for i, (dataset, mech) in enumerate(ordered):

        ax = axes[i]
        subset = df[
            (df["dataset"] == dataset) &
            (df["method"] == mech)
        ]

        for eps in sorted(subset["epsilon_m"].unique()):

            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")

            ax.plot(
                curve["L0"],
                curve["improvement"],
                marker='o',
                label=rf"$\varepsilon_m = {eps}$"
            )

            ax.fill_between(
                curve["L0"],
                curve["improvement"] - curve["ci_improvement"],
                curve["improvement"] + curve["ci_improvement"],
                alpha=0.18
            )

        ax.set_title(f"{dataset.capitalize()} ({mech.capitalize()})", pad=2)
        if mech == "gumbel":
            ax.set_title(f"{dataset.capitalize()} (Gum)", pad = 2)
        ax.set_xlabel(r"Re-inference Leakage Threshold$L_0$")
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)

        if i == 0:
            ax.set_ylabel("Mask Size Improvement (%)")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.subplots_adjust(top=0.82, wspace=0.25)

    with PdfPages("mask_improvement.pdf") as pdf:
        pdf.savefig(fig)

    plt.close()

# =============================================================================
# LEAKAGE (6 PANELS)
# =============================================================================
def plot_leakage():

    df = load_all_methods()

    fig, axes = plt.subplots(1, 6, figsize=(26, 3.0), sharey=False)

    ordered = (
        [(ds, "exp") for ds in DATASETS] +
        [(ds, "gumbel") for ds in DATASETS]
    )

    for i, (dataset, mech) in enumerate(ordered):

        ax = axes[i]
        subset = df[
            (df["dataset"] == dataset) &
            (df["method"] == mech)
        ]

        for eps in sorted(subset["epsilon_m"].unique()):

            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")

            mean_leak = 100 * curve["mean_leakage"]
            lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])

            ax.plot(curve["L0"], mean_leak, marker='o')

            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)

        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)

        ax.set_title(f"{dataset.capitalize()} ({mech.capitalize()})", pad=2)
        if mech == "gumbel":
            ax.set_title(f"{dataset.capitalize()} (Gum)", pad=2)
        ax.set_xlabel(r"Re-inference Leakage Threshold $L_0$")
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)

        if i == 0:
            ax.set_ylabel("Achieved Re-inference Leakage (%)")

    plt.subplots_adjust(top=0.85, wspace=0.25)

    with PdfPages("leakage.pdf") as pdf:
        pdf.savefig(fig)

    plt.close()

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":

    plot_mask()
    plot_leakage()

    print("Generated combined 6-panel PDFs.")