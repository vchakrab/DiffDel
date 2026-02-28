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
    "font.family": "serif",
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
    "axes.labelpad": 1.5,
})

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

# =============================================================================
# CONFIG
# =============================================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]
METHODS = ["exp", "gumbel"]

delmin_masks = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3
}

# =============================================================================
# LOAD DATA
# =============================================================================
def load_method_df(method):

    records = []

    for dataset in DATASETS:

        path = os.path.join(BASE_PATH, method, dataset, "full_data.csv")

        if not os.path.exists(path):
            print(f"Skipping missing: {path}")
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
# MASK IMPROVEMENT PLOT
# =============================================================================
def plot_mask(df, method_name):

    fig, axes = plt.subplots(1, 5, figsize=(22, 3.2), sharey=False)

    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):

        subset = df[df["dataset"] == dataset]

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

        ax.set_title(dataset.capitalize(), pad=2)
        ax.set_xlabel(r"Re-inference Threshold $L_0$")
        ax.set_xlim(0.0, 0.9)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        # ✅ Only far-left gets y-label
        if i == 0:
            ax.set_ylabel("Mask Size Improvement (%)")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
        borderpad=0.3
    )

    plt.subplots_adjust(left=0.08, top=0.84, wspace=0.25)

    with PdfPages(f"{method_name}_mask_improvement.pdf") as pdf:
        pdf.savefig(fig)

    plt.close()
def plot_leakage(df, method_name):

    fig, axes = plt.subplots(1, 5, figsize=(22, 2.8), sharey=False)

    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):

        subset = df[df["dataset"] == dataset]

        for eps in sorted(subset["epsilon_m"].unique()):

            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")

            mean_leak = 100 * curve["mean_leakage"]
            lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])

            ax.plot(curve["L0"], mean_leak, marker='o')
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)

        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)

        ax.set_title(dataset.capitalize(), pad=2)
        ax.set_xlabel(r"Re-inference Threshold $L_0$")
        ax.set_xlim(0.0, 0.9)

        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        # EXACT SAME LOGIC AS MASK
        if i == 0:
            ax.set_ylabel("Achieved Re-inference Leakage (%)")

    plt.subplots_adjust(left=0.08, top=0.84, wspace=0.25)

    with PdfPages(f"{method_name}_leakage.pdf") as pdf:
        pdf.savefig(fig)

    plt.close()
if __name__ == "__main__":

    for method in METHODS:

        df = load_method_df(method)

        if df.empty:
            print(f"No data for {method}")
            continue

        plot_mask(df, method)
        plot_leakage(df, method)

    print("Generated:")
    print("  exp_mask_improvement.pdf")
    print("  exp_leakage.pdf")
    print("  gumbel_mask_improvement.pdf")
    print("  gumbel_leakage.pdf")