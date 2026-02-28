import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# STYLE (unchanged)
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
ZIP_PATH = "Archive.zip"
EXTRACT_PATH = "tmp_extract"

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]

delmin_masks = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3
}

METHODS = ["exp", "gumbel"]

# =============================================================================
# Extract ZIP
# =============================================================================
if os.path.exists(EXTRACT_PATH):
    import shutil
    shutil.rmtree(EXTRACT_PATH)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_PATH)

# =============================================================================
# Load Data
# =============================================================================
def load_method_df(method):

    records = []

    for dataset in DATASETS:
        path = os.path.join(EXTRACT_PATH, method, dataset, "full_data.csv")

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        # Ignore epsilon_m == 0
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
# Plot Mask Improvement (unchanged structure)
# =============================================================================
def plot_mask(df, method_name):

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)

    for ax, dataset in zip(axes, DATASETS):

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

    axes[0].set_ylabel("Mask Size Improvement (%)")

    handles, labels = axes[0].get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        bbox_to_anchor=(0.5, 0.99),
        borderpad=0.3
    )

    plt.subplots_adjust(top=0.84, wspace=0.15)

    with PdfPages(f"{method_name}_mask_improvement.pdf") as pdf:
        pdf.savefig(fig)

    plt.close()

# =============================================================================
# Plot Leakage (unchanged structure)
# =============================================================================
def plot_leakage(df, method_name):

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)

    for ax, dataset in zip(axes, DATASETS):

        subset = df[df["dataset"] == dataset]

        for eps in sorted(subset["epsilon_m"].unique()):

            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")

            ax.plot(
                curve["L0"],
                curve["mean_leakage"],
                marker='o',
                label=rf"$\varepsilon_m = {eps}$"
            )

            ax.fill_between(
                curve["L0"],
                curve["mean_leakage"] - curve["ci_leakage"],
                curve["mean_leakage"] + curve["ci_leakage"],
                alpha=0.18
            )

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

        ax.set_title(dataset.capitalize(), pad=2)
        ax.set_xlabel(r"Re-inference Threshold $L_0$")
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Expected Re-Inference Leakage")

    handles, labels = axes[0].get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        bbox_to_anchor=(0.5, 0.99),
        borderpad=0.3
    )

    plt.subplots_adjust(top=0.84, wspace=0.15)

    with PdfPages(f"{method_name}_leakage.pdf") as pdf:
        pdf.savefig(fig)

    plt.close()

# =============================================================================
# Run
# =============================================================================
for method in METHODS:

    df = load_method_df(method)

    if df.empty:
        continue

    plot_mask(df, method)
    plot_leakage(df, method)

print("Generated 4 plots:")
print(" - exp_mask_improvement.pdf")
print(" - exp_leakage.pdf")
print(" - gumbel_mask_improvement.pdf")
print(" - gumbel_leakage.pdf")