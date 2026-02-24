# =============================================================================
# DiffDel — VLDB Camera-Ready Mask Improvement (XP STRUCTURE)
# Handles:
#   - Top-level 500-row files (group by dataset column)
#   - Nested _exp folders
#   - _2ph.csv files
#   - All ε_m and L0 values
# =============================================================================

import os
import re
import zipfile
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
# DelMin mask sizes
# =============================================================================
delmin_masks = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3
}

# =============================================================================
# Extract ZIP
# =============================================================================
zip_path = "data 2.zip"
extract_path = "tmp_extract"

if os.path.exists(extract_path):
    import shutil
    shutil.rmtree(extract_path)

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_path)

data_path = os.path.join(extract_path, "data")

records = []
datasets = ["airport", "hospital", "adult", "flight", "tax"]

# =============================================================================
# Recursively scan ALL CSV files
# =============================================================================
for root, dirs, files in os.walk(data_path):

    if "__MACOSX" in root:
        continue

    for file in files:

        if not file.endswith(".csv"):
            continue

        # Extract epsilon_m and L0 from filename
        match = re.search(r"_em_(.*)_L0_(.*?)(?:_2ph)?\.csv", file)
        if not match:
            continue

        epsilon_m = float(match.group(1))
        L0 = float(match.group(2))

        full_path = os.path.join(root, file)
        df = pd.read_csv(full_path)

        # Must have dataset column
        if "dataset" not in df.columns:
            continue

        # Group by dataset INSIDE file
        for dataset, group in df.groupby("dataset"):

            dataset = dataset.lower()

            if dataset not in delmin_masks:
                continue

            delmin = delmin_masks[dataset]

            mean_mask = group["mask_size"].mean()
            std_mask = group["mask_size"].std()
            n = len(group)

            improvement = 100 * abs(delmin - mean_mask) / delmin
            ci = 1.96 * std_mask / np.sqrt(n)
            ci_improvement = 100 * ci / delmin

            records.append({
                "dataset": dataset,
                "epsilon_m": epsilon_m,
                "L0": L0,
                "improvement": improvement,
                "ci": ci_improvement
            })

plot_df = pd.DataFrame(records)
plot_df = plot_df[plot_df["epsilon_m"] != 0]

print("Detected L0 values:", sorted(plot_df["L0"].unique()))

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)

for ax, dataset in zip(axes, datasets):

    ax.tick_params(labelleft=True)

    subset = plot_df[plot_df["dataset"] == dataset]

    for eps_m in sorted(subset["epsilon_m"].unique()):

        curve = subset[subset["epsilon_m"] == eps_m].sort_values("L0")

        ax.plot(
            curve["L0"],
            curve["improvement"],
            marker='o',
            label=rf"$\varepsilon_m = {eps_m}$"
        )

        ax.fill_between(
            curve["L0"],
            curve["improvement"] - curve["ci"],
            curve["improvement"] + curve["ci"],
            alpha=0.18
        )

    ax.set_title(dataset.capitalize(), pad=2)
    ax.set_xlabel(r"Re-inference Threshold $L_0$")

    # Include small L0 values
    ax.set_xlim(0.0, 0.9)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

    ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

axes[0].set_ylabel("Mask Size Improvement (%)")

# =============================================================================
# Global Legend
# =============================================================================
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

legend.get_frame().set_linewidth(0.8)

plt.subplots_adjust(top=0.84, wspace=0.15)

with PdfPages("Exp_MaskImprovement_data2_structure.pdf") as pdf:
    pdf.savefig(fig)

plt.close()

print("Generated: MaskImprovement_data2_structure.pdf")