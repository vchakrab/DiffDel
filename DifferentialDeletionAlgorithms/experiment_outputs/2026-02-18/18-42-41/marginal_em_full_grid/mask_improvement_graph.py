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
zip_path = "marginal_em_final_data.zip"
extract_path = "tmp_extract"

if os.path.exists(extract_path):
    import shutil
    shutil.rmtree(extract_path)

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_path)

base_path = os.path.join(extract_path, "New Folder With Items")

records = []

# =============================================================================
# Iterate over ALL dataset folders (including marginal)
# =============================================================================
for folder in os.listdir(base_path):

    if folder.startswith("__"):
        continue

    folder_path = os.path.join(base_path, folder)

    if not os.path.isdir(folder_path):
        continue

    # Extract dataset name
    dataset = folder.replace("_marginal", "")

    if dataset not in delmin_masks:
        continue

    for file in os.listdir(folder_path):

        if not file.endswith(".csv"):
            continue

        # Flexible pattern: optional _marginal suffix
        match = re.match(
            rf"{dataset}_em_(.*)_L0_(.*?)(?:_marginal)?\.csv",
            file
        )

        if not match:
            continue

        epsilon_m = float(match.group(1))
        L0 = float(match.group(2))

        df = pd.read_csv(os.path.join(folder_path, file))

        delmin = delmin_masks[dataset]

        mean_mask = df["mask_size"].mean()
        std_mask = df["mask_size"].std()
        n = len(df)

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

# =============================================================================
# Plot
# =============================================================================
datasets = ["airport", "hospital", "adult", "flight", "tax"]

fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)

for ax, dataset in zip(axes, datasets):

    ax.tick_params(labelleft=True)

    subset = plot_df[plot_df["dataset"] == dataset]

    L0_values = sorted(subset["L0"].unique())

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

legend.get_frame().set_linewidth(0.8)

plt.subplots_adjust(top=0.84, wspace=0.15)

with PdfPages("fig_marginal_mask_improvement.pdf") as pdf:
    pdf.savefig(fig)

plt.close()