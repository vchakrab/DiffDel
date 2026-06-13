import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/release_data"

FS = 11

plt.rcParams.update({
    "text.usetex":           True,
    "font.family":           "serif",
    "font.serif":            ["Computer Modern Roman"],
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
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"

DATASETS_5 = ["airport", "hospital", "adult", "flight", "tax"]
MIN_MASK   = {"airport": 5, "hospital": 9, "adult": 9, "flight": 11, "tax": 3}

# Shared layout — every one of the 6 figures uses exactly these values
FIGSIZE  = (16.5, 3.5)
ADJUST   = dict(wspace=0.18, left=0.07, right=0.97, top=0.82, bottom=0.15)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_curves_data() -> pd.DataFrame:
    records = []
    for method in ["exp", "gum"]:
        for dataset in DATASETS_5:
            path = DATA_DIR / method / dataset / "full_data.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df["epsilon_m"] != 0]
            if df.empty:
                continue
            delmin = MIN_MASK[dataset]
            for (eps, L0), grp in df.groupby(["epsilon_m", "L0"]):
                n            = len(grp)
                mean_mask    = grp["mask_size"].mean()
                mean_leakage = grp["leakage"].mean()
                std_mask     = grp["mask_size"].std()
                std_leakage  = grp["leakage"].std()
                ci_mask      = 1.96 * std_mask    / np.sqrt(n)
                ci_leakage   = 1.96 * std_leakage / np.sqrt(n)
                records.append({
                    "method":         method,
                    "dataset":        dataset,
                    "epsilon_m":      eps,
                    "L0":             L0,
                    "improvement":    100 * abs(delmin - mean_mask) / delmin,
                    "ci_improvement": 100 * ci_mask / delmin,
                    "mean_leakage":   mean_leakage,
                    "ci_leakage":     ci_leakage,
                })
    return pd.DataFrame(records)


def load_ablation_data() -> pd.DataFrame:
    ablation_dir = DATA_DIR / "gum_score_ablation"
    records = []
    for variant in ["random", "zero"]:
        for dataset in DATASETS_5:
            path = ablation_dir / variant / dataset / "full_data.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df["epsilon_m"] != 0]
            if df.empty:
                continue
            delmin = MIN_MASK[dataset]
            for (eps, L0), grp in df.groupby(["epsilon_m", "L0"]):
                n            = len(grp)
                mean_mask    = grp["mask_size"].mean()
                mean_leakage = grp["leakage"].mean()
                std_mask     = grp["mask_size"].std(ddof=1) if n > 1 else 0.0
                std_leakage  = grp["leakage"].std(ddof=1)  if n > 1 else 0.0
                ci_mask      = 1.96 * std_mask    / np.sqrt(n)
                ci_leakage   = 1.96 * std_leakage / np.sqrt(n)
                records.append({
                    "variant":        variant,
                    "dataset":        dataset,
                    "epsilon_m":      eps,
                    "L0":             L0,
                    "improvement":    100 * abs(delmin - mean_mask) / delmin,
                    "ci_improvement": 100 * ci_mask / delmin,
                    "mean_leakage":   mean_leakage,
                    "ci_leakage":     ci_leakage,
                })
    return pd.DataFrame(records)


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_mask(df: pd.DataFrame, filter_col: str, filter_val: str,
              legend: bool) -> plt.Figure:
    fig, axes = plt.subplots(1, 5, figsize=FIGSIZE, sharey=False)
    plt.subplots_adjust(**ADJUST)
    subset_all = df[df[filter_col] == filter_val]
    for col, ds in enumerate(DATASETS_5):
        ax     = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]
        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(curve["L0"], curve["improvement"],
                    marker="o", label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"],
                            curve["improvement"] - curve["ci_improvement"],
                            curve["improvement"] + curve["ci_improvement"],
                            alpha=0.18)
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=FS + 3)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        if col == 0:
            ax.set_ylabel(r"Mask Size Improvement (\%)", fontsize=FS + 3)
        else:
            ax.set_ylabel(None)
    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   frameon=True, bbox_to_anchor=(0.5, 1.05), fontsize=FS + 3)
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=FS + 3)
    return fig


def plot_leakage(df: pd.DataFrame, filter_col: str, filter_val: str,
                 legend: bool) -> plt.Figure:
    fig, axes = plt.subplots(1, 5, figsize=FIGSIZE, sharey=False)
    plt.subplots_adjust(**ADJUST)
    subset_all = df[df[filter_col] == filter_val]
    for col, ds in enumerate(DATASETS_5):
        ax     = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]
        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean  = 100 * curve["mean_leakage"]
            lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean, marker="o",
                    label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)
        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=FS + 3)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])
        if col == 0:
            ax.set_ylabel(r"Achieved Re-inference Leakage (\%)", fontsize=FS + 3)
        else:
            ax.set_ylabel(None)
    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   frameon=True, bbox_to_anchor=(0.5, 1.05), fontsize=FS + 3)
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=FS + 3)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

FIG_DIR = BASE_DIR / "fig_ablation"
FIG_DIR.mkdir(exist_ok=True)

df_curves = load_curves_data()
df_abl    = load_ablation_data()

# 1. Gum — WITH legend
fig = plot_mask(df_curves, "method", "gum", legend=True)
fig.savefig(FIG_DIR / "mask_gum.pdf", bbox_inches="tight"); plt.close(fig)

# 2. Random ablation — no legend
fig = plot_mask(df_abl, "variant", "random", legend=False)
fig.savefig(FIG_DIR / "mask_random.pdf", bbox_inches="tight"); plt.close(fig)

# 3. Zero ablation — no legend
fig = plot_mask(df_abl, "variant", "zero", legend=False)
fig.savefig(FIG_DIR / "mask_zero.pdf", bbox_inches="tight"); plt.close(fig)

# 4. Gum leakage — WITH legend
fig = plot_leakage(df_curves, "method", "gum", legend=True)
fig.savefig(FIG_DIR / "leakage_gum.pdf", bbox_inches="tight"); plt.close(fig)

# 5. Random ablation leakage — no legend
fig = plot_leakage(df_abl, "variant", "random", legend=False)
fig.savefig(FIG_DIR / "leakage_random.pdf", bbox_inches="tight"); plt.close(fig)

# 6. Zero ablation leakage — no legend
fig = plot_leakage(df_abl, "variant", "zero", legend=False)
fig.savefig(FIG_DIR / "leakage_zero.pdf", bbox_inches="tight"); plt.close(fig)

print(f"All 6 figures saved to: {FIG_DIR.resolve()}")