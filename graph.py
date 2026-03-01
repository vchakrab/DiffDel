#!/usr/bin/env python3
"""
graph.py — Unified plotting script.

Assumes data is in a `data/` folder relative to this script, structured as:
    data/
        exp/<dataset>/full_data.csv
        gum/<dataset>/full_data.csv
        min/<dataset>/full_data.csv

Generates:
    all_figures.pdf  (heatmap, pareto, mask curves, leakage curves, radar, runtime)
    budget_split_table_tau_022.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from pathlib import Path
import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# =============================================================================
# STYLE
# =============================================================================
FS = 13

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
# SHARED CONFIG
# =============================================================================
DATASETS_5 = ["airport", "hospital", "adult", "flight", "tax"]
DATASETS_3 = ["airport", "hospital", "flight"]

MIN_MASK = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}

# Heatmap colormap
PASTEL_COLORS = [
    "#FFF8F0", "#F0F7EC", "#D4EDDA", "#A8D8B9",
    "#7BC89C", "#4DAF7A", "#2D8B57",
]
PASTEL_CMAP = LinearSegmentedColormap.from_list("pastel_green", PASTEL_COLORS, N=256)

TAU_CONTOURS = [0.15, 0.32, 0.52, 0.73]
TAU_COLORS = {
    0.15: "#d62728",
    0.32: "#0072B2",
    0.52: "#9467bd",
    0.73: "#17BECF",
}

# Radar / runtime config
DATASET_ORDER = ["Airport", "Hospital", "Adult", "Flight", "Tax"]
METHOD_ORDER  = ["DelMin", "DelExp", "DelMarg"]
METHOD_LABEL  = {"DelMin": "Min", "DelExp": "Exp", "DelMarg": "Gum"}

LINESTYLES = {"DelMin": "-", "DelExp": "--", "DelMarg": ":"}
METHOD_COLORS = {
    "DelMin": "#5B8FF9",
    "DelExp": "#5AD8A6",
    "DelMarg": "#F6BD16",
}

PHASE_COLORS = {
    "Instantiation": "#6F6F6F",
    "Modeling":      "#A0A0A0",
    "Update Masks":  "#3F3F3F",
}
PHASE_HATCH = {
    "Instantiation": "///",
    "Modeling":      "\\\\",
    "Update Masks":  "...",
}

ZONE_COLOR_LIGHT = "#D2B48C"
ZONE_COLOR_DARK  = "#8B4513"

# =============================================================================
# TAU HELPER
# =============================================================================
def tau_fn(eps_m, L0):
    a = np.exp(eps_m) * L0
    return a / ((1 - L0) + a)

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_heatmap_data():
    """Load exp + gum CSVs for heatmap / pareto / budget plots."""
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
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    exp = load_folder(DATA_DIR / "exp", "Exp")
    gum = load_folder(DATA_DIR / "gum", "Gum")

    df = pd.concat([exp, gum], ignore_index=True)
    df["min_mask"] = df["dataset"].map(MIN_MASK)
    return df


def load_curves_data(datasets):
    """
    Load exp + gum CSVs into a DataFrame with per-(eps, L0) aggregation.
    Returns a DataFrame with columns:
        method, dataset, epsilon_m, L0,
        improvement, ci_improvement, mean_leakage, ci_leakage
    """
    records = []

    for method in ["exp", "gum"]:
        for dataset in datasets:
            path = DATA_DIR / method / dataset / "full_data.csv"
            if not path.exists():
                print(f"  [skip] {path}")
                continue

            df = pd.read_csv(path)
            df = df[df["epsilon_m"] != 0]
            if df.empty:
                continue

            delmin = MIN_MASK[dataset]

            for (eps, L0), group in df.groupby(["epsilon_m", "L0"]):
                n = len(group)
                mean_mask    = group["mask_size"].mean()
                std_mask     = group["mask_size"].std()
                mean_leakage = group["leakage"].mean()
                std_leakage  = group["leakage"].std()

                ci_mask     = 1.96 * std_mask     / np.sqrt(n)
                ci_leakage  = 1.96 * std_leakage  / np.sqrt(n)

                improvement    = 100 * abs(delmin - mean_mask) / delmin
                ci_improvement = 100 * ci_mask / delmin

                records.append({
                    "method":         method,
                    "dataset":        dataset,
                    "epsilon_m":      eps,
                    "L0":             L0,
                    "improvement":    improvement,
                    "ci_improvement": ci_improvement,
                    "mean_leakage":   mean_leakage,
                    "ci_leakage":     ci_leakage,
                })

    return pd.DataFrame(records)


def load_main_data():
    """Load min + exp + gum data for radar and runtime plots."""
    dfs = []

    # DelMin (no epsilon / L0 filter)
    for dataset in [d.lower() for d in DATASET_ORDER]:
        path = DATA_DIR / "min" / dataset / "full_data.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["dataset"] = dataset.capitalize()
            df["method"]  = "DelMin"
            dfs.append(df)

    # DelExp and DelMarg — filtered to a single (eps, L0) point
    for folder, method in [("exp", "DelExp"), ("gum", "DelMarg")]:
        for dataset in [d.lower() for d in DATASET_ORDER]:
            path = DATA_DIR / folder / dataset / "full_data.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[
                (df["epsilon_m"].round(5) == 0.1) &
                (df["L0"].round(5)        == 0.2)
            ]
            if not df.empty:
                df["dataset"] = dataset.capitalize()
                df["method"]  = method
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    df["init_time_ms"]   = df["init_time"]   * 1000.0
    df["model_time_ms"]  = df["model_time"]  * 1000.0
    df["update_time_ms"] = df["del_time"]    * 1000.0
    df["time_ms"]        = (
        df["init_time_ms"] + df["model_time_ms"] + df["update_time_ms"]
    )
    df["memory_kb"] = df["memory_overhead_bytes"] / 1024.0

    denom = df["num_instantiated_cells"].clip(lower=1.0)
    df["deletion_ratio"] = (df["mask_size"] / denom).clip(0.0, 1.0)

    return df

# =============================================================================
# PLOT 1 — HEATMAP (6 panels, Exp + Gum × 3 datasets)
# =============================================================================
def plot_heatmap(df):
    """Mask improvement heatmap for airport/hospital/flight × Exp/Gum."""

    datasets = DATASETS_3
    eps_vals = sorted(df["epsilon_m"].unique())
    L0_vals  = sorted(df["L0"].unique())
    L0_plot  = L0_vals[:-1]

    df = df.copy()
    df["improvement"] = 100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]

    agg = (
        df.groupby(["dataset", "mechanism", "epsilon_m", "L0"])["improvement"]
        .mean()
        .reset_index()
    )

    fig = plt.figure(figsize=(19.3, 3.5))
    gs  = GridSpec(1, 6, wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]

    norm    = Normalize(vmin=0, vmax=75)
    ordered = (
        [(ds, "Exp") for ds in datasets] +
        [(ds, "Gum") for ds in datasets]
    )

    im_ref = None
    for i, (ds, mech) in enumerate(ordered):
        ax = axes[i]
        ax.set_title(
            rf"$\mathbf{{{ds.capitalize()}}}$ ({mech})",
            pad=8, fontsize=FS,
        )

        sub   = agg[(agg.dataset == ds) & (agg.mechanism == mech)]
        pivot = sub.pivot_table(
            index="L0", columns="epsilon_m", values="improvement"
        ).reindex(index=L0_plot, columns=eps_vals)

        im_ref = ax.imshow(
            pivot.values, cmap=PASTEL_CMAP, norm=norm,
            origin="lower", aspect="auto",
        )

        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels(eps_vals, rotation=45)
        ax.set_yticks(range(len(L0_plot)))
        ax.set_yticklabels(L0_plot)

        if i == 0:
            ax.set_ylabel(
                r"Re-inference Leakage Threshold $L_0$", fontsize=FS + 3
            )

        # τ contours
        for tau in TAU_CONTOURS:
            x_coords, y_coords = [], []
            for ei, em in enumerate(eps_vals):
                l0 = tau / (np.exp(em) * (1 - tau) + tau)
                if L0_plot[0] <= l0 <= L0_plot[-1]:
                    yp = np.interp(l0, L0_plot, np.arange(len(L0_plot)))
                    x_coords.append(ei)
                    y_coords.append(yp)
            if len(x_coords) >= 2:
                ax.plot(
                    x_coords, y_coords,
                    linestyle="--", linewidth=2.5,
                    color=TAU_COLORS[tau], zorder=5, clip_on=False,
                )

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=PASTEL_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.26, 0.915, 0.28, 0.028])
    cb  = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=FS - 1)
    cb.set_label("")
    cax.text(
        -0.025, 0.5, "Mask Improvement (%)",
        transform=cax.transAxes, va="center", ha="right", fontsize=FS + 3,
    )

    # τ legend
    tau_handles = [
        Line2D([0], [0], color=TAU_COLORS[t], linestyle="--",
               linewidth=2.0, label=rf"$\tau={t}$")
        for t in TAU_CONTOURS
    ]
    fig.legend(
        handles=tau_handles,
        loc="upper center",
        bbox_to_anchor=(0.7, 0.985),
        ncol=len(TAU_CONTOURS),
        frameon=True,
        fontsize=FS + 3,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.3,
    )

    fig.supxlabel(
        r"Masking-Privacy Budget $\epsilon_m$", y=-0.04, fontsize=FS + 3
    )
    plt.subplots_adjust(top=0.75, bottom=0.18)
    return fig


# =============================================================================
# PLOT 2 — MASK IMPROVEMENT CURVES (6 panels)
# =============================================================================
def plot_mask_curves():
    """Mask size improvement vs L0 for exp + gum × 3 datasets."""

    df = load_curves_data(DATASETS_3)

    fig, axes = plt.subplots(1, 6, figsize=(19.8, 3), sharey=False)

    ordered = (
        [(ds, "exp")    for ds in DATASETS_3] +
        [(ds, "gum") for ds in DATASETS_3]
    )

    for i, (dataset, mech) in enumerate(ordered):
        ax = axes[i]
        subset = df[(df["dataset"] == dataset) & (df["method"] == mech)]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(
                curve["L0"], curve["improvement"],
                marker="o", label=rf"$\varepsilon_m = {eps}$",
            )
            ax.fill_between(
                curve["L0"],
                curve["improvement"] - curve["ci_improvement"],
                curve["improvement"] + curve["ci_improvement"],
                alpha=0.18,
            )

        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.set_title(
            rf"$\mathbf{{{dataset.capitalize()}}}$ ({mech_label})",
            pad=2, fontsize=FS + 2,
        )
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)

        if i == 0:
            ax.set_ylabel("Mask Size Improvement (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(handles),
        frameon=True, bbox_to_anchor=(0.5, 1.05),
    )
    fig.supxlabel(r"Re-inference Leakage Threshold $L_0$", y=-0.06)
    plt.subplots_adjust(top=0.82, wspace=0.25)

    return fig


# =============================================================================
# PLOT 3 — LEAKAGE CURVES (6 panels)
# =============================================================================
def plot_leakage_curves():
    """Achieved re-inference leakage vs L0 for exp + gum × 3 datasets."""

    df = load_curves_data(DATASETS_3)

    fig, axes = plt.subplots(1, 6, figsize=(19.8, 3), sharey=False)

    ordered = (
        [(ds, "exp")    for ds in DATASETS_3] +
        [(ds, "gum") for ds in DATASETS_3]
    )

    for i, (dataset, mech) in enumerate(ordered):
        ax = axes[i]
        subset = df[(df["dataset"] == dataset) & (df["method"] == mech)]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve      = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean_leak  = 100 * curve["mean_leakage"]
            lower      = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper      = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean_leak, marker="o")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)

        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)

        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.set_title(
            rf"$\mathbf{{{dataset.capitalize()}}}$ ({mech_label})",
            pad=2, fontsize=FS + 2,
        )
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)

        if i == 0:
            ax.set_ylabel("Achieved Re-inference Leakage (%)")

    fig.supxlabel(r"Re-inference Leakage Threshold $L_0$", y=0.02)
    plt.subplots_adjust(top=0.85, bottom=0.18, wspace=0.25)

    return fig


# =============================================================================
# PLOT 4 — PARETO FRONTIER (5 panels)
# =============================================================================
def plot_pareto(df):
    """Pareto frontier: mask size (%) vs expected leakage."""

    EM_VALS = [0.1, 1.0]
    EPS_COLOR    = {1.0: "#d62728", 0.1: "#1f77b4"}
    STAR_COLOR   = "black"
    DIAMOND_COLOR = "black"

    TARGET_ONLY_LEAK = {
        "airport":  0.49830908053361767,
        "hospital": 0.6623992294064142,
        "adult":    0.49122554744404323,
        "flight":   0.9826408586840785,
        "tax":      0.5548418449270073,
    }

    agg = (
        df[df["epsilon_m"].isin(EM_VALS)]
        .groupby(["dataset", "mechanism", "epsilon_m", "L0"])
        .agg(mean_mask=("mask_size", "mean"), mean_leak=("leakage", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 2.75))

    for col, ds in enumerate(DATASETS_5):
        ax       = axes[col]
        baseline = MIN_MASK[ds]
        ax.tick_params(axis="y", left=True, labelleft=True)
        ax.set_title(
            rf"$\mathbf{{{ds.capitalize()}}}$",
            pad=2, fontweight="bold", fontsize=FS + 2,
        )

        for mech in ["Exp", "Gum"]:
            for em in EM_VALS:
                sub = agg[
                    (agg.dataset   == ds) &
                    (agg.mechanism == mech) &
                    (agg.epsilon_m == em)
                ].sort_values("L0")
                if sub.empty:
                    continue

                mask_pct      = 100 * sub["mean_mask"] / baseline
                marker_style  = "o" if em == 1.0 else "x"
                marker_face   = "none" if em == 1.0 else None
                line_style    = "-" if mech == "Exp" else ":"

                ax.plot(
                    sub["mean_leak"], mask_pct,
                    marker=marker_style,
                    linestyle=line_style,
                    color=EPS_COLOR[em],
                    markerfacecolor=marker_face,
                    markeredgewidth=1.8,
                    markersize=6,
                    linewidth=1.8,
                    label=f"{mech}, ε={em}",
                )

        # Baseline star
        ax.scatter(
            0, 100,
            marker="*", s=260,
            facecolor=STAR_COLOR, edgecolor="black",
            linewidth=1.2, zorder=7,
            label=r"$M_{\mathrm{MIN}}\,(M_{\mathrm{det}})$" if col == 0 else None,
        )

        # Empty-mask diamond
        ax.scatter(
            TARGET_ONLY_LEAK[ds], 0,
            marker="D", s=120,
            facecolor=DIAMOND_COLOR, edgecolor="black",
            zorder=6,
            label=r"$M = \emptyset$" if col == 0 else None,
        )

        max_leak    = max(agg[agg.dataset == ds]["mean_leak"].max(),
                         TARGET_ONLY_LEAK[ds])
        rounded_max = np.ceil(max_leak / 0.2) * 0.2
        xticks      = np.arange(0, rounded_max + 0.001, 0.2)
        ax.set_xlim(-0.03, rounded_max + 0.04)
        ax.set_xticks(xticks)
        ax.set_ylim(-8, 124)

        if col == 0:
            ax.set_ylabel("Mask Size (% of Baseline)", fontsize=FS + 3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", frameon=True,
        bbox_to_anchor=(0.5, 1.12), ncol=6,
    )
    fig.supxlabel(
        r"Expected Leakage ($\mathbb{E}[\mathcal{L}]$)",
        y=-0.06, fontsize=FS + 2,
    )

    return fig


# =============================================================================
# PLOT 5 — RADAR (5 datasets)
# =============================================================================
def _plot_radar_row(fig, subspec, df):

    metrics   = [r"$|M|/|I|$", "T (ms)", "Mem", r"$\mathcal{L}$", "Paths"]
    metric_map = {
        r"$|M|/|I|$":        "deletion_ratio",
        r"$\mathcal{L}$":    "leakage",
        "Mem":               "memory_kb",
        "T (ms)":            "time_ms",
        "Paths":             "total_paths",
    }

    n      = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    axes = [fig.add_subplot(subspec[0, i], polar=True) for i in range(5)]

    for i, dataset in enumerate(DATASET_ORDER):
        ax  = axes[i]
        ddf = df[df["dataset"] == dataset]
        if ddf.empty:
            ax.set_axis_off()
            continue

        agg = (
            ddf.groupby("method")
               .mean(numeric_only=True)
               .reindex(METHOD_ORDER)
               .fillna(0.0)
        )

        mat  = np.column_stack([agg[metric_map[m]].values for m in metrics])
        mins = np.nanmin(mat, axis=0)
        maxs = np.nanmax(mat, axis=0)
        denom = np.where((maxs - mins) < 1e-9, 1.0, (maxs - mins))
        norm  = np.nan_to_num((mat - mins) / denom, nan=0.0)

        for mi, method in enumerate(METHOD_ORDER):
            vals = norm[mi].tolist() + [norm[mi][0]]
            ax.plot(
                angles, vals,
                color=METHOD_COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.6, marker="o", markersize=3,
            )
            ax.fill(angles, vals, color=METHOD_COLORS[method], alpha=0.08)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.tick_params(axis="x", pad=7)
        ax.set_ylim(0, 1.05)
        ax.set_yticklabels([])
        ax.set_title(dataset, pad=16, fontweight="bold")

    return axes


def build_radar(df):
    fig = plt.figure(figsize=(16.5, 3.0))
    gs  = fig.add_gridspec(1, 1)
    _plot_radar_row(fig, gs[0, 0].subgridspec(1, 5, wspace=0.15), df)

    fig.legend(
        handles=[
            Line2D([0], [0], color=METHOD_COLORS[m],
                   linestyle=LINESTYLES[m], lw=1.5,
                   label=METHOD_LABEL[m])
            for m in METHOD_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=True,
    )
    fig.subplots_adjust(top=0.68)
    return fig


# =============================================================================
# PLOT 6 — RUNTIME (5 datasets)
# =============================================================================
def _plot_runtime_row(fig, subspec, df):

    axes = [fig.add_subplot(subspec[0, i]) for i in range(5)]

    for i, dataset in enumerate(DATASET_ORDER):
        ax  = axes[i]
        ddf = df[df["dataset"] == dataset]
        if ddf.empty:
            ax.set_axis_off()
            continue

        s = (
            ddf.groupby("method")
               .mean(numeric_only=True)
               .reindex(METHOD_ORDER)
               .fillna(0.0)
        )

        x      = np.arange(len(METHOD_ORDER))
        w      = 0.25
        bottom = np.zeros(len(METHOD_ORDER))

        for key, label in [
            ("init_time_ms",   "Instantiation"),
            ("model_time_ms",  "Modeling"),
            ("update_time_ms", "Update Masks"),
        ]:
            vals = s[key].values
            ax.bar(
                x - w, vals, width=w, bottom=bottom,
                color=PHASE_COLORS[label], hatch=PHASE_HATCH[label],
                edgecolor="black", linewidth=0.3,
            )
            bottom += vals

        ax.set_xticks(x - w)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])

        ax2 = ax.twinx()

        deleted_pct      = s["deletion_ratio"] * 100.0
        instantiated_pct = 100.0 - deleted_pct

        ax2.bar(x + w, instantiated_pct, width=w,
                color=ZONE_COLOR_LIGHT, edgecolor="black", linewidth=0.3)
        ax2.bar(x + w, deleted_pct, width=w, bottom=instantiated_pct,
                color=ZONE_COLOR_DARK, edgecolor="black", linewidth=0.3)
        ax2.set_ylim(0, 100)

        ax.tick_params(axis="y", labelleft=True)
        ax2.tick_params(axis="y", labelright=True)

        if i == 0:
            ax.set_ylabel("Time (ms)")
        else:
            ax.set_ylabel(None)

        if i == len(DATASET_ORDER) - 1:
            ax2.set_ylabel("Deleted Cells (%)")
        else:
            ax2.set_ylabel(None)
            ax2.spines["right"].set_visible(False)

    return axes


def _runtime_legend_handles():
    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p], hatch=PHASE_HATCH[p],
              edgecolor="black", linewidth=0.3, label=p)
        for p in PHASE_COLORS
    ]
    zone_handles = [
        Patch(facecolor=ZONE_COLOR_LIGHT, edgecolor="black",
              linewidth=0.3, label="Instantiated Cells"),
        Patch(facecolor=ZONE_COLOR_DARK, edgecolor="black",
              linewidth=0.3, label="Mask Size"),
    ]
    return phase_handles + zone_handles


def build_runtime(df):
    fig = plt.figure(figsize=(16.5, 3.5))
    gs  = fig.add_gridspec(1, 1)
    axes = _plot_runtime_row(fig, gs[0, 0].subgridspec(1, 5, wspace=0.35), df)

    for ax, dataset in zip(axes, DATASET_ORDER):
        ax.set_title(dataset, pad=10, fontweight="bold")

    fig.legend(
        handles=_runtime_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        fontsize=FS,
        frameon=True,
    )
    fig.subplots_adjust(top=0.72)
    return fig


# =============================================================================
# TABLE — BUDGET SPLIT (CSV)
# =============================================================================
def generate_budget_table(df):
    """Numerical budget-split table at tau ≈ 0.22."""

    df = df.copy()
    df["improvement"] = 100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]

    agg = (
        df.groupby(["dataset", "mechanism", "epsilon_m", "L0"])["improvement"]
        .mean()
        .reset_index()
    )
    agg["tau"] = agg.apply(lambda r: tau_fn(r["epsilon_m"], r["L0"]), axis=1)

    TARGET_TAU, tol = 0.22, 0.04
    rows = []

    for mech in ["Exp", "Gum"]:
        for ds in DATASETS_5:
            sub   = agg[(agg.dataset == ds) & (agg.mechanism == mech)]
            close = sub[np.abs(sub["tau"] - TARGET_TAU) <= tol]
            if len(close) < 2:
                continue

            best_row  = close.loc[close["improvement"].idxmax()]
            worst_row = close.loc[close["improvement"].idxmin()]

            rows.append({
                "Dataset":                  ds,
                "Mechanism":                mech,
                "Total Budget (tau)":       round(TARGET_TAU, 3),
                "Best ε_m":                 best_row["epsilon_m"],
                "Best L0":                  best_row["L0"],
                "Best Mask Improvement (%)": round(best_row["improvement"], 2),
                "Worst ε_m":                worst_row["epsilon_m"],
                "Worst L0":                 worst_row["L0"],
                "Worst Mask Improvement (%)": round(worst_row["improvement"], 2),
            })

    pd.DataFrame(rows).to_csv("budget_split_table_tau_022.csv", index=False)
    print("  -> budget_split_table_tau_022.csv")


# =============================================================================
# MAIN — all figures into one PDF
# =============================================================================
if __name__ == "__main__":
    print("Loading data …")
    df_heat = load_heatmap_data()
    df_main = load_main_data()

    OUTPUT_PDF = "all_figures.pdf"
    print(f"\nWriting all figures to {OUTPUT_PDF} …")

    with PdfPages(OUTPUT_PDF) as pdf:

        if not df_heat.empty:
            print("  heatmap …")
            fig = plot_heatmap(df_heat)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print("  pareto …")
            fig = plot_pareto(df_heat)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            generate_budget_table(df_heat)
        else:
            print("  [skip] heatmap / pareto / budget — no heatmap data found")

        print("  mask improvement curves …")
        fig = plot_mask_curves()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        print("  leakage curves …")
        fig = plot_leakage_curves()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not df_main.empty:
            print("  radar …")
            fig = build_radar(df_main)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print("  runtime …")
            fig = build_runtime(df_main)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        else:
            print("  [skip] radar / runtime — no main data found")

    print(f"\nDone. All figures saved to {OUTPUT_PDF}")
    print("  budget_split_table_tau_022.csv also written.")