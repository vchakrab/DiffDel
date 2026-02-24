#!/usr/bin/env python3
"""
PDF 3: Radar + Runtime (FINAL CLEAN VERSION)

Clockwise from 12 o’clock:
    M/I → T → Mem → L → Paths

Output:
    fig_radar_fixed_order.pdf
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
})

# =============================================================================
# CONFIG
# =============================================================================
DATASET_ORDER = ["Airport", "Hospital", "Adult", "Flight", "Tax"]

METHOD_ORDER = ["DelMin", "DelExp", "DelMarg"]

METHOD_LABEL = {
    "DelMin": "MIN",
    "DelExp": "EXP",
    "DelMarg": "MARG",
}

LINESTYLES = {
    "DelMin": "-",
    "DelExp": "--",
    "DelMarg": ":",
}

METHOD_COLORS = {
    "DelMin": "#5B8FF9",
    "DelExp": "#5AD8A6",
    "DelMarg": "#F6BD16",
}

PHASE_COLORS = {
    "Instantiation": "#6F6F6F",
    "Modeling": "#A0A0A0",
    "Update Masks": "#3F3F3F",
}

PHASE_HATCH = {
    "Instantiation": "///",
    "Modeling": "\\\\",
    "Update Masks": "...",
}

ZONE_COLOR_LIGHT = "#D2B48C"
ZONE_COLOR_DARK  = "#8B4513"

# =============================================================================
# DATA LOADER
# =============================================================================
def load_pdf3_data(data_dir: Path, delmin_csv: Path) -> pd.DataFrame:

    dfs = []

    delmin = pd.read_csv(delmin_csv)
    delmin.columns = [str(c).strip() for c in delmin.columns]

    if "Dataset" in delmin.columns:
        delmin = delmin.rename(columns={"Dataset": "dataset"})

    delmin["dataset"] = delmin["dataset"].astype(str).str.strip().str.capitalize()
    delmin["method"] = "DelMin"
    dfs.append(delmin)

    for f in sorted(data_dir.glob("*.csv")):
        name = f.name.lower()

        if name.startswith("exp_"):
            method = "DelExp"
        elif name.startswith("marginal_"):
            method = "DelMarg"
        else:
            continue

        dataset = None
        for ds in ["airport", "hospital", "adult", "flight", "tax"]:
            if f"_{ds}_" in name:
                dataset = ds.capitalize()
                break

        if dataset is None:
            continue

        df = pd.read_csv(f)
        df.columns = [str(c).strip() for c in df.columns]
        df["dataset"] = dataset
        df["method"] = method
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    numeric_cols = [
        "init_time", "model_time", "del_time",
        "mask_size", "model_size",
        "num_instantiated_cells", "leakage", "total_paths"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0

    df["init_time_ms"] = df["init_time"] * 1000.0
    df["model_time_ms"] = df["model_time"] * 1000.0
    df["update_time_ms"] = df["del_time"] * 1000.0
    df["time_ms"] = df["init_time_ms"] + df["model_time_ms"] + df["update_time_ms"]
    df["memory_kb"] = df["model_size"] / 1024.0

    denom = df["num_instantiated_cells"].clip(lower=1.0)
    df["deletion_ratio"] = (df["mask_size"] / denom).clip(0.0, 1.0)

    return df

# =============================================================================
# RADAR
# =============================================================================
def plot_radar_row(fig, subspec, df):

    metrics = [
        r"$|M|/|I|$",
        "T (ms)",
        "Mem",
        r"$\mathcal{L}$",
        "Paths",
    ]

    axes = [fig.add_subplot(subspec[0, i], polar=True) for i in range(5)]
    n = len(metrics)

    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    metric_map = {
        r"$|M|/|I|$": "deletion_ratio",
        r"$\mathcal{L}$": "leakage",
        "Mem": "memory_kb",
        "T (ms)": "time_ms",
        "Paths": "total_paths",
    }

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        ddf = df[df["dataset"] == dataset]

        if ddf.empty:
            ax.set_axis_off()
            continue

        agg = ddf.groupby("method").mean(numeric_only=True).reindex(METHOD_ORDER)
        mat = np.column_stack([agg[metric_map[m]].values for m in metrics])

        mins = np.nanmin(mat, axis=0)
        maxs = np.nanmax(mat, axis=0)
        denom = np.where((maxs - mins) < 1e-9, 1.0, (maxs - mins))
        norm = np.nan_to_num((mat - mins) / denom, nan=0.0)

        for mi, method in enumerate(METHOD_ORDER):
            vals = norm[mi].tolist() + [norm[mi][0]]
            ax.plot(
                angles, vals,
                color=METHOD_COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.6
            )
            ax.fill(angles, vals, color=METHOD_COLORS[method], alpha=0.08)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.tick_params(axis="x", pad=6)

        ax.set_ylim(0, 1.05)
        ax.set_yticklabels([])
        ax.set_title(dataset, pad=18, fontweight="bold")

    return axes

# =============================================================================
# RUNTIME
# =============================================================================
def plot_runtime_row(fig, subspec, df):

    axes = [fig.add_subplot(subspec[0, i]) for i in range(5)]

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        ddf = df[df["dataset"] == dataset]

        if ddf.empty:
            ax.set_axis_off()
            continue

        s = ddf.groupby("method").mean(numeric_only=True).reindex(METHOD_ORDER)

        x = np.arange(len(METHOD_ORDER))
        w = 0.25
        bottom = np.zeros(len(METHOD_ORDER))

        phase_map = [
            ("init_time_ms", "Instantiation"),
            ("model_time_ms", "Modeling"),
            ("update_time_ms", "Update Masks"),
        ]

        for key, label in phase_map:
            vals = s[key].values
            ax.bar(
                x - w, vals,
                width=w, bottom=bottom,
                color=PHASE_COLORS[label],
                hatch=PHASE_HATCH[label],
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += vals

        ax.set_xticks(x - w)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])

        if i == 0:
            ax.set_ylabel("Time (ms)")

        ax2 = ax.twinx()
        ax2.bar(x + w, s["num_instantiated_cells"], width=w,
                color=ZONE_COLOR_LIGHT, edgecolor="black", linewidth=0.3)
        ax2.bar(x + w, s["mask_size"], width=w,
                color=ZONE_COLOR_DARK, edgecolor="black", linewidth=0.3)

    return axes

# =============================================================================
# LEGEND HANDLES
# =============================================================================
def legend_handles():
    method_handles = [
        Line2D([0],[0], color=METHOD_COLORS[m],
               linestyle=LINESTYLES[m], lw=1.5,
               label=METHOD_LABEL[m])
        for m in METHOD_ORDER
    ]

    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p],
              hatch=PHASE_HATCH[p],
              edgecolor="black",
              linewidth=0.3,
              label=p)
        for p in PHASE_COLORS
    ]

    zone_handles = [
        Patch(facecolor=ZONE_COLOR_LIGHT, edgecolor="black",
              linewidth=0.3, label="Instantiated Cells"),
        Patch(facecolor=ZONE_COLOR_DARK, edgecolor="black",
              linewidth=0.3, label="Mask Size"),
    ]

    return method_handles + phase_handles + zone_handles

# =============================================================================
# BUILD FIGURE
# =============================================================================
# =============================================================================
# ADD VERTICAL SEPARATORS
# =============================================================================
def add_vertical_separators(fig, axes_row, linewidth=1.2, color="black"):

    # Use positions of the first row of axes (radar row)
    for i in range(len(axes_row) - 1):
        right_edge = axes_row[i].get_position().x1
        next_left  = axes_row[i+1].get_position().x0

        x = (right_edge + next_left) / 2.0

        fig.add_artist(
            plt.Line2D(
                [x, x],
                [0.12, 0.88],   # vertical span (adjust if needed)
                transform=fig.transFigure,
                linewidth=linewidth,
                color=color,
            )
        )
def build_single_figure(df: pd.DataFrame, out_dir: Path):

    out_file = out_dir / "fig_radar_fixed_order.pdf"

    fig = plt.figure(figsize=(16.5, 7.4))

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1.3, 1.3],
        hspace=0.12
    )

    radar_spec = gs[0, 0].subgridspec(1, 5, wspace=0.45)
    runtime_spec = gs[1, 0].subgridspec(1, 5, wspace=0.4)

    radar_axes = plot_radar_row(fig, radar_spec, df)
    runtime_axes = plot_runtime_row(fig, runtime_spec, df)
    add_vertical_separators(fig, radar_axes, linewidth = 1.0)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=9,
        frameon=False
    )

    fig.subplots_adjust(top=0.88)

    fig.savefig(out_file)
    plt.close(fig)

    print("Wrote", out_file)

# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./main_plot_data")
    ap.add_argument("--delmin_csv", default="jan26_min.csv")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    df = load_pdf3_data(Path(args.data_dir), Path(args.delmin_csv))
    build_single_figure(df, Path(args.out_dir))


if __name__ == "__main__":
    main()