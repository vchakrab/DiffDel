#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# STYLE (UNCHANGED)
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
# CONFIG (UNCHANGED)
# =============================================================================
BASE_PATH = Path(__file__).parent

DATASET_ORDER = ["Airport", "Hospital", "Adult", "Flight", "Tax"]

METHOD_ORDER = ["DelMin", "DelExp", "DelMarg"]

METHOD_LABEL = {
    "DelMin": "MIN",
    "DelExp": "EXP",
    "DelMarg": "GUM",
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
# LOAD DATA (UNCHANGED)
# =============================================================================
def load_data():

    dfs = []

    for dataset in ["airport","hospital","adult","flight","tax"]:
        path = BASE_PATH / "min" / dataset / "full_data.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["dataset"] = dataset.capitalize()
            df["method"] = "DelMin"
            dfs.append(df)

    for folder, method in [("exp","DelExp"),("gumbel","DelMarg")]:
        for dataset in ["airport","hospital","adult","flight","tax"]:
            path = BASE_PATH / folder / dataset / "full_data.csv"
            if path.exists():
                df = pd.read_csv(path)
                df = df[
                    (df["epsilon_m"].round(5) == 0.1) &
                    (df["L0"].round(5) == 0.2)
                ]
                if not df.empty:
                    df["dataset"] = dataset.capitalize()
                    df["method"] = method
                    dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["init_time_ms"] = df["init_time"] * 1000.0
    df["model_time_ms"] = df["model_time"] * 1000.0
    df["update_time_ms"] = df["del_time"] * 1000.0
    df["time_ms"] = df["init_time_ms"] + df["model_time_ms"] + df["update_time_ms"]
    df["memory_kb"] = df["memory_overhead_bytes"] / 1024.0

    denom = df["num_instantiated_cells"].clip(lower=1.0)
    df["deletion_ratio"] = (df["mask_size"] / denom).clip(0.0, 1.0)

    return df

# =============================================================================
# RADAR (WITH MARKERS)
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

        agg = (
            ddf.groupby("method")
               .mean(numeric_only=True)
               .reindex(METHOD_ORDER)
               .fillna(0.0)
        )

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
                linewidth=1.6,
                marker="o",
                markersize=3
            )

            ax.fill(angles, vals, color=METHOD_COLORS[method], alpha=0.08)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.tick_params(axis = "x", pad = 7)
        ax.set_ylim(0, 1.05)
        ax.set_yticklabels([])
        ax.set_title(dataset, pad=16, fontweight="bold")

    return axes

# =============================================================================
# RUNTIME (REDUCED SPACING + MARKERS)
# =============================================================================
# =============================================================================
# RUNTIME (NUMERIC Y AXES ON ALL)
# =============================================================================
# =============================================================================
# RUNTIME (CLEAN — NO MARKERS)
# =============================================================================
def plot_runtime_row(fig, subspec, df):

    axes = [fig.add_subplot(subspec[0, i]) for i in range(5)]

    for i, dataset in enumerate(DATASET_ORDER):

        ax = axes[i]
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

        x = np.arange(len(METHOD_ORDER))
        w = 0.25
        bottom = np.zeros(len(METHOD_ORDER))

        # ---- Runtime stacked bars ----
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

        # ---- Twin axis for percentage ----
        ax2 = ax.twinx()

        deleted_pct = s["deletion_ratio"] * 100.0
        instantiated_pct = 100.0 - deleted_pct

        ax2.bar(
            x + w,
            instantiated_pct,
            width=w,
            color=ZONE_COLOR_LIGHT,
            edgecolor="black",
            linewidth=0.3,
        )

        ax2.bar(
            x + w,
            deleted_pct,
            width=w,
            bottom=instantiated_pct,
            color=ZONE_COLOR_DARK,
            edgecolor="black",
            linewidth=0.3,
        )

        ax2.set_ylim(0, 100)

        # ---- Keep numeric ticks on all subplots ----
        ax.tick_params(axis="y", labelleft=True)
        ax2.tick_params(axis="y", labelright=True)

        # ---- Only outer axes get labels ----
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

# =============================================================================
# BUILD RADAR
# =============================================================================
def build_radar(df):

    fig = plt.figure(figsize=(16.5, 3.0))
    gs = fig.add_gridspec(1, 1)
    radar_spec = gs[0, 0].subgridspec(1, 5, wspace=0.15)

    plot_radar_row(fig, radar_spec, df)

    fig.legend(
        handles=[
            Line2D([0],[0], color=METHOD_COLORS[m],
                   linestyle=LINESTYLES[m], lw=1.5,
                   label=METHOD_LABEL[m])
            for m in METHOD_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=True
    )


    fig.subplots_adjust(top=0.68)
    fig.savefig("fig_radar.pdf")
    plt.close(fig)
# =============================================================================
# RUNTIME LEGEND HANDLES
# =============================================================================
def runtime_legend_handles():

    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p],
              hatch=PHASE_HATCH[p],
              edgecolor="black",
              linewidth=0.3,
              label=p)
        for p in PHASE_COLORS
    ]

    zone_handles = [
        Patch(facecolor=ZONE_COLOR_LIGHT,
              edgecolor="black",
              linewidth=0.3,
              label="Instantiated Cells"),
        Patch(facecolor=ZONE_COLOR_DARK,
              edgecolor="black",
              linewidth=0.3,
              label="Mask Size"),
    ]

    return phase_handles + zone_handles
# =============================================================================
# BUILD RUNTIME (SPACING REDUCED)
# =============================================================================
def build_runtime(df):

    fig = plt.figure(figsize=(16.5, 3.0))

    gs = fig.add_gridspec(1, 1)

    # 🔥 Reduced spacing here
    runtime_spec = gs[0, 0].subgridspec(1, 5, wspace=0.35)

    axes = plot_runtime_row(fig, runtime_spec, df)

    for ax, dataset in zip(axes, DATASET_ORDER):
        ax.set_title(dataset, pad=10, fontweight="bold")

    fig.legend(
        handles=runtime_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=5,
        frameon=True
    )

    fig.subplots_adjust(top=0.72)
    fig.savefig("fig_runtime.pdf")
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df = load_data()
    build_radar(df)
    build_runtime(df)
    print("Wrote fig_radar.pdf and fig_runtime.pdf")