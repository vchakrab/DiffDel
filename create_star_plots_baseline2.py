#!/usr/bin/env python3
import csv
from math import pi
import os
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

# -----------------------------
# Global style
# -----------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.variant": "small-caps",
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": 12,
})

# Clockwise order (as requested):
# Mask Size, Leakage, Total Time, Model Size, Total Paths
CATEGORIES = [
    ("Mask Size",   "cells_sum",          "log"),     # log
    ("Leakage",     "leakage",            "linear"),  # linear (0..1)
    ("Total Time",  "total_time",         "log"),     # log
    ("Model Size",  "space_overhead_sum", "log"),     # log (MB)
    ("Total Paths", "dependencies",       "log"),     # log
]

BASELINE_DISPLAY = {
    "Baseline 3": "DelMin",
    "Exponential Deletion": "DelExp",
    "Greedy Gumbel": "DelGum",
    "2-Phase Deletion": "Del2Ph",
}

# Styles (you can tweak)
LINE_STYLE = {
    "DelMin": dict(color="red",      linestyle=":",  linewidth=1.6),
    "DelExp": dict(color="blue",     linestyle="--", linewidth=1.6),
    "DelGum": dict(color="#006400",  linestyle="-",  linewidth=1.6),  # deep green
    "Del2Ph": dict(color="black",    linestyle="-.", linewidth=1.6),
}

FILL_ALPHA = 0.06

# -----------------------------
# CSV parsing
# -----------------------------
def parse_csv_for_baselines(file_path: str):
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping: {file_path}")
        return data

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        current_dataset = None
        header_map = {}

        for row in reader:
            if not row:
                continue

            if row[0].startswith("-----"):
                current_dataset = row[0].strip("-").lower()
                data.setdefault(current_dataset, [])
                header_map = {}
                continue

            if not current_dataset:
                continue

            if not header_map:
                header_map = {h.strip(): i for i, h in enumerate(row)}
                continue

            try:
                time = float(row[header_map["total_time"]]) if "total_time" in header_map else 0.0

                dep_key = next((k for k in ["num_paths", "dependencies", "num_constraints", "num_explanations"]
                                if k in header_map), None)
                dependencies = float(row[header_map[dep_key]]) if dep_key else 0.0

                cells_key = next((k for k in ["mask_size", "cells_deleted"] if k in header_map), None)
                cells = int(row[header_map[cells_key]]) if cells_key else 0

                mem_key = next((k for k in ["memory_overhead_bytes", "memory_bytes"] if k in header_map), None)
                memory_mb = (float(row[header_map[mem_key]]) / (1024 ** 2)) if mem_key else 0.0

                leakage = float(row[header_map["leakage"]]) if "leakage" in header_map else 0.0
                leakage = max(0.0, min(1.0, leakage))  # keep per-run leakage sane

                # keep only if you truly need it; otherwise remove
                if "exponential_deletion_data" in file_path:
                    cells += 1

                data[current_dataset].append({
                    "time": time,
                    "dependencies": dependencies,
                    "cells": cells,
                    "space_overhead": memory_mb,  # MB
                    "leakage": leakage,
                })

            except (ValueError, IndexError, KeyError):
                continue

    return data


def calculate_aggregated_metrics(dataset_data):
    if not dataset_data:
        return None

    n = len(dataset_data)
    leakage_avg = sum(d["leakage"] for d in dataset_data) / n
    leakage_avg = max(0.0, min(1.0, leakage_avg))  # ensure in [0,1]

    return {
        "total_time": sum(d["time"] for d in dataset_data),
        "space_overhead_sum": sum(d["space_overhead"] for d in dataset_data),  # MB
        "cells_sum": sum(d["cells"] for d in dataset_data),
        "dependencies": sum(d["dependencies"] for d in dataset_data),
        "leakage": leakage_avg,  # <-- average, not sum
    }

# -----------------------------
# Scaling helpers
# -----------------------------
def log_norm(val: float, vmax: float) -> float:
    # log(val+1) so 0 maps to 0
    val = max(val, 0.0)
    vmax = max(vmax, 0.0)
    if vmax <= 0:
        return 0.0
    return float(np.log10(val + 1.0) / np.log10(vmax + 1.0))

def log_inv(r: float, vmax: float) -> float:
    vmax = max(vmax, 0.0)
    if vmax <= 0:
        return 0.0
    return float((vmax + 1.0) ** r - 1.0)

def lin_norm(val: float, vmax: float) -> float:
    if vmax <= 0:
        return 0.0
    return float(max(val, 0.0) / vmax)

def lin_inv(r: float, vmax: float) -> float:
    return float(r * vmax)

def fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    if v >= 1000:
        return f"{v/1000:.0f}K"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.2g}"

# -----------------------------
# Plot
# -----------------------------
def create_star_plot(ax, metrics_data, dataset_name, axis_max):
    labels = [c[0] for c in CATEGORIES]
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # spokes
    for a in angles:
        ax.plot([a, a], [0, 1], color="lightgrey", linewidth=0.7, alpha=0.7, zorder=1)

    # grid rings + per-axis tick labels
    grid_levels = [0.25, 0.5, 0.75, 1.0]
    for r in grid_levels:
        pts = np.array([(r, a) for a in angles_closed])
        ax.plot(pts[:, 1], pts[:, 0], color="lightgrey", linewidth=0.6, alpha=0.6, zorder=1)

        for i, (title, key, scale) in enumerate(CATEGORIES):
            vmax = axis_max[title]
            if title == "Leakage":
                # always linear ticks for leakage
                v = lin_inv(r, vmax)
                txt = f"{v:.2f}"
            else:
                if scale == "log":
                    v = log_inv(r, vmax)
                    txt = fmt_tick(v)
                else:
                    v = lin_inv(r, vmax)
                    txt = fmt_tick(v)

            ax.text(
                angles[i], r, txt,
                ha="center", va="center",
                fontsize=6, color="black",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.65),
                zorder=5
            )

    # plot each baseline (stable order)
    baseline_order = ["Baseline 3", "Exponential Deletion", "Greedy Gumbel", "2-Phase Deletion"]
    for b in baseline_order:
        if b not in metrics_data or not metrics_data[b]:
            continue

        display = BASELINE_DISPLAY.get(b, b)
        style = LINE_STYLE.get(display, dict(color="grey", linestyle="-", linewidth=1.5))

        raw = [float(metrics_data[b].get(key, 0.0)) for (_, key, _) in CATEGORIES]

        # normalize per-axis with independent vmax
        norm = []
        for val, (title, key, scale) in zip(raw, CATEGORIES):
            vmax = axis_max[title]
            if title == "Leakage":
                norm.append(lin_norm(max(0.0, min(1.0, val)), vmax))
            else:
                norm.append(log_norm(val, vmax) if scale == "log" else lin_norm(val, vmax))

        norm_closed = norm + norm[:1]

        ax.plot(
            angles_closed, norm_closed,
            label=display,
            marker="o", markersize=4,
            markeredgewidth=0.9, markeredgecolor="white",
            zorder=4,
            **style
        )
        ax.fill(angles_closed, norm_closed, alpha=FILL_ALPHA, color=style["color"], zorder=2)

    # category labels: horizontal, smaller, pushed outward
    for i, a in enumerate(angles):
        ax.text(a, 1.12, labels[i], ha="center", va="center", fontsize=9, rotation=0)

    display_ds = "NCVoter" if dataset_name.lower() == "ncvoter" else dataset_name.strip().title()
    ax.set_title(display_ds, fontsize=13, y=1.22)

    ax.set_ylim(0, 1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if "polar" in ax.spines:
        ax.spines["polar"].set_visible(False)


def compute_axis_max_for_dataset(metrics_data):
    """
    Independent axis scaling:
    each axis max = max value actually plotted for that dataset (no headroom).
    Leakage max is capped at 1.0 because leakage is a fraction.
    """
    axis_max = {}
    for (title, key, scale) in CATEGORIES:
        vals = []
        for _, m in metrics_data.items():
            if not m:
                continue
            vals.append(float(m.get(key, 0.0)))
        observed = max(vals) if vals else 0.0

        if title == "Leakage":
            observed = max(0.0, min(1.0, observed))
            axis_max[title] = observed if observed > 0 else 1.0
        else:
            axis_max[title] = observed if observed > 0 else 1.0

    return axis_max


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    baseline_files = {
        "Baseline 3": "delmin_data_v12.csv",
        "Exponential Deletion": "delexp_data_v12.csv",
        "Greedy Gumbel": "delgum_data_v12.csv",
        # "2-Phase Deletion": "2phase_deletion_data_v12.csv",
    }

    datasets_to_plot = ["airport", "hospital", "ncvoter", "onlineretail", "adult"]

    all_data = {name: parse_csv_for_baselines(path) for name, path in baseline_files.items()}

    final_metrics = {
        ds: {name: calculate_aggregated_metrics(all_data[name].get(ds, []))
             for name in baseline_files}
        for ds in datasets_to_plot
    }

    fig, axes = plt.subplots(
        1, len(datasets_to_plot),
        figsize=(30, 6),
        subplot_kw={"projection": "polar"}
    )
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes, datasets_to_plot):
        md = final_metrics.get(ds, {})
        if not md or all(v is None for v in md.values()):
            ax.set_title(("NCVoter" if ds == "ncvoter" else ds.title()) + "\n(No Data)", fontsize=13, y=1.22)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue

        axis_max = compute_axis_max_for_dataset(md)
        create_star_plot(ax, md, ds, axis_max)

    # legend (top) with consistent order
    handles, labels = axes[0].get_legend_handles_labels()
    desired = ["DelMin", "DelExp", "DelGum", "Del2Ph"]
    ordered = [(h, l) for l in desired for h, ll in zip(handles, labels) if ll == l]
    if ordered:
        handles, labels = zip(*ordered)

    fig.legend(list(handles), list(labels), loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.05), fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    out = "star_plots_AllBaselines_v12.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison star plot saved to {out}")
