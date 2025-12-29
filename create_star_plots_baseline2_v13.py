#!/usr/bin/env python3
import csv
from math import pi
import math
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

# -----------------------------
# Surprisal settings
# -----------------------------
# Surprisal = -log(1 - leakage)
# Use natural log by default (units = nats). If you want bits, set SURPRISAL_LOG_BASE = 2.
SURPRISAL_LOG_BASE = math.e

# Numerical safety for leakage very close to 1
LEAKAGE_EPS = 1e-12

def leakage_to_surprisal(leakage: float) -> float:
    """Convert leakage in [0,1] to surprisal = -log(1-leakage)."""
    l = float(leakage)
    l = max(0.0, min(1.0, l))
    one_minus = max(1.0 - l, LEAKAGE_EPS)  # avoid log(0)
    if SURPRISAL_LOG_BASE == 2:
        return float(-math.log2(one_minus))
    if SURPRISAL_LOG_BASE == 10:
        return float(-math.log10(one_minus))
    return float(-math.log(one_minus))  # natural log


# Clockwise order (as requested):
# Avg Mask Size, Surprisal, Avg Time, Avg Model Size, Avg Paths
CATEGORIES = [
    ("Avg Mask Size", "cells_sum",          "log"),     # log
    ("Surprisal",     "surprisal",          "log"),     # log (unbounded-ish, use log scaling)
    ("Avg Time",      "total_time",         "log"),     # log
    ("Avg Model Size","space_overhead_sum", "log"),     # log (MB)
    ("Avg Paths",     "dependencies",       "log"),     # log
]

BASELINE_DISPLAY = {
    "Baseline 3": "DelMin",
    "Exponential Deletion": "DelExp",
    "Greedy Gumbel": "DelGum"
    # "2-Phase Deletion": "Del2Ph",
}

# Styles
LINE_STYLE = {
    "DelMin": dict(color="red",      linestyle=":",  linewidth=1.6),
    "DelExp": dict(color="blue",     linestyle="--", linewidth=1.6),
    "DelGum": dict(color="#006400",  linestyle="-",  linewidth=1.6),  # deep green
    # "Del2Ph": dict(color="black",    linestyle="-.", linewidth=1.6),
}

FILL_ALPHA = 0.06


# -----------------------------
# CSV parsing (STANDARDIZED FORMAT)
# -----------------------------
def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default

def _norm_ds(ds: str) -> str:
    ds = (ds or "").strip().lower()
    if ds in ("online_retail", "onlineretail"):
        return "onlineretail"
    return ds

def parse_standardized_csv(file_path: str):
    """
    Reads standardized CSV rows like:
      method,dataset,target_attribute,total_time,init_time,model_time,del_time,update_time,
      leakage,utility,paths_blocked,mask_size,num_paths,memory_overhead_bytes,num_instantiated_cells,num_cells_updated

    Returns:
      data[dataset] = list of dicts with keys:
        time, dependencies, cells, space_overhead, leakage
    """
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping: {file_path}")
        return data

    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = _norm_ds(row.get("dataset", ""))
            if not ds:
                continue
            data.setdefault(ds, [])

            time_val = _safe_float(row.get("total_time"), 0.0)

            # In standardized CSV, use num_paths as "Avg Paths"
            num_paths = _safe_int(row.get("num_paths"), 0)

            cells = _safe_int(row.get("mask_size"), 0)

            mem_bytes = _safe_float(row.get("memory_overhead_bytes"), 0.0)
            memory_mb = mem_bytes / (1024 ** 2)

            leakage = _safe_float(row.get("leakage"), 0.0)
            leakage = max(0.0, min(1.0, leakage))

            data[ds].append({
                "time": time_val,
                "dependencies": float(max(0, num_paths)),
                "cells": int(max(0, cells)),
                "space_overhead": float(max(0.0, memory_mb)),  # MB
                "leakage": float(leakage),
            })

    return data


def calculate_aggregated_metrics(dataset_data):
    if not dataset_data:
        return None

    n = len(dataset_data)
    if n == 0:
        return {
            "total_time": 0.0,
            "space_overhead_sum": 0.0,
            "cells_sum": 0.0,
            "dependencies": 0.0,
            "surprisal": 0.0,
        }

    leakage_avg = sum(d["leakage"] for d in dataset_data) / n
    leakage_avg = max(0.0, min(1.0, leakage_avg))
    surprisal = leakage_to_surprisal(leakage_avg)

    return {
        "total_time": sum(d["time"] for d in dataset_data) / n,
        "space_overhead_sum": sum(d["space_overhead"] for d in dataset_data) / n,
        "cells_sum": sum(d["cells"] for d in dataset_data) / n,
        "dependencies": sum(d["dependencies"] for d in dataset_data) / n,
        "surprisal": float(max(0.0, surprisal)),
    }


# -----------------------------
# Scaling helpers
# -----------------------------
def log_norm(val: float, vmax: float) -> float:
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

    # plot each baseline
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

    # category labels (horizontal)
    for i, a in enumerate(angles):
        ax.text(a, 1.12, labels[i], ha="center", va="center", fontsize=9, rotation=0)

    display_ds = "NCVoter" if dataset_name.lower() == "ncvoter" else dataset_name.strip().title()
    if dataset_name.lower() == "onlineretail":
        display_ds = "Onlineretail"
    ax.set_title(display_ds, fontsize=13, y=1.22)

    ax.set_ylim(0, 1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if "polar" in ax.spines:
        ax.spines["polar"].set_visible(False)


def compute_axis_max_for_dataset(metrics_data):
    axis_max = {}
    for (title, key, scale) in CATEGORIES:
        vals = []
        for _, m in metrics_data.items():
            if not m:
                continue
            vals.append(float(m.get(key, 0.0)))
        observed = max(vals) if vals else 0.0
        axis_max[title] = observed if observed > 0 else 1.0
    return axis_max


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Point these to your standardized outputs
    baseline_files = {
        "Baseline 3": "delmin_data_standardized_vFinal.csv",
        "Exponential Deletion": "delexp_data_standardized_vFinal.csv",
        "Greedy Gumbel": "delgum_data_standardized_vFinal.csv",
        # "2-Phase Deletion": "2phase_data_standardized.csv",
    }

    datasets_to_plot = ["airport", "hospital", "ncvoter", "onlineretail", "adult", "tax"]

    all_data = {name: parse_standardized_csv(path) for name, path in baseline_files.items()}

    final_metrics = {
        ds: {
            name: calculate_aggregated_metrics(all_data[name].get(ds, []))
            for name in baseline_files
        }
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
            title = ("NCVoter" if ds == "ncvoter" else ("Onlineretail" if ds == "onlineretail" else ds.title()))
            ax.set_title(title + "\n(No Data)", fontsize=13, y=1.22)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue

        axis_max = compute_axis_max_for_dataset(md)
        create_star_plot(ax, md, ds, axis_max)

    # legend (top)
    handles, labels = axes[0].get_legend_handles_labels()
    desired = ["DelMin", "DelExp", "DelGum", "Del2Ph"]
    ordered = [(h, l) for l in desired for h, ll in zip(handles, labels) if ll == l]
    if ordered:
        handles, labels = zip(*ordered)

    fig.legend(list(handles), list(labels), loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.05), fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    out = "star_plots_standardized_avg_surprisal.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison star plot saved to {out}")
