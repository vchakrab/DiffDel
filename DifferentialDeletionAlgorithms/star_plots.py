#!/usr/bin/env python3
import csv
import os
import sys
from math import pi
import numpy as np

import matplotlib.pyplot as plt

# =============================
# USER CONFIG
# =============================

CSV_PATHS = {
    "Baseline 3": "delmin_data_v2026_2.csv",
    "Exponential Deletion": "del2ph_January09,202611:02:23AM_efflog.csv",
    "Greedy Gumbel": "delgum_January09,202611:03:47AM_efflog.csv",
}

DATASETS = ["airport", "hospital", "flight", "adult", "tax"]

OUTPUT_PDF = "star_plots_avg_metrics_efflog.pdf"

# =============================
# Radar categories (clockwise)
# =============================
# (label, aggregated_key, scale)
CATEGORIES = [
    ("Avg Mask Size",  "mask_size",    "log"),
    ("Leakage",        "leakage",      "linear"),
    ("Avg Time",       "total_time",   "log"),
    ("Avg Memory (MB)","memory_mb",    "log"),
    ("Avg Paths",      "paths_blocked","log"),
]

DISPLAY_NAME = {
    "Baseline 3": "Min",
    "Exponential Deletion": "2Ph",
    "Greedy Gumbel": "Gum",
}

LINE_STYLE = {
    "DelMin": dict(color="red",     linestyle=":",  linewidth=1.6),
    "DelExp": dict(color="blue",    linestyle="--", linewidth=1.6),
    "DelGum": dict(color="#006400", linestyle="-",  linewidth=1.6),
}

FILL_ALPHA = 0.06

# =============================
# Helpers
# =============================

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default

def norm_ds(ds: str) -> str:
    ds = (ds or "").strip().lower()
    if ds in ("online_retail", "onlineretail"):
        return "onlineretail"
    return ds

# =============================
# Load + aggregate
# =============================

def load_and_aggregate(csv_path):
    sums = {}
    cnts = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = norm_ds(row.get("dataset"))
            if not ds:
                continue

            sums.setdefault(ds, {
                "mask_size": 0.0,
                "leakage": 0.0,
                "total_time": 0.0,
                "memory_mb": 0.0,
                "paths_blocked": 0.0,
            })
            cnts.setdefault(ds, 0)

            sums[ds]["mask_size"] += safe_float(row.get("mask_size"))
            sums[ds]["leakage"] += safe_float(row.get("leakage"))
            sums[ds]["total_time"] += safe_float(row.get("total_time"))
            sums[ds]["paths_blocked"] += safe_float(row.get("paths_blocked"))
            sums[ds]["memory_mb"] += safe_float(row.get("memory_overhead_bytes")) / (1024 ** 2)

            cnts[ds] += 1

    avgs = {}
    for ds in sums:
        n = max(1, cnts[ds])
        avgs[ds] = {k: v / n for k, v in sums[ds].items()}

    return avgs

# =============================
# Scaling
# =============================

def log_norm(v, vmax):
    return np.log10(v + 1.0) / np.log10(vmax + 1.0) if vmax > 0 else 0.0

def lin_norm(v, vmax):
    return v / vmax if vmax > 0 else 0.0

def inv_log(r, vmax):
    return (vmax + 1.0) ** r - 1.0

# =============================
# Radar plot
# =============================

def compute_axis_max(metrics):
    out = {}
    for title, key, scale in CATEGORIES:
        if scale == "linear" and key == "leakage":
            out[title] = 1.0
        else:
            out[title] = max(
                (m.get(key, 0.0) for m in metrics.values() if m),
                default=1.0
            )
    return out

def star_plot(ax, metrics_by_method, dataset, axis_max):
    labels = [c[0] for c in CATEGORIES]
    keys   = [c[1] for c in CATEGORIES]
    scales = [c[2] for c in CATEGORIES]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # grid
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1], color="lightgrey", linewidth=0.7)

    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r]*len(angles), color="lightgrey", linewidth=0.6)

    # data
    for method, metrics in metrics_by_method.items():
        if not metrics:
            continue

        disp = DISPLAY_NAME.get(method, method)
        style = LINE_STYLE.get(disp, {})

        raw = [metrics.get(k, 0.0) for k in keys]
        norm = []
        for v, (title, _, scale) in zip(raw, CATEGORIES):
            vmax = axis_max[title]
            if scale == "log":
                norm.append(log_norm(v, vmax))
            else:
                norm.append(lin_norm(v, vmax))

        norm += norm[:1]

        ax.plot(angles, norm, label=disp, **style)
        ax.fill(angles, norm, color=style.get("color", "gray"), alpha=FILL_ALPHA)

    for a, lab in zip(angles[:-1], labels):
        ax.text(a, 1.12, lab, ha="center", fontsize=9)

    ax.set_title(dataset.title(), y=1.22)
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_visible(False)

# =============================
# Main
# =============================

def main():
    all_metrics = {
        method: load_and_aggregate(path)
        for method, path in CSV_PATHS.items()
    }

    fig, axes = plt.subplots(
        1, len(DATASETS),
        figsize=(30, 6),
        subplot_kw={"projection": "polar"},
    )

    for ax, ds in zip(axes, DATASETS):
        metrics = {
            method: all_metrics[method].get(ds)
            for method in CSV_PATHS
        }
        axis_max = compute_axis_max(metrics)
        star_plot(ax, metrics, ds, axis_max)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])

    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
