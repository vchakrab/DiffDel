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
# Radar categories (CLOCKWISE)
# -----------------------------
# Avg Mask Size, Leakage, Avg Time, Avg Model Size, Avg Paths
CATEGORIES = [
    ("Avg Mask Size",  "cells_sum",          "log"),
    ("Leakage",        "leakage",            "log"),
    ("Avg Time",       "total_time",         "log"),
    ("Avg Model Size", "space_overhead_sum", "log"),
    ("Avg Paths",      "dependencies",       "log"),
]

BASELINE_DISPLAY = {
    "Baseline 3": "DelMin",
    "Exponential Deletion": "DelExp",
    "Greedy Gumbel": "DelGum"
}

LINE_STYLE = {
    "DelMin": dict(color="red",      linestyle=":",  linewidth=1.6),
    "DelExp": dict(color="blue",     linestyle="--", linewidth=1.6),
    "DelGum": dict(color="#006400",  linestyle="-",  linewidth=1.6),
}

FILL_ALPHA = 0.06


# -----------------------------
# CSV parsing
# -----------------------------
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def _norm_ds(ds: str) -> str:
    ds = (ds or "").strip().lower()
    if ds in ("online_retail", "onlineretail"):
        return "onlineretail"
    return ds

def parse_standardized_csv(file_path: str):
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return data

    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = _norm_ds(row.get("dataset", ""))
            if not ds:
                continue

            data.setdefault(ds, [])

            leakage = max(0.0, min(1.0, _safe_float(row.get("leakage"), 0.0)))

            data[ds].append({
                "time": _safe_float(row.get("total_time"), 0.0),
                "dependencies": _safe_int(row.get("num_paths"), 0),
                "cells": _safe_int(row.get("mask_size"), 0),
                "space_overhead": _safe_float(row.get("memory_overhead_bytes"), 0.0) / (1024 ** 2),
                "leakage": leakage,
            })

    return data


def calculate_aggregated_metrics(dataset_data):
    if not dataset_data:
        return None

    n = len(dataset_data)
    return {
        "total_time": sum(d["time"] for d in dataset_data) / n,
        "space_overhead_sum": sum(d["space_overhead"] for d in dataset_data) / n,
        "cells_sum": sum(d["cells"] for d in dataset_data) / n,
        "dependencies": sum(d["dependencies"] for d in dataset_data) / n,
        "leakage": sum(d["leakage"] for d in dataset_data) / n,
    }


# -----------------------------
# Scaling helpers
# -----------------------------
def log_norm(val: float, vmax: float) -> float:
    return np.log10(val + 1.0) / np.log10(vmax + 1.0) if vmax > 0 else 0.0

def log_inv(r: float, vmax: float) -> float:
    return (vmax + 1.0) ** r - 1.0 if vmax > 0 else 0.0

def fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    if v >= 1000:
        return f"{v/1000:.0f}K"
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.2g}"


# -----------------------------
# Radar plot
# -----------------------------
def create_star_plot(ax, metrics_data, dataset_name, axis_max):
    labels = [c[0] for c in CATEGORIES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    for a in angles:
        ax.plot([a, a], [0, 1], color="lightgrey", linewidth=0.7)

    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles_closed, [r]*len(angles_closed), color="lightgrey", linewidth=0.6)

        for i, (title, key, _) in enumerate(CATEGORIES):
            v = log_inv(r, axis_max[title])
            ax.text(
                angles[i], r, fmt_tick(v),
                fontsize=6, ha="center", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.7)
            )

    for baseline, metrics in metrics_data.items():
        if not metrics:
            continue

        disp = BASELINE_DISPLAY.get(baseline, baseline)
        style = LINE_STYLE.get(disp, {})

        raw = [metrics.get(k, 0.0) for (_, k, _) in CATEGORIES]
        norm = [log_norm(v, axis_max[t]) for v, (t, _, _) in zip(raw, CATEGORIES)]
        norm_closed = norm + norm[:1]

        ax.plot(angles_closed, norm_closed, label=disp, **style)
        ax.fill(angles_closed, norm_closed, alpha=FILL_ALPHA, color=style.get("color", "gray"))

    for i, a in enumerate(angles):
        ax.text(a, 1.12, labels[i], ha="center", fontsize=9)

    ax.set_title(dataset_name.title(), fontsize=13, y=1.22)
    ax.set_ylim(0, 1.0)
    ax.spines["polar"].set_visible(False)


def compute_axis_max(metrics_data):
    axis_max = {}
    for title, key, _ in CATEGORIES:
        if key == "leakage":
            axis_max[title] = 1.0   # Leakage is probabilistic
        else:
            axis_max[title] = max(
                (m.get(key, 0.0) for m in metrics_data.values() if m),
                default=1.0
            )
    return axis_max



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    baseline_files = {
        "Baseline 3": "delmin_data_standarized_f3.csv",
        "Exponential Deletion": "delexp_data_standardized_non_canonical_v3.csv",
        "Greedy Gumbel": "delgum_data_standardized_vFinal.csv",
    }

    datasets = ["airport", "hospital", "ncvoter", "onlineretail", "adult", "tax"]

    all_data = {b: parse_standardized_csv(p) for b, p in baseline_files.items()}

    metrics = {
        ds: {
            b: calculate_aggregated_metrics(all_data[b].get(ds, []))
            for b in baseline_files
        }
        for ds in datasets
    }

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(30, 6),
        subplot_kw={"projection": "polar"}
    )

    for ax, ds in zip(axes, datasets):
        md = metrics.get(ds, {})
        axis_max = compute_axis_max(md)
        create_star_plot(ax, md, ds, axis_max)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])

    out = "star_plots_standardized_avg_leakage.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
