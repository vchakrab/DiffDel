#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Global matplotlib configuration
# ------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.variant": "small-caps",
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": 13,
})

# ------------------------------------------------------------------
# CSV Parsing
# ------------------------------------------------------------------
REQUIRED_COLS = {
    "method", "dataset",
    "init_time", "model_time", "update_time",
    # for the shaded "cells" bar plot
    "num_instantiated_cells", "mask_size",
}

def parse_standardized_csv(file_path: str):
    if not os.path.exists(file_path):
        print(f"[WARN] File not found: {file_path}")
        return []

    rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"[WARN] No header found in: {file_path}")
            return []

        fieldnames = [h.strip() for h in reader.fieldnames]
        fieldset = set(fieldnames)

        missing = REQUIRED_COLS - fieldset
        if missing:
            print(f"[WARN] Missing required columns in {file_path}: {sorted(missing)}")

        def fnum(r, key, default=0.0):
            v = r.get(key, "")
            if v is None:
                return default
            v = str(v).strip()
            if v == "":
                return default
            try:
                return float(v)
            except Exception:
                return default

        for r in reader:
            method = (r.get("method") or "").strip().lower()
            dataset = (r.get("dataset") or "").strip().lower()
            if not method or not dataset:
                continue

            rows.append({
                "method": method,
                "dataset": dataset,
                "init_time": fnum(r, "init_time"),
                "model_time": fnum(r, "model_time"),
                "update_time": fnum(r, "update_time"),
                "num_instantiated_cells": fnum(r, "num_instantiated_cells"),
                "mask_size": fnum(r, "mask_size"),
            })

    return rows

# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
def mean_of(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr)) if arr.size else 0.0

def aggregate_metrics(rows, datasets, methods):
    """
    Returns:
      metrics_time[method][dataset] = {init, model, update}
      metrics_cells[method][dataset] = {instantiated, deleted}
    """
    metrics_time = {m: {ds: {"init": 0.0, "model": 0.0, "update": 0.0} for ds in datasets} for m in methods}
    metrics_cells = {m: {ds: {"instantiated": 0.0, "deleted": 0.0} for ds in datasets} for m in methods}

    buckets = {(m, ds): [] for m in methods for ds in datasets}
    for r in rows:
        if r["method"] in methods and r["dataset"] in datasets:
            buckets[(r["method"], r["dataset"])].append(r)

    for m in methods:
        for ds in datasets:
            b = buckets[(m, ds)]

            init_avg = mean_of([x["init_time"] for x in b])
            model_avg = mean_of([x["model_time"] for x in b])
            update_avg = mean_of([x["update_time"] for x in b])

            # ✅ force del2ph init_time to plot as 0 (as requested)
            if m == "del2ph":
                init_avg = 0.0

            metrics_time[m][ds] = {"init": init_avg, "model": model_avg, "update": update_avg}

            metrics_cells[m][ds] = {
                "instantiated": mean_of([x["num_instantiated_cells"] for x in b]),
                "deleted": mean_of([x["mask_size"] for x in b]),
            }

    return metrics_time, metrics_cells

# ------------------------------------------------------------------
# Plotting (TIME)
# ------------------------------------------------------------------
def create_time_bar_plot(metrics_time, metrics_cells, datasets, method_order):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 7))
    if len(datasets) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = {
        "init": "#56B4E9",
        "model": "#D55E00",
        "update": "#009E73",
    }
    cell_color = "#7F7F7F"

    label_map = {
        "delgum": r"Gum",
        "delexp_canonical": r"Exp",
        "delmin": r"Min",
        "del2ph": r"2Ph",
    }

    # bar geometry: time bar left, cell bar right
    group_w = 0.8
    w = group_w / 2.2
    dx = w / 1.2

    ax2_list = []

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        ax2 = ax.twinx()
        ax2_list.append(ax2)

        # keep twin axis background transparent
        ax2.patch.set_alpha(0.0)

        labels = [label_map.get(m, m) for m in method_order]
        x = np.arange(len(labels))
        x_time = x - dx
        x_cell = x + dx
        eps = 1e-12

        # -------------------------------
        # LEFT AXIS: TIME (stacked) - left bar in each group
        # -------------------------------
        init = np.array([metrics_time[m][dataset]["init"] for m in method_order], dtype=float)
        model = np.array([metrics_time[m][dataset]["model"] for m in method_order], dtype=float)
        update = np.array([metrics_time[m][dataset]["update"] for m in method_order], dtype=float)

        ax.bar(x_time, init + eps, w, color=colors["init"], label="Instantiation Time", zorder=3)
        ax.bar(x_time, model + eps, w, bottom=init + eps,
               color=colors["model"], label="Modeling Time", zorder=3)
        ax.bar(x_time, update + eps, w, bottom=init + model + eps,
               color=colors["update"], label="Update to Null Time", zorder=3)

        # -------------------------------
        # RIGHT AXIS: CELLS (stacked) - right bar in each group
        # total height = deleted
        # hatched part = initialized (capped by deleted)
        # -------------------------------
        instantiated = np.array([metrics_cells[m][dataset]["instantiated"] for m in method_order], dtype=float)
        deleted = np.array([metrics_cells[m][dataset]["deleted"] for m in method_order], dtype=float)

        inst_part = np.minimum(instantiated, deleted)
        rem_deleted = np.maximum(deleted - inst_part, 0.0)

        # bottom: initialized (hatched)
        ax2.bar(
            x_cell, inst_part + eps, w,
            color=cell_color, alpha=0.25,
            hatch="///", edgecolor="black", linewidth=0.8,
            label="Initialized Cells",
            zorder=2
        )
        # top: remaining deleted (solid)
        ax2.bar(
            x_cell, rem_deleted + eps, w,
            bottom=inst_part + eps,
            color=cell_color, alpha=0.45,
            edgecolor="black", linewidth=0.8,
            label="Deleted Cells",
            zorder=2
        )

        # -------------------------------
        # Axes cosmetics
        # -------------------------------
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Average Time (s)")
        ax2.set_ylabel("Average Cell Count")

        title = "NCVoter" if dataset == "ncvoter" else dataset.title()
        ax.set_title(title)

        ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

        # y-lims
        max_time = float(np.max(init + model + update)) if len(init) else 0.0
        ax.set_ylim(0.0, max_time if max_time > 0 else 1.0)

        max_cells = float(np.max(deleted)) if len(deleted) else 0.0
        ax2.set_ylim(0.0, max_cells if max_cells > 0 else 1.0)

        # optional: scientific notation for big cell counts
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # -------------------------------
    # Global legend: collect from first subplot (both axes) and dedupe
    # -------------------------------
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2_list[0].get_legend_handles_labels()

    handles, labels = [], []
    for hh, ll in list(zip(h1, l1)) + list(zip(h2, l2)):
        if ll not in labels:
            handles.append(hh)
            labels.append(ll)

    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ------------------------------------------------------------------
# Plotting (CELLS: "deleted" bar with hatched "instantiated" portion)
# ------------------------------------------------------------------
def create_cells_shaded_bar_plot(metrics_cells, datasets, method_order):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 7))
    if len(datasets) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # One base color for the "deleted cells" total; hatched overlay indicates instantiated portion.
    deleted_color = "#7F7F7F"  # neutral gray

    label_map = {
        "delgum": r"Gum",
        "delexp_canonical": r"Exp",
        "delmin": r"Min",
        "del2ph": r"2Ph",
    }

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        labels = [label_map.get(m, m) for m in method_order]

        instantiated = np.array([metrics_cells[m][dataset]["instantiated"] for m in method_order], dtype=float)
        deleted = np.array([metrics_cells[m][dataset]["deleted"] for m in method_order], dtype=float)

        # We visualize "deleted" as the total bar height.
        # The "instantiated" portion is shown as a hatched segment inside the deleted bar.
        hatched_part = np.minimum(instantiated, deleted)
        remainder_deleted = np.maximum(deleted - hatched_part, 0.0)

        x = np.arange(len(labels))
        w = 0.6
        eps = 1e-12

        # Bottom hatched portion (instantiated, capped to deleted)
        ax.bar(
            x, hatched_part + eps, w,
            color=deleted_color, alpha=0.35,
            hatch="///",
            edgecolor="black",
            label="Initialized Cells (hatched)"
        )

        # Top solid portion (deleted cells not in initialized portion)
        ax.bar(
            x, remainder_deleted + eps, w,
            bottom=hatched_part + eps,
            color=deleted_color, alpha=0.85,
            edgecolor="black",
            label="Deleted Cells (total)"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Average Cell Count")

        title = "NCVoter" if dataset == "ncvoter" else dataset.title()
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        max_y = float(np.max(hatched_part + remainder_deleted)) if len(hatched_part) else 0.0
        ax.set_ylim(0.0, max_y if max_y > 0 else 1.0)

    # Global legend (dedupe by label)
    h0, l0 = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(h0, l0):
        if l not in uniq:
            uniq[l] = h
    fig.legend(list(uniq.values()), list(uniq.keys()),
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    file_paths = [
        "delmin_final_data.csv",
        "delgum_January10,202610:46:58AM_hinge.csv",
        "del2ph_January09,202610:50:09PM_hinge.csv",
    ]

    datasets = ["airport", "hospital", "flight", "adult", "tax"]
    method_order = ["delmin", "del2ph", "delgum"]

    all_rows = []
    for p in file_paths:
        all_rows.extend(parse_standardized_csv(p))

    metrics_time, metrics_cells = aggregate_metrics(all_rows, datasets, set(method_order))

    # 1) Time stacked bars
    fig = create_time_bar_plot(metrics_time, metrics_cells, datasets, method_order)


    out_time = "bar_plot_standardized_times_efflog.pdf"
    fig.savefig(out_time, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_time}")

    # 2) Cells shaded bars (deleted total with hatched initialized portion)
    # fig_cells = create_cells_shaded_bar_plot(metrics_cells, datasets, method_order)
    # out_cells = "bar_plot_cells_shaded_efflog.pdf"
    # fig_cells.savefig(out_cells, bbox_inches="tight")
    # plt.close(fig_cells)
    # print(f"Saved {out_cells}")

if __name__ == "__main__":
    main()
