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
# CSV Parsing (NEW format: one header, with method + dataset columns)
# ------------------------------------------------------------------
REQUIRED_COLS = {
    "method", "dataset",
    "init_time", "model_time", "update_time",
    "memory_overhead_bytes"
}

def parse_standardized_csv(file_path: str):
    """
    Reads a 'standardized' CSV with columns like:
      method,dataset,...,init_time,model_time,del_time,update_time,...,memory_overhead_bytes,...
    Returns: list of dict rows with normalized method/dataset and numeric metrics.
    """
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
            # still try to read; but metrics may be zeros

        for r in reader:
            try:
                method = (r.get("method") or "").strip().lower()
                dataset = (r.get("dataset") or "").strip().lower()

                if not method or not dataset:
                    continue

                def fnum(key, default=0.0):
                    v = r.get(key, "")
                    if v is None:
                        return default
                    v = str(v).strip()
                    if v == "":
                        return default
                    return float(v)

                init_time = fnum("init_time", 0.0)
                model_time = fnum("model_time", 0.0)
                update_time = fnum("update_time", 0.0)

                mem_bytes = fnum("memory_overhead_bytes", 0.0)

                rows.append({
                    "method": method,
                    "dataset": dataset,
                    "init_time": init_time,
                    "model_time": model_time,
                    "update_time": update_time,
                    "memory_overhead_bytes": mem_bytes
                })
            except Exception:
                # skip malformed row
                continue

    return rows

# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
def mean_of(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))

def aggregate_metrics(rows, datasets, methods):
    """
    Returns:
      metrics[method][dataset] = dict(init_time_avg, model_time_avg, update_time_avg, memory_mb_avg)
    """
    metrics = {m: {ds: {
        "init_time_avg": 0.0,
        "model_time_avg": 0.0,
        "update_time_avg": 0.0,
        "memory_mb_avg": 0.0,
    } for ds in datasets} for m in methods}

    # bucket
    buckets = {(m, ds): [] for m in methods for ds in datasets}
    for r in rows:
        m = r["method"]
        ds = r["dataset"]
        if m in methods and ds in datasets:
            buckets[(m, ds)].append(r)

    # compute means
    for m in methods:
        for ds in datasets:
            b = buckets[(m, ds)]
            metrics[m][ds] = {
                "init_time_avg": mean_of([x["init_time"] for x in b]),
                "model_time_avg": mean_of([x["model_time"] for x in b]),
                "update_time_avg": mean_of([x["update_time"] for x in b]),
                "memory_mb_avg": mean_of([x["memory_overhead_bytes"] for x in b]) / (1024.0 ** 2),
            }

    return metrics

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def create_combined_bar_plot(metrics, datasets, method_order):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 7))
    if len(datasets) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = {
        "Instantiation Time": "#56B4E9",
        "Modeling Time": "#D55E00",
        "Update to NULL Time": "#009E73",
        "Model Size": "brown",
    }

    # display labels for methods
    label_map = {
        "delgum": r"DelGum",
        "delexp": r"DelExp",
        "delmin": r"DelMin",
    }

    for i, dataset in enumerate(datasets):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        labels = [label_map.get(m, m) for m in method_order]

        init = np.array([metrics[m][dataset]["init_time_avg"] for m in method_order], dtype=float)
        model = np.array([metrics[m][dataset]["model_time_avg"] for m in method_order], dtype=float)
        update = np.array([metrics[m][dataset]["update_time_avg"] for m in method_order], dtype=float)
        memory = np.array([metrics[m][dataset]["memory_mb_avg"] for m in method_order], dtype=float)

        x = np.arange(len(labels))
        w = 0.35
        eps = 1e-12

        # stacked time bar (left axis)
        ax1.bar(x - w/2, init + eps, w, bottom=0, color=colors["Instantiation Time"], label="Instantiation Time")
        ax1.bar(x - w/2, model + eps, w, bottom=init + eps, color=colors["Modeling Time"], label="Modeling Time")
        ax1.bar(
            x - w/2, update + eps, w,
            bottom=init + model + eps,
            color=colors["Update to NULL Time"],
            label="Update to Null Time"
        )

        # memory bar (right axis)
        ax2.bar(x + w/2, memory + eps, w, color=colors["Model Size"], label="Model Size")

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=0)

        ax1.set_ylabel("Average Time (s)")
        ax2.set_ylabel("Model Size (MB)")

        if dataset == "ncvoter":
            title = "NCVoter"
        elif dataset == "onlineretail":
            title = "Onlineretail"
        else:
            title = dataset.title()
        ax1.set_title(title)

        ax1.grid(axis="y", linestyle="--", alpha=0.6)

        # ----------------------------------------------------------
        # FIX: force y-max to equal the max plotted value (no padding)
        # ----------------------------------------------------------
        time_stack = init + model + update
        max_time = float(np.max(time_stack)) if time_stack.size else 0.0
        max_mem = float(np.max(memory)) if memory.size else 0.0

        # Avoid a zero-height axis if everything is 0
        if max_time <= 0.0:
            ax1.set_ylim(0.0, 1.0)
        else:
            ax1.set_ylim(0.0, max_time)

        if max_mem <= 0.0:
            ax2.set_ylim(0.0, 1.0)
        else:
            ax2.set_ylim(0.0, max_mem)

    # global legend (dedupe)
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                labels.append(ll)
                handles.append(hh)

    desired_order = ["Instantiation Time", "Modeling Time", "Update to Null Time", "Model Size"]
    ordered_handles = [handles[labels.index(l)] for l in desired_order if l in labels]

    fig.legend(
        ordered_handles,
        desired_order,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.99)
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    file_paths = [
        "delmin_data_standarized_f3.csv",
        "delexp_data_standardized_non_canonical.csv",
    ]

    datasets = ["airport", "hospital", "ncvoter", "onlineretail", "adult", "tax"]
    method_order = ["delmin", "delexp"]  # fixed plot order

    all_rows = []
    for p in file_paths:
        all_rows.extend(parse_standardized_csv(p))

    metrics = aggregate_metrics(all_rows, datasets, set(method_order))

    fig = create_combined_bar_plot(metrics, datasets, method_order)
    out = "bar_plot_standardized_times_and_memory.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
