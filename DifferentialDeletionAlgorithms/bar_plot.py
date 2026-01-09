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
# CSV Parsing (time-only)
# ------------------------------------------------------------------
REQUIRED_COLS = {
    "method", "dataset",
    "init_time", "model_time", "update_time",
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

                rows.append({
                    "method": method,
                    "dataset": dataset,
                    "init_time": fnum("init_time"),
                    "model_time": fnum("model_time"),
                    "update_time": fnum("update_time"),
                })
            except Exception:
                continue

    return rows

# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
def mean_of(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr)) if arr.size else 0.0

def aggregate_metrics(rows, datasets, methods):
    metrics = {m: {ds: {
        "init": 0.0,
        "model": 0.0,
        "update": 0.0,
    } for ds in datasets} for m in methods}

    buckets = {(m, ds): [] for m in methods for ds in datasets}
    for r in rows:
        if r["method"] in methods and r["dataset"] in datasets:
            buckets[(r["method"], r["dataset"])].append(r)

    for m in methods:
        for ds in datasets:
            b = buckets[(m, ds)]
            metrics[m][ds] = {
                "init": mean_of([x["init_time"] for x in b]),
                "model": mean_of([x["model_time"] for x in b]),
                "update": mean_of([x["update_time"] for x in b]),
            }

    return metrics

# ------------------------------------------------------------------
# Plotting (TIME ONLY)
# ------------------------------------------------------------------
def create_time_bar_plot(metrics, datasets, method_order):
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

    label_map = {
        "delgum": r"Gum",
        "delexp_canonical": r"Exp",
        "delmin": r"Min",
        "del2ph": r"2Ph",
    }

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        labels = [label_map.get(m, m) for m in method_order]

        init = np.array([metrics[m][dataset]["init"] for m in method_order])
        model = np.array([metrics[m][dataset]["model"] for m in method_order])
        update = np.array([metrics[m][dataset]["update"] for m in method_order])

        x = np.arange(len(labels))
        w = 0.6
        eps = 1e-12

        ax.bar(x, init + eps, w, color=colors["init"], label="Instantiation Time")
        ax.bar(x, model + eps, w, bottom=init + eps,
               color=colors["model"], label="Modeling Time")
        ax.bar(x, update + eps, w, bottom=init + model + eps,
               color=colors["update"], label="Update to Null Time")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Average Time (s)")

        title = "NCVoter" if dataset == "ncvoter" else dataset.title()
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        max_time = float(np.max(init + model + update))
        ax.set_ylim(0.0, max_time if max_time > 0 else 1.0)

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    file_paths = [
        "delmin_data_v2026_2.csv",
        "delgum_January09,202611:03:47AM_efflog.csv",
        "del2ph_January09,202611:02:23AM_efflog.csv",
    ]

    datasets = ["airport", "hospital", "flight", "adult", "tax"]
    method_order = ["delmin", "del2ph", "delgum"]

    all_rows = []
    for p in file_paths:
        all_rows.extend(parse_standardized_csv(p))

    metrics = aggregate_metrics(all_rows, datasets, set(method_order))
    fig = create_time_bar_plot(metrics, datasets, method_order)

    out = "bar_plot_standardized_times_efflog.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
