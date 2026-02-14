#!/usr/bin/env python3

import os
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

Z_95 = 1.96

DATASET_COLORS = {
    "Airport": "#1f77b4",
    "Hospital": "#ff7f0e",
    "Adult": "#2ca02c",
    "Flight": "#d62728",
    "Tax": "#9467bd",
}

def parse_sweep(fname, sweep):
    patterns = {
        "L0": r"L0[_=]?([0-9.]+)",
        "lambda": r"(?:lambda|lam)[_=]?([0-9.]+)",
        "epsilon": r"(?:epsilon|eps)[_=]?([0-9.]+)"
    }
    m = re.search(patterns[sweep], fname, re.IGNORECASE)
    return float(m.group(1)) if m else None

def load_data(ablation_dir, sweep, mech):
    rows = []
    for root, _, files in os.walk(ablation_dir):
        for f in files:
            if not f.endswith(".csv"):
                continue

            sval = parse_sweep(f, sweep)
            if sval is None:
                continue

            if mech != "Both":
                if mech == "Exp" and ("gum" in f.lower()):
                    continue
                if mech == "Gum" and ("gum" not in f.lower()):
                    continue

            df = pd.read_csv(Path(root)/f)
            if not {"dataset","mask_size","leakage"}.issubset(df.columns):
                continue

            df = df.copy()
            df["sweep"] = sval
            df["leakage"] = pd.to_numeric(df["leakage"], errors="coerce")
            df["mask_size"] = pd.to_numeric(df["mask_size"], errors="coerce")

            rows.append(df)

    if not rows:
        raise RuntimeError("No matching CSVs found.")

    df = pd.concat(rows, ignore_index=True)
    return df.dropna()

def compute_stats(df):
    df["leak_pct"] = np.where(df["leakage"] <= 1.5,
                              df["leakage"]*100,
                              df["leakage"])

    stats = (
        df.groupby(["dataset","sweep"])
          .agg(
              n=("leak_pct","count"),
              leak_mean=("leak_pct","mean"),
              leak_std=("leak_pct","std"),
              mask_mean=("mask_size","mean")
          )
          .reset_index()
    )
    stats["leak_ci"] = Z_95 * (stats["leak_std"].fillna(0) /
                               np.sqrt(stats["n"].clip(lower=1)))
    return stats

def make_regions(ax, x, sweeps):
    # distinct categorical colors
    unique = np.unique(sweeps)
    cmap = plt.get_cmap("tab20")
    color_map = {v: cmap(i % 20) for i,v in enumerate(sorted(unique))}

    # compute boundaries
    if len(x) == 1:
        left = [x[0]-0.2]
        right = [x[0]+0.2]
    else:
        mids = (x[:-1] + x[1:]) / 2
        left = np.concatenate(([x[0]-(mids[0]-x[0])], mids))
        right = np.concatenate((mids, [x[-1]+(x[-1]-mids[-1])]))

    patches = []
    seen = set()

    for i in range(len(x)):
        val = sweeps[i]
        col = color_map[val]
        ax.axvspan(left[i], right[i], color=col, alpha=0.18, zorder=0)
        if val not in seen:
            patches.append(Patch(facecolor=col,
                                 label=f"{val:g}",
                                 alpha=0.4))
            seen.add(val)

    return patches

def plot_dataset(stats, dataset, sweep_label, out_path):
    sub = stats[stats["dataset"]==dataset].sort_values("mask_mean")
    if sub.empty:
        return

    x = sub["mask_mean"].to_numpy()
    y = sub["leak_mean"].to_numpy()
    ci = sub["leak_ci"].to_numpy()
    sweeps = sub["sweep"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5,4.3))

    patches = make_regions(ax, x, sweeps)

    base_color = DATASET_COLORS.get(dataset, "#333333")
    ax.fill_between(x, y-ci, y+ci,
                    color=base_color,
                    alpha=0.25)
    ax.plot(x, y,
            color=base_color,
            lw=2.4)

    ax.set_xlabel("Mask Size")
    ax.set_ylabel("Leakage (%)")
    ax.set_title(f"{dataset} (regions by {sweep_label})")

    ax.legend(handles=patches,
              title="Ablation values",
              frameon=False,
              loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("Wrote", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sweep", required=True,
                        choices=["L0","lambda","epsilon"])
    parser.add_argument("--mech", default="Exp",
                        choices=["Exp","Gum","Both"])
    args = parser.parse_args()

    df = load_data(args.ablation_dir, args.sweep, args.mech)
    stats = compute_stats(df)

    print("Detected sweep values:",
          sorted(stats["sweep"].unique()))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for dataset in stats["dataset"].unique():
        out = Path(args.out_dir)/f"{dataset}_{args.sweep}_regions.pdf"
        plot_dataset(stats, dataset,
                     args.sweep, out)

if __name__ == "__main__":
    main()
