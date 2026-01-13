#!/usr/bin/env python3
"""
lambda_ablation_per_dataset_ci.py

Makes 3 figures (one per metric):
  - leakage vs lambda
  - utility vs lambda
  - mask_size vs lambda

Layout: 1 row x 5 columns (airport, hospital, adult, flight, tax)

In each subplot:
  - two method curves: del2ph (solid) vs delgum (dashed)
  - 95% CI band around each curve (computed from the 100 rows in that dataset block)

Assumes files exist exactly at:
  lambda_ablation/edel2ph_1.csv ... edel2ph_10.csv
  lambda_ablation/edelgum_1.csv  ... edelgum_10.csv

Each CSV:
  - columns: lambda, leakage, utility, mask_size
  - 500 rows arranged as 5 dataset blocks of 100 rows each in order:
      airport, hospital, adult, flight, tax
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = "lambda_ablation"
METHODS = ["del2ph", "delgum"]
LAMBDA_IDX = list(range(1, 11))  # 1..10 => lambda = i/10 (also stored in file)
DATASETS = ["airport", "hospital", "adult", "flight", "tax"]
ROWS_PER_DATASET = 100

METRICS = ["leakage", "utility", "mask_size"]

OUTFILES = {
    "leakage": "lambda_vs_leakage_per_dataset_ci.png",
    "utility": "lambda_vs_utility_per_dataset_ci.png",
    "mask_size": "lambda_vs_mask_size_per_dataset_ci.png",
}

# 95% CI via normal approx
Z_95 = 1.96


# ----------------------------
# HELPERS
# ----------------------------
def csv_path(method: str, i: int) -> str:
    return os.path.join(BASE_DIR, f"{method}_{i}.csv")


def assign_dataset_blocks(df: pd.DataFrame) -> pd.DataFrame:
    expected_rows = ROWS_PER_DATASET * len(DATASETS)
    print(df)
    print(expected_rows)
    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows ({len(DATASETS)}*{ROWS_PER_DATASET}), got {len(df)}"
        )
    dataset_col: List[str] = []
    for d in DATASETS:
        dataset_col.extend([d] * ROWS_PER_DATASET)
    out = df.copy()
    out["dataset"] = dataset_col
    return out


@dataclass
class SummaryPoint:
    mean: float
    lo: float
    hi: float


def mean_ci(values: np.ndarray, z: float = Z_95) -> SummaryPoint:
    """
    95% CI for mean: mean ± z * (std/sqrt(n))
    """
    v = np.asarray(values, dtype=float)
    n = int(v.size)
    if n <= 1:
        m = float(np.nanmean(v)) if n == 1 else float("nan")
        return SummaryPoint(m, m, m)

    m = float(np.mean(v))
    s = float(np.std(v, ddof=1))
    sem = s / np.sqrt(n)
    half = z * sem
    return SummaryPoint(m, m - half, m + half)


def load_summaries() -> pd.DataFrame:
    """
    Returns long-form summary rows:
      method, dataset, lambda, metric, mean, lo, hi
    """
    rows = []
    required_cols = {"lambda", "leakage", "utility", "mask_size"}

    for method in METHODS:
        for i in LAMBDA_IDX:
            path = csv_path(method, i)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            df = pd.read_csv(path)
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"{path} missing columns: {missing} (found {list(df.columns)})")

            df = assign_dataset_blocks(df)

            # Each file should have a single lambda value; but we trust the column.
            # Compute mean+CI within each dataset block (100 rows).
            for dataset in DATASETS:
                block = df[df["dataset"] == dataset]
                lam = float(block["lambda"].iloc[0])

                for metric in METRICS:
                    sp = mean_ci(block[metric].to_numpy())
                    rows.append(
                        {
                            "method": method,
                            "dataset": dataset,
                            "lambda": lam,
                            "metric": metric,
                            "mean": sp.mean,
                            "lo": sp.lo,
                            "hi": sp.hi,
                        }
                    )

    out = pd.DataFrame(rows)
    out = out.sort_values(["metric", "dataset", "method", "lambda"]).reset_index(drop=True)
    return out


def plot_metric_grid(summ: pd.DataFrame, metric: str, outfile: str) -> None:
    """
    1x5 grid (datasets). In each subplot, plot del2ph and delgum curves with CI bands.
    """
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 3.8), sharey=True)

    if len(DATASETS) == 1:
        axes = [axes]  # type: ignore

    for ax, dataset in zip(axes, DATASETS):
        sub = summ[(summ["metric"] == metric) & (summ["dataset"] == dataset)]

        # Keep consistent plotting order
        for method, linestyle in [("del2ph", "-"), ("delgum", "--")]:
            s = sub[sub["method"] == method].sort_values("lambda")

            x = s["lambda"].to_numpy()
            y = s["mean"].to_numpy()
            lo = s["lo"].to_numpy()
            hi = s["hi"].to_numpy()

            ax.plot(x, y, linestyle=linestyle, marker="o", label=method)
            ax.fill_between(x, lo, hi, alpha=0.2)

        ax.set_title(dataset)
        ax.set_xlabel("lambda")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(metric)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle(f"{metric} vs lambda (per-dataset; 95% CI bands)", y=1.02)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summ = load_summaries()
    # Sanity print
    print("[OK] Loaded summary rows:", len(summ))
    print(summ.head(10))

    for metric in METRICS:
        plot_metric_grid(summ, metric, OUTFILES[metric])
        print("[WROTE]", OUTFILES[metric])


if __name__ == "__main__":
    main()
