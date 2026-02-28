#!/usr/bin/env python3
"""
compute_main_statistics.py

Adds:
    - Average paths
    - 95% confidence interval for leakage
"""

import os
import pandas as pd
import numpy as np

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]

EPSILON_M = 0.1
L0 = 0.4


# ------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------
def compute_metrics(df):
    n = len(df)

    # Core metrics
    mask_mean = df["mask_size"].mean()
    leakage_mean = df["leakage"].mean()

    deletion_ratio = (df["mask_size"] / df["num_instantiated_cells"]).mean()
    efficiency = ((1 - df["leakage"]) / (df["mask_size"] + 1)).mean()

    time_ms = df["total_time"].mean() * 1000
    memory = df["memory_overhead_bytes"].mean()

    # NEW: average paths
    avg_paths = df["total_paths"].mean() if "total_paths" in df.columns else np.nan

    # ------------------------------------------------------------
    # 95% Confidence Interval for Leakage
    # CI = mean ± 1.96 * (std / sqrt(n))
    # ------------------------------------------------------------
    if n > 1:
        leakage_std = df["leakage"].std(ddof=1)
        margin = 1.96 * leakage_std / np.sqrt(n)
        ci_lower = leakage_mean - margin
        ci_upper = leakage_mean + margin
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    return (
        mask_mean,
        leakage_mean,
        ci_lower,
        ci_upper,
        deletion_ratio,
        efficiency,
        time_ms,
        memory,
        avg_paths,
    )


def load_exp_or_gumbel(method, dataset):
    path = os.path.join(method, dataset, "full_data.csv")
    df = pd.read_csv(path)
    df = df[(df["epsilon_m"] == EPSILON_M) & (df["L0"] == L0)]
    return df


def load_min(dataset):
    path = os.path.join("min", dataset, "full_data.csv")
    return pd.read_csv(path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    rows = []

    # Min baseline
    min_baseline_masks = {}

    for d in DATASETS:
        df_min = load_min(d)

        (
            m, l, ci_lo, ci_hi,
            dr, eta, t, mem, paths
        ) = compute_metrics(df_min)

        min_baseline_masks[d] = m

        rows.append({
            "method": "min",
            "dataset": d,
            "avg_mask_size": m,
            "mask_improvement_percent": 0.0,
            "avg_leakage": l,
            "leakage_ci_lower": ci_lo,
            "leakage_ci_upper": ci_hi,
            "avg_deletion_ratio": dr,
            "avg_efficiency": eta,
            "avg_time_ms": t,
            "avg_memory_bytes": mem,
            "avg_paths": paths
        })

    # Exp + Gumbel
    for method in ["exp", "gumbel"]:
        for d in DATASETS:
            df = load_exp_or_gumbel(method, d)

            (
                m, l, ci_lo, ci_hi,
                dr, eta, t, mem, paths
            ) = compute_metrics(df)

            m_min = min_baseline_masks[d]
            improvement = 100.0 * (m_min - m) / m_min

            rows.append({
                "method": method,
                "dataset": d,
                "avg_mask_size": m,
                "mask_improvement_percent": improvement,
                "avg_leakage": l,
                "leakage_ci_lower": ci_lo,
                "leakage_ci_upper": ci_hi,
                "avg_deletion_ratio": dr,
                "avg_efficiency": eta,
                "avg_time_ms": t,
                "avg_memory_bytes": mem,
                "avg_paths": paths
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("main_statistics.csv", index=False)

    print("\nMain statistics written to main_statistics.csv\n")
    print(out_df)


if __name__ == "__main__":
    main()