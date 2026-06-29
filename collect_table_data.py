#!/usr/bin/env python3
"""
collect_table_data.py

Collects statistics from data/main_data/ after it is built by collect_data.build_main_data().

    data/main_data/
        exp/
        gumbel/
        min/

Outputs:
    main_data_statistics.csv
"""

import os
import pandas as pd
import numpy as np

MAIN_DIR = os.path.join(os.path.dirname(__file__), "data", "main_data")

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]

MIN_MASK = {"airport": 5, "hospital": 9, "adult": 9, "flight": 11, "tax": 3}


def compute_metrics(df):
    mask_mean = df["mask_size"].mean()

    leakage_mean = df["leakage"].mean()
    leakage_std = df["leakage"].std(ddof=1)
    n = len(df)
    leakage_ci_margin = 1.96 * (leakage_std / np.sqrt(n))

    deletion_ratio = (df["mask_size"] / df["num_instantiated_cells"]).mean()
    efficiency = ((1 - df["leakage"]) / (df["mask_size"] + 1)).mean()
    time_ms = df["total_time"].mean() * 1000
    memory = df["memory_overhead_bytes"].mean()

    return (
        mask_mean,
        leakage_mean, leakage_ci_margin,
        deletion_ratio, efficiency, time_ms, memory,
    )


def main():
    rows = []

    for d in DATASETS:
        path = os.path.join(MAIN_DIR, "min", d, "full_data.csv")
        df = pd.read_csv(path)

        m, l, l_margin, dr, eta, t, mem = compute_metrics(df)

        rows.append({
            "method": "min",
            "dataset": d,
            "avg_mask_size": m,
            "mask_improvement_percent": 0.0,
            "avg_leakage": l,
            "leakage_ci_margin_95": l_margin,
            "avg_deletion_ratio": dr,
            "avg_efficiency": eta,
            "avg_time_ms": t,
            "avg_memory_bytes": mem,
        })

    for method in ["exp", "gumbel"]:
        for d in DATASETS:
            path = os.path.join(MAIN_DIR, method, d, "full_data.csv")
            df = pd.read_csv(path)

            m, l, l_margin, dr, eta, t, mem = compute_metrics(df)

            m_min = MIN_MASK[d]
            improvement = 100.0 * (m_min - m) / m_min

            rows.append({
                "method": method,
                "dataset": d,
                "avg_mask_size": m,
                "mask_improvement_percent": improvement,
                "avg_leakage": l,
                "leakage_ci_margin_95": l_margin,
                "avg_deletion_ratio": dr,
                "avg_efficiency": eta,
                "avg_time_ms": t,
                "avg_memory_bytes": mem,
            })

    out_df = pd.DataFrame(rows)
    out_file = os.path.join(os.path.dirname(__file__), "main_data_statistics.csv")
    out_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()