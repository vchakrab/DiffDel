#!/usr/bin/env python3
"""
collect_main_data_statistics.py

Collects statistics from:

    main_data/
        exp/
        gumbel/
        min/

Assumes:
    - exp & gumbel already filtered to epsilon_m = 0.1 and L0 = 0.1
    - min contains all runs

Outputs:
    main_data_statistics.csv
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.getcwd()
MAIN_DIR = os.path.join(BASE_DIR)

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]


# ------------------------------------------------------------
# Metric Formulas
# ------------------------------------------------------------
#
# 1) Average Mask Size:
#       E[|M|] = mean(mask_size)
#
# 2) Mask Improvement (%):
#       100 * (E[|M_min|] - E[|M|]) / E[|M_min|]
#
# 3) Inferential Leakage:
#       E[L] = mean(leakage)
#
# 4) Leakage 95% CI:
#       E[L] ± 1.96 * (std(leakage) / sqrt(n))
#
# 5) Deletion Ratio:
#       E[ |M| / |I(c*) \ {c*}| ]
#       = mean(mask_size / num_instantiated_cells)
#
# 6) Efficiency:
#       η = E[ (1 - leakage) / (|M| + 1) ]
#
# 7) Time (ms):
#       E[T] = mean(total_time) * 1000
#
# 8) Memory (bytes):
#       E[mem] = mean(memory_overhead_bytes)
# ------------------------------------------------------------


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
        deletion_ratio, efficiency, time_ms, memory
    )

MIN_MASK = {"airport": 5, "hospital": 9, "adult": 9, "flight": 11, "tax": 3}
def main():
    rows = []

    # First compute Min baseline mask sizes
    min_masks = {}

    for d in DATASETS:
        path = os.path.join(MAIN_DIR, "min", d, "full_data.csv")
        df = pd.read_csv(path)

        m, l, l_margin, dr, eta, t, mem = compute_metrics(df)

        min_masks[d] = m

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
            "avg_memory_bytes": mem
        })

    # Then exp and gumbel
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
                "avg_memory_bytes": mem
            })

    out_df = pd.DataFrame(rows)
    out_file = "main_data_statistics.csv"
    out_df.to_csv(out_file, index=False)

    print("\nMain statistics collected successfully.")
    print(f"Saved to: {out_file}\n")
    print(out_df)


if __name__ == "__main__":
    main()