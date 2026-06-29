#!/usr/bin/env python3
"""
collect_table_data.py

Collects statistics from data/main_data/ after it is built by collect_data.build_main_data().

    data/main_data/
        exp/
        gum/
        min/

Outputs:
    main_data_statistics.csv
"""

import ast
import os
import re

import numpy as np
import pandas as pd

MAIN_DIR = os.path.join(os.path.dirname(__file__), "data", "main_data")

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]

MIN_MASK = {"airport": 5, "hospital": 9, "adult": 9, "flight": 11, "tax": 3}

DC_FILES = {
    "airport":  "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topAirportDCs_parsed.py",
    "hospital": "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topHospitalDCs_parsed.py",
    "tax":      "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topTaxDCs_parsed.py",
    "adult":    "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topAdultDCs_parsed.py",
    "flight":   "/Users/adhariya/DiffDel/DCandDelset/dc_configs/topFlightDCs_parsed.py",
}

TARGET_ATTR = {
    "airport":  "continent",
    "hospital": "ProviderNumber",
    "tax":      "city",
    "adult":    "education",
    "flight":   "FlightDate",
}


# ---------------------------------------------------------------------------
# rho1 helpers (from rho1_bfs.py)
# ---------------------------------------------------------------------------

def _extract_attr(predicate):
    """Strip the 't1.'/'t2.' prefix off a predicate's column reference."""
    return predicate[0].split(".")[1]


def _load_dcs(path):
    """Read a *_parsed.py file and return its denial_constraints list."""
    with open(path) as f:
        src = f.read()
    match = re.search(r"denial_constraints\s*=\s*(\[.*\])", src, re.DOTALL)
    return ast.literal_eval(match.group(1))


def _compute_rho1(dcs):
    """
    Build the attribute co-occurrence graph from the DC list and return
    rho1 = |E| / |V|, where:
      |V| = number of unique attributes appearing in any DC predicate
      |E| = number of unique unordered attribute pairs that co-occur
            in at least one DC
    """
    all_attrs = set()
    for dc in dcs:
        for pred in dc:
            all_attrs.add(_extract_attr(pred))

    edge_set = set()
    for dc in dcs:
        attrs_in_dc = list({_extract_attr(p) for p in dc})
        for i in range(len(attrs_in_dc)):
            for j in range(i + 1, len(attrs_in_dc)):
                a, b = sorted([attrs_in_dc[i], attrs_in_dc[j]])
                edge_set.add((a, b))

    V = len(all_attrs)
    E = len(edge_set)
    return E / V if V > 0 else 0.0


def compute_rho1_per_dataset():
    """Return a dict mapping dataset name -> rho1."""
    rho1_map = {}
    for name, path in DC_FILES.items():
        dcs = _load_dcs(path)
        rho1_map[name] = _compute_rho1(dcs)
    return rho1_map


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

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
    rho1_map = compute_rho1_per_dataset()

    rows = []

    for d in DATASETS:
        path = os.path.join(MAIN_DIR, "min", d, "full_data.csv")
        df = pd.read_csv(path)

        m, l, l_margin, dr, eta, t, mem = compute_metrics(df)

        rows.append({
            "method": "min",
            "dataset": d,
            "rho": rho1_map[d],
            "avg_mask_size": m,
            "mask_improvement_percent": 0.0,
            "avg_leakage": l,
            "leakage_ci_margin_95": l_margin,
            "avg_deletion_ratio": dr,
            "avg_efficiency": eta,
            "avg_time_ms": t,
            "avg_memory_bytes": mem,
        })

    for method in ["exp", "gum"]:
        for d in DATASETS:
            path = os.path.join(MAIN_DIR, method, d, "full_data.csv")
            df = pd.read_csv(path)

            m, l, l_margin, dr, eta, t, mem = compute_metrics(df)

            m_min = MIN_MASK[d]
            improvement = 100.0 * (m_min - m) / m_min

            rows.append({
                "method": method,
                "dataset": d,
                "rho": rho1_map[d],
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