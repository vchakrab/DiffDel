#!/usr/bin/env python3
"""
export_all_masks_all_leakage_methods.py

Writes EVERY mask in the FULL powerset of I(c*) for EACH dataset,
with leakage computed under ALL THREE methods per row:
  - noisy_or
  - greedy_disjoint
  - max_leakage

Each row = one (dataset, target, mask) triple with 3 leakage columns + 3 utility columns.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

# =============================================================================
# CONFIG (edit these)
# =============================================================================

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]

TARGET_ATTR: Dict[str, str] = {
    "airport": "continent",
    "hospital": "ProviderNumber",
    "tax": "city",
    "adult": "education",
    "flight": "FlightDate",
}

HYPERGRAPH_MODE = "MAX"         # "MAX" or "ACTUAL"
TAU = None

LEAKAGE_METHODS = ["noisy_or", "greedy_disjoint", "max_leakage"]

# Utility params
LAM = 1000
L0 = 0.2

OUTPUT_CSV = "data/all_masks_all_methods.csv"
PROGRESS_EVERY = 10000

# =============================================================================
# Path robustness
# =============================================================================

def _find_repo_root(start: str) -> Optional[str]:
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, "DCandDelset")) or os.path.isdir(os.path.join(cur, "weights")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _find_repo_root(HERE) or _find_repo_root(os.getcwd())

if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO and REPO not in sys.path:
    sys.path.insert(0, REPO)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# =============================================================================
# Imports
# =============================================================================

try:
    from leakage import (  # type: ignore
        dc_to_rdrs_and_weights,
        construct_hypergraph_max,
        construct_hypergraph_actual,
        leakage as leakage_fn,
        max_leakage as max_leakage_fn,
        Hypergraph,
        compute_utility_em,
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import leakage.py.\n"
        f"sys.path[0:6]={sys.path[0:6]}\n"
        f"Import error: {e}"
    )

# =============================================================================
# Helpers
# =============================================================================

def load_parsed_dcs_for_dataset(dataset: str):
    ds = str(dataset).lower()
    mod_name = "NCVoter" if ds == "ncvoter" else ds.capitalize()
    dc_module_path = f"DCandDelset.dc_configs.top{mod_name}DCs_parsed"
    module = __import__(dc_module_path, fromlist=["denial_constraints"])
    dcs = getattr(module, "denial_constraints", None)
    if not dcs:
        raise RuntimeError(f"Parsed DCs missing/empty for dataset='{dataset}'")
    return dcs

def build_hypergraph(dataset: str, target: str) -> "Hypergraph":
    raw_dcs = load_parsed_dcs_for_dataset(dataset)

    class _Init:
        def __init__(self, dataset, denial_constraints):
            self.dataset = dataset
            self.denial_constraints = denial_constraints

    init_manager = _Init(dataset, raw_dcs)
    rdrs, rdr_weights = dc_to_rdrs_and_weights(init_manager)

    if str(HYPERGRAPH_MODE).upper() == "ACTUAL":
        return construct_hypergraph_actual(target, rdrs, rdr_weights)
    return construct_hypergraph_max(target, rdrs, rdr_weights)

def inference_zone(hg: "Hypergraph", target: str) -> List[str]:
    zone: Set[str] = set()
    for (edge_verts, _w) in getattr(hg, "edges", []):
        for v in edge_verts:
            if v != target:
                zone.add(v)
    return sorted(zone)

def iter_powerset(zone: List[str]) -> Iterable[Tuple[str, ...]]:
    for r in range(len(zone) + 1):
        for comb in combinations(zone, r):
            yield comb

def mask_to_str(mask_tuple: Tuple[str, ...]) -> str:
    return "|".join(mask_tuple)

def compute_leakage_all_methods(
    mask_set: Set[str],
    target: str,
    hg: "Hypergraph",
) -> Dict[str, float]:
    """
    Returns leakage for all 3 methods in one dict.
    max_leakage is called directly (it's not routed through leakage_fn).
    """
    results = {}

    for method in ("noisy_or", "greedy_disjoint"):
        results[method] = float(
            leakage_fn(
                mask=mask_set,
                target_cell=target,
                hypergraph=hg,
                tau=TAU,
                leakage_method=method,
            )
        )

    results["max_leakage"] = float(
        max_leakage_fn(
            mask=mask_set,
            target_cell=target,
            hypergraph=hg,
            tau=TAU,
        )
    )

    return results

def compute_utility(leakage: float, mask_size: int, zone_size: int) -> float:
    if zone_size == 0:
        return 0.0
    return float(
        compute_utility_em(
            leakage=leakage,
            mask_size=mask_size,
            lambda_penalty=float(LAM),
            zone_size=zone_size,
            L0=float(L0),
        )
    )

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    out_path = os.path.abspath(OUTPUT_CSV)
    # print(f"[INFO] Output CSV: {out_path}")
    # print(f"[INFO] HYPERGRAPH_MODE={HYPERGRAPH_MODE}  TAU={TAU}")
    # print(f"[INFO] Methods: {LEAKAGE_METHODS}")

    fieldnames = [
        "dataset",
        "target",
        "hypergraph_mode",
        "tau",
        "mask",
        "mask_size",
        "inference_zone_size",
        # leakage per method
        "leakage_noisy_or",
        "leakage_greedy_disjoint",
        "leakage_max",
        # utility per method
        "utility_noisy_or",
        "utility_greedy_disjoint",
        "utility_max",
        # sanity check: bounds hold?
        "bounds_ok",          # max <= greedy <= noisy_or
    ]

    total_rows = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ds in DATASETS:
            ds = ds.lower()
            if ds not in TARGET_ATTR:
                raise RuntimeError(f"Missing TARGET_ATTR for dataset '{ds}'.")

            target = TARGET_ATTR[ds]
            # print(f"\n[INFO] Dataset={ds}  Target={target}")

            hg = build_hypergraph(ds, target)
            zone = inference_zone(hg, target)
            zone_size = len(zone)

            # print(f"[INFO] |I(c*)| = {zone_size}")
            if zone_size < 63:
                pass
                # print(f"[INFO] total masks = {1 << zone_size:,}")

            ds_rows = 0
            t0 = time.time()

            for mask_tuple in iter_powerset(zone):
                mask_set = set(mask_tuple)
                mask_size = len(mask_tuple)

                L = compute_leakage_all_methods(mask_set, target, hg)

                L_nor = L["noisy_or"]
                L_gd  = L["greedy_disjoint"]
                L_max = L["max_leakage"]

                U_nor = compute_utility(L_nor, mask_size, zone_size)
                U_gd  = compute_utility(L_gd,  mask_size, zone_size)
                U_max = compute_utility(L_max, mask_size, zone_size)

                # eq (7): max <= greedy <= noisy_or  (greedy is between bounds)
                bounds_ok = (L_max <= L_gd + 1e-9) and (L_gd <= L_nor + 1e-9)

                writer.writerow({
                    "dataset":              ds,
                    "target":               target,
                    "hypergraph_mode":      HYPERGRAPH_MODE,
                    "tau":                  "" if TAU is None else float(TAU),
                    "mask":                 mask_to_str(mask_tuple),
                    "mask_size":            mask_size,
                    "inference_zone_size":  zone_size,
                    "leakage_noisy_or":     L_nor,
                    "leakage_greedy_disjoint": L_gd,
                    "leakage_max":          L_max,
                    "utility_noisy_or":     U_nor,
                    "utility_greedy_disjoint": U_gd,
                    "utility_max":          U_max,
                    "bounds_ok":            bounds_ok,
                })

                ds_rows += 1
                total_rows += 1

                if PROGRESS_EVERY and (ds_rows % PROGRESS_EVERY == 0):
                    dt = time.time() - t0
                    rate = ds_rows / max(1e-9, dt)
                    # print(f"[PROGRESS] {ds}: {ds_rows:,} rows  ({rate:,.1f} rows/s)")
                    f.flush()

            dt = time.time() - t0
            # print(f"[DONE] {ds}: {ds_rows:,} rows in {dt:,.1f}s")

    # print(f"\n[ALL DONE] Total rows: {total_rows:,}")
    # print(f"[ALL DONE] CSV: {out_path}")


if __name__ == "__main__":
    main()
