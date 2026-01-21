#!/usr/bin/env python3
"""
export_all_masks_all_datasets_streaming.py

Writes EVERY mask in the FULL powerset of I(c*) for EACH dataset, with leakage.

Key properties:
  ✅ Does NOT skip datasets
  ✅ Does NOT skip masks
  ✅ Streams (does not build candidate_masks list in memory)
  ✅ Writes rows incrementally to CSV (safe for huge outputs)

WARNING:
  Enumerating all masks is 2^(|I(c*)|). If zone is big, this will run "forever" and produce
  a massive CSV. That's inherent.

Requires your repo modules:
  - leakage.py
  - DCandDelset.dc_configs.top*DCs_parsed
  - weights.weights_corrected.<dataset>_weights (loaded indirectly by leakage pipeline)
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
    "airport": "home_link",
    "hospital": "ProviderNumber",
    "adult": "education",
    "flight": "FlightDate",
    "tax": "city",
}

HYPERGRAPH_MODE = "MAX"         # "MAX" or "ACTUAL"
LEAKAGE_METHOD = "greedy_disjoint"
TAU = None

# Optional utility columns (leakage independent)
LAM = 0.75
L0 = 0.25

OUTPUT_CSV = "all_masks_all_datasets_with_leakage.csv"

# Print progress every N masks
PROGRESS_EVERY = 10000

# =============================================================================
# Path robustness: add repo root + local dir to sys.path
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
# Imports from your leakage.py
# =============================================================================

try:
    from leakage import (  # type: ignore
        dc_to_rdrs_and_weights,
        construct_hypergraph_max,
        construct_hypergraph_actual,
        leakage as leakage_model,
        compute_utility_max,
        Hypergraph,
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import leakage.py. Put this script in your repo (or where leakage.py is importable).\n"
        f"sys.path[0:6]={sys.path[0:6]}\n"
        f"Import error: {e}"
    )

# =============================================================================
# Helpers
# =============================================================================

def load_parsed_dcs_for_dataset(dataset: str) -> List[List[Tuple[str, str, str]]]:
    """
    Loads:
      DCandDelset.dc_configs.top{Dataset}DCs_parsed.denial_constraints
    """
    ds = str(dataset).lower()
    mod_name = "NCVoter" if ds == "ncvoter" else ds.capitalize()
    dc_module_path = f"DCandDelset.dc_configs.top{mod_name}DCs_parsed"
    module = __import__(dc_module_path, fromlist=["denial_constraints"])
    dcs = getattr(module, "denial_constraints", None)
    if not dcs:
        raise RuntimeError(f"Parsed DCs missing/empty for dataset='{dataset}' module='{dc_module_path}'")
    return dcs

def build_hypergraph(dataset: str, target: str) -> "Hypergraph":
    raw_dcs = load_parsed_dcs_for_dataset(dataset)

    class _Init:
        def __init__(self, dataset: str, denial_constraints):
            self.dataset = dataset
            self.denial_constraints = denial_constraints

    init_manager = _Init(dataset, raw_dcs)
    rdrs, rdr_weights = dc_to_rdrs_and_weights(init_manager)

    if str(HYPERGRAPH_MODE).upper() == "ACTUAL":
        return construct_hypergraph_actual(target, rdrs, rdr_weights)
    return construct_hypergraph_max(target, rdrs, rdr_weights)

def inference_zone(hg: "Hypergraph", target: str) -> List[str]:
    """
    I(c*) = union of vertices in edges incident to target, excluding target itself.
    """
    zone: Set[str] = set()
    for (edge_verts, _w) in getattr(hg, "edges", []):
        for v in edge_verts:
            if v != target:
                zone.add(v)
    return sorted(zone)

def iter_powerset(zone: List[str]) -> Iterable[Tuple[str, ...]]:
    """
    Streaming powerset generator.
    """
    n = len(zone)
    for r in range(n + 1):
        for comb in combinations(zone, r):
            yield comb

def mask_to_str(mask_tuple: Tuple[str, ...]) -> str:
    # zone is sorted, combinations preserve order => already stable
    return "|".join(mask_tuple)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    out_path = os.path.abspath(OUTPUT_CSV)
    print(f"[INFO] Output CSV: {out_path}")
    print(f"[INFO] HYPERGRAPH_MODE={HYPERGRAPH_MODE}  LEAKAGE_METHOD={LEAKAGE_METHOD}  TAU={TAU}")

    fieldnames = [
        "dataset",
        "target",
        "hypergraph_mode",
        "leakage_method",
        "tau",
        "mask",
        "mask_size",
        "inference_zone_size",
        "leakage",
        "utility",
    ]

    total_rows = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ds in DATASETS:
            ds = ds.lower()
            if ds not in TARGET_ATTR:
                raise RuntimeError(f"Missing TARGET_ATTR for dataset '{ds}' (you said don't skip anything).")

            target = TARGET_ATTR[ds]
            print(f"\n[INFO] Dataset={ds}  Target={target}")

            hg = build_hypergraph(ds, target)
            zone = inference_zone(hg, target)
            zone_size = len(zone)

            print(f"[INFO] inference_zone_size |I(c*)| = {zone_size}")
            expected = None
            if zone_size < 63:
                expected = 1 << zone_size
                print(f"[INFO] total masks to enumerate = {expected:,}")

            ds_rows = 0
            t0 = time.time()

            for mask_tuple in iter_powerset(zone):
                mask_set = set(mask_tuple)

                L = leakage_model(
                    mask=mask_set,
                    target_cell=target,
                    hypergraph=hg,
                    tau=TAU,
                    leakage_method=LEAKAGE_METHOD,
                )
                Lf = float(L)

                # Utility is optional; safe if zone_size=0
                if zone_size == 0:
                    Uf = 0.0
                else:
                    Uf = float(
                        compute_utility_max(
                            leakage=Lf,
                            mask_size=len(mask_tuple),
                            lam=float(LAM),
                            zone_size=zone_size,
                            L0=float(L0),
                        )
                    )

                writer.writerow(
                    {
                        "dataset": ds,
                        "target": target,
                        "hypergraph_mode": str(HYPERGRAPH_MODE).upper(),
                        "leakage_method": str(LEAKAGE_METHOD),
                        "tau": "" if TAU is None else float(TAU),
                        "mask": mask_to_str(mask_tuple),
                        "mask_size": int(len(mask_tuple)),
                        "inference_zone_size": int(zone_size),
                        "leakage": Lf,
                        "utility": Uf,
                    }
                )

                ds_rows += 1
                total_rows += 1

                if PROGRESS_EVERY and (ds_rows % PROGRESS_EVERY == 0):
                    dt = time.time() - t0
                    rate = ds_rows / max(1e-9, dt)
                    msg = f"[PROGRESS] {ds}: wrote {ds_rows:,} rows"
                    if expected is not None:
                        msg += f" / {expected:,}"
                    msg += f"  ({rate:,.1f} rows/s)"
                    print(msg)
                    f.flush()  # keep file safe

            dt = time.time() - t0
            print(f"[DONE] {ds}: wrote {ds_rows:,} rows in {dt:,.1f}s")

    print(f"\n[ALL DONE] Total rows written: {total_rows:,}")
    print(f"[ALL DONE] CSV: {out_path}")


if __name__ == "__main__":
    main()
