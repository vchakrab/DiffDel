#!/usr/bin/env python3
import csv
import itertools
import time
from typing import Set, List, Tuple

# =========================
# USER SETTINGS
# =========================

DATASETS = [
    ("tax", "marital_status"),
]

TAUS = [1.0]  # 0.1 .. 1.0
RHO = 1.0

OUT_CSV = "leakage_trends_tax_tight.csv"

# Flush to disk every N rows (flush only; no fsync)
FLUSH_EVERY = 50

# =========================
# IMPORT YOUR CORE LOGIC
# =========================
from DifferentialDeletionAlgorithms.baseline_deletion_3 import (
    dc_to_rdrs_and_weights_strict,
    construct_local_hypergraph,
    compute_leakage_delexp,  # <-- will be updated to support return_counts=True
)

from rtf_core.initialization_phase import InitializationManager


# =========================
# Helpers
# =========================

def clone_hypergraph_with_edges(H, edges):
    """
    Create a shallow clone of a Hypergraph-like object.
    Assumes attributes: vertices (set), edges (list).
    """
    H2 = H.__class__()
    H2.vertices = set(getattr(H, "vertices", set()))
    H2.edges = list(edges)
    return H2


def filter_hypergraph_by_tau(H, tau: float):
    """Keep only hyperedges with weight <= tau."""
    tau_f = float(tau)
    # NOTE: verts in your Hypergraph are already sets; keep them as-is.
    kept = [(vs, w) for (vs, w) in H.edges if float(w) <= tau_f]
    return clone_hypergraph_with_edges(H, kept)


def enumerate_all_masks(zone: List[str]):
    for k in range(len(zone) + 1):
        for comb in itertools.combinations(zone, k):
            yield set(comb)


def is_edge_active_by_mask_rule(edge_verts: Set[str], mask: Set[str], target_cell: str) -> bool:
    """
    Your requested "blocked edge" rule:
      - treat target_cell as masked
      - edge is ACTIVE iff it contains < 2 masked vertices
    """
    # Avoid allocating a new set(mask) for every edge check:
    # We'll just check membership against mask plus target.
    cnt = 0
    for v in edge_verts:
        if v == target_cell or v in mask:
            cnt += 1
            if cnt >= 2:
                return False
    return True


def count_active_blocked_edges(H, mask: Set[str], target_cell: str) -> Tuple[int, int]:
    active = 0
    blocked = 0
    for verts, _w in H.edges:
        if is_edge_active_by_mask_rule(verts, mask, target_cell):
            active += 1
        else:
            blocked += 1
    return active, blocked


# =========================
# Main
# =========================

def main():
    fieldnames = [
        "dataset",
        "target_attr",
        "tau",
        "rho",
        "zone_size",
        "num_edges_after_tau",

        "mask",
        "mask_size",

        "leakage",

        "active_edges",
        "blocked_edges",

        "num_chains",
        "active_chains",
        "blocked_chains",
        "total_time"
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        rows_written = 0

        for dataset, target_attr in DATASETS:
            print(f"\n=== {dataset} / target={target_attr} ===")
            start_time = time.time()
            # ---- Initialization phase (DC loading only)
            init_mgr = InitializationManager(
                {"key": 500, "attribute": target_attr},
                dataset,
                threshold=0,
            )
            #init_mgr.initialize()

            # ---- Build hypergraph

            rdrs, rdr_weights = dc_to_rdrs_and_weights_strict(init_mgr)
            H_base = construct_local_hypergraph(target_attr, rdrs, rdr_weights)
            print(H_base.edges)
            inference_zone = sorted(H_base.vertices - {target_attr})
            zone_size = len(inference_zone)

            print(f"|I(c*)| = {zone_size}  -> masks = {2 ** zone_size}")

            for tau in TAUS:
                H_tau = filter_hypergraph_by_tau(H_base, tau)
                num_edges_tau = len(H_tau.edges)

                print(f"  tau={tau:.1f}  |E_tau|={num_edges_tau}")

                for mask in enumerate_all_masks(inference_zone):
                    start_time_mask = time.time()
                    mask_size = len(mask)

                    # Leakage + chain COUNTS only (no chain lists/weights returned)
                    L, num_chains, active_chains, blocked_chains = compute_leakage_delexp(
                        mask=mask,
                        target_cell=target_attr,
                        hypergraph=H_tau,
                        rho=RHO,
                        return_counts=True,
                        leakage_method = "greedy_disjoint"
                    )

                    # Edge activity stats (mask rule)
                    active_edges, blocked_edges = count_active_blocked_edges(H_tau, mask, target_attr)
                    end_time_mask = time.time() - start_time_mask
                    writer.writerow({
                        "dataset": dataset,
                        "target_attr": target_attr,
                        "tau": float(tau),
                        "rho": float(RHO),
                        "zone_size": zone_size,
                        "num_edges_after_tau": num_edges_tau,

                        "mask": "{" + ",".join(sorted(mask)) + "}",
                        "mask_size": mask_size,

                        "leakage": float(L),

                        "active_edges": active_edges,
                        "blocked_edges": blocked_edges,

                        "num_chains": int(num_chains),
                        "active_chains": int(active_chains),
                        "blocked_chains": int(blocked_chains),
                        "total_time": float(end_time_mask),
                    })
                    rows_written += 1
                    if FLUSH_EVERY and (rows_written % FLUSH_EVERY == 0):
                        f.flush()
            end_time = start_time - time.time()
            print(f"{dataset} elapsed time: {end_time}")
        f.flush()

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
