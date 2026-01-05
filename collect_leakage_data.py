#!/usr/bin/env python3
import csv
import itertools
from typing import Set, List

# =========================
# USER SETTINGS
# =========================

DATASETS = [
    ("airport", "iso_country"),
    ("hospital", "EmergencyService"),
    ("tax", "marital_status"),
    ("adult", "education"),
    ("Onlineretail", "InvoiceNo"),
]

TAUS = [round(i / 10, 1) for i in range(1, 11)]  # 0.1 .. 1.0
RHO = 0.9

OUT_CSV = "leakage_trends_actuals.csv"

# =========================
# IMPORT YOUR CORE LOGIC
# =========================
from baseline_deletion_3 import (
    dc_to_rdrs_and_weights_strict,
    construct_local_hypergraph,
    compute_leakage_delexp,
)

from rtf_core.initialization_phase import InitializationManager


# =========================
# Helpers
# =========================

def filter_hypergraph_by_tau(H, tau: float):
    """Keep only hyperedges with weight < tau."""
    H2 = type(H)()
    H2.vertices = set(H.vertices)
    H2.edges = [(vs, w) for (vs, w) in H.edges if float(w) < tau]
    return H2


def enumerate_all_masks(zone: List[str]):
    for k in range(len(zone) + 1):
        for comb in itertools.combinations(zone, k):
            yield set(comb)


def is_edge_active(edge_verts: Set[str], mask: Set[str], target_cell: str) -> bool:
    """
    Same rule as your chain logic:
      - target_cell is always treated as masked
      - edge is ACTIVE iff it has < 2 masked vertices
    """
    masked = set(mask)
    masked.add(target_cell)

    cnt = 0
    for v in edge_verts:
        if v in masked:
            cnt += 1
            if cnt >= 2:
                return False
    return True


def count_active_rdrs(H, mask: Set[str], target_cell: str) -> int:
    return sum(
        1
        for verts, _ in H.edges
        if is_edge_active(set(verts), mask, target_cell)
    )


# =========================
# Main
# =========================

def main():
    rows = []

    for dataset, target_attr in DATASETS:
        print(f"\n=== {dataset} / target={target_attr} ===")

        # ---- Initialization phase (DC loading only)
        init_mgr = InitializationManager(
            {"key": 500, "attribute": target_attr},
            dataset,
            threshold=0,
        )
        init_mgr.initialize()

        # ---- Build hypergraph
        rdrs, rdr_weights = dc_to_rdrs_and_weights_strict(init_mgr)
        H_base = construct_local_hypergraph(target_attr, rdrs, rdr_weights)

        inference_zone = sorted(H_base.vertices - {target_attr})
        zone_size = len(inference_zone)

        print(f"|I(c*)| = {zone_size}  -> masks = {2 ** zone_size}")

        for tau in TAUS:
            H_tau = filter_hypergraph_by_tau(H_base, tau)
            num_edges_tau = len(H_tau.edges)

            print(f"  tau={tau:.1f}  |E_tau|={num_edges_tau}")

            for mask in enumerate_all_masks(inference_zone):
                mask_size = len(mask)

                leakage = float(
                    compute_leakage_delexp(
                        mask=mask,
                        target_cell=target_attr,
                        hypergraph=H_tau,
                        rho=RHO,
                    )
                )

                active_rdrs = count_active_rdrs(H_tau, mask, target_attr)

                rows.append({
                    "dataset": dataset,
                    "target_attr": target_attr,
                    "tau": tau,
                    "zone_size": zone_size,
                    "num_edges_after_tau": num_edges_tau,
                    "mask": "{" + ",".join(sorted(mask)) + "}",
                    "mask_size": mask_size,
                    "leakage": leakage,
                    "active_rdrs": active_rdrs,
                })

    # ---- Write CSV
    fieldnames = [
        "dataset",
        "target_attr",
        "tau",
        "zone_size",
        "num_edges_after_tau",
        "mask",
        "mask_size",
        "leakage",
        "active_rdrs",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
