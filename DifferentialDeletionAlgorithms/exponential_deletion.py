#!/usr/bin/env python3
"""
exponential_deletion.py - Hypergraph-Based Implementation (Schema-level)

REFactor request:
- REMOVE local inference-chain / leakage / BFS / graph-related code
- IMPORT the single-source-of-truth leakage implementation from your `leakage.py`
  (the one that already has: Hypergraph, construct_hypergraph_max/actual, leakage(...))

Assumptions:
- You have a module `leakage.py` on PYTHONPATH that defines:
    - get_dataset_weights
    - Hypergraph
    - construct_hypergraph_max
    - construct_hypergraph_actual
    - leakage(mask, target_cell, hypergraph, *, tau=None, return_counts=False, leakage_method="noisy_or")
- You have `rtf_core.initialization_phase.InitializationManager`
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from itertools import chain, combinations

import numpy as np

from rtf_core import initialization_phase

# ✅ Import leakage + hypergraph construction from your single source of truth
from leakage import (
    get_dataset_weights,
    Hypergraph,
    construct_hypergraph_max,
    construct_hypergraph_actual,
    leakage as compute_leakage,  # rename for this file
    compute_utility,
    map_dc_to_weight
)


# ============================================================
# Utilities
# ============================================================

def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def compute_possible_mask_set_str(target_cell: str, hypergraph: Hypergraph) -> List[Set[str]]:
    """
    Candidate masks are the powerset of the inference zone:
      I(c*) = V(H_max) \ {c*}
    """
    inference_zone = hypergraph.vertices - {target_cell}
    return [set(s) for s in powerset(sorted(inference_zone))]


def exponential_mechanism_sample(
    candidates: List[Set[str]],
    *,
    target_cell: str,
    hypergraph: Hypergraph,
    epsilon: float,
    lam: float,
    rho: float = 0.9,
    tau: Optional[float] = None,
    leakage_method: str = "noisy_or",  # or "greedy_disjoint" if your leakage.py supports it
) -> Tuple[Set[str], float, float]:
    """
    Sample a mask using exponential mechanism.

    Returns: (mask, utility, leakage)
    """
    zone_size = len((hypergraph.vertices - {target_cell}))
    if zone_size <= 0:
        zone_size = 1

    utilities = np.empty(len(candidates), dtype=float)
    leakages = np.empty(len(candidates), dtype=float)

    for i, M in enumerate(candidates):
        # ✅ use imported leakage implementation
        L = compute_leakage(
            M,
            target_cell,
            hypergraph,
            tau=tau,
            leakage_method=leakage_method,
            return_counts=False,
        )

        # If your leakage.py enforces rho-safe internally, keep rho there.
        # If rho-safe is handled outside, you can clamp here. Most of your
        # codebase does rho-safe inside compute_leakage, so we do nothing here.
        leakages[i] = float(L)
        utilities[i] = compute_utility(
            leakage=float(L),
            mask_size=len(M),
            lam=float(lam),
            zone_size=int(zone_size),
        )

    # Exponential mechanism probabilities
    # NOTE: Utility sensitivity handling is project-specific; keeping your prior style:
    scores = (float(epsilon) * utilities) / (2.0 * max(1e-10, float(lam)))
    max_score = float(np.max(scores)) if len(scores) else 0.0
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / np.sum(exp_scores)

    idx = int(np.random.choice(len(candidates), p=probs))
    return candidates[idx], float(utilities[idx]), float(leakages[idx])


def estimate_paths_proxy_from_channels(
    *,
    num_channel_edges: int,
    L_empty: float,
    L_mask: float
) -> Dict[str, int]:
    """
    Proxy for paths blocked based on leakage reduction.
    """
    total = int(max(0, num_channel_edges))
    if total == 0 or not np.isfinite(L_empty) or L_empty <= 1e-15 or not np.isfinite(L_mask):
        return {"num_paths_est": total, "paths_blocked_est": 0}

    frac = 1.0 - (float(L_mask) / float(L_empty))
    frac = float(max(0.0, min(1.0, frac)))
    blocked = int(round(frac * total))
    blocked = int(max(0, min(total, blocked)))
    return {"num_paths_est": total, "paths_blocked_est": blocked}


def estimate_memory_overhead_bytes_delexp(
    *,
    hypergraph: Hypergraph,
    mask_size: int,
    num_candidate_masks: int,
    candidate_mask_members: int,
    includes_channel_map: bool = True,
) -> int:
    """Memory overhead estimation for the exponential mechanism (rough)."""
    num_vertices = len(hypergraph.vertices)
    num_edges = len(hypergraph.edges)
    edge_members = sum(len(vertices) for vertices, _ in hypergraph.edges)

    BYTES_PER_VERTEX = 112
    BYTES_PER_EDGE = 184
    BYTES_PER_EDGE_MEMBER = 72
    BYTES_PER_MASK_SET = 96
    BYTES_PER_MASK_MEMBER = 72
    BYTES_PER_FLOAT = 8
    BYTES_PER_INT = 28
    BYTES_PER_CAND_MASK = 96

    est = 0
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER
    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    if includes_channel_map:
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    est += num_candidate_masks * BYTES_PER_CAND_MASK
    est += candidate_mask_members * BYTES_PER_MASK_MEMBER
    est += num_candidate_masks * 8  # utilities array

    return int(est)


# ============================================================
# DC -> RDR hyperedges (schema-level)
# ============================================================

def dc_to_hyperedges(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    """
    Convert denial constraints into RDRs (tuples of attributes) + aligned weights.
    """
    rdrs: List[Tuple[str, ...]] = []
    rdr_weights: List[float] = []

    dataset_weights = get_dataset_weights(init_manager.dataset)

    for dc in getattr(init_manager, "denial_constraints", []) or []:
        attrs: Set[str] = set()
        weight = map_dc_to_weight(init_manager, dc, dataset_weights)

        for pred in dc:
            if isinstance(pred, (list, tuple)) and len(pred) >= 1:
                tok0 = pred[0]
                if isinstance(tok0, str) and "." in tok0:
                    attrs.add(tok0.split(".")[-1])

            if isinstance(pred, (list, tuple)) and len(pred) >= 3:
                tok2 = pred[2]
                if isinstance(tok2, str) and "." in tok2:
                    attrs.add(tok2.split(".")[-1])

        if len(attrs) >= 2:
            rdrs.append(tuple(sorted(attrs)))
            rdr_weights.append(float(weight))

    return rdrs, rdr_weights


# ============================================================
# Main orchestrator
# ============================================================

def exponential_deletion_main(
    dataset: str,
    key: int,
    target_cell: str,
    *,
    epsilon: float = 10.0,
    lam: float = 0.67,
    rho: float = 0.9,
    tau: Optional[float] = None,
    leakage_method: str = "noisy_or",  # or "greedy_disjoint"
) -> Dict[str, Any]:
    """
    Hypergraph-based exponential deletion mechanism, schema-level.

    Uses:
      - construct_hypergraph_max/actual from leakage.py
      - compute_leakage (imported) from leakage.py
    """
    # ----------------------
    # INIT
    # ----------------------
    init_start = time.time()

    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target_cell},
        dataset,
        0,
    )

    rdrs, rdr_weights = dc_to_hyperedges(init_manager)

    instantiated_cells: Set[str] = set()
    for rdr in rdrs:
        instantiated_cells.update(rdr)
    num_instantiated_cells = len(instantiated_cells)

    init_time = time.time() - init_start

    # ----------------------
    # HYPERGRAPH CONSTRUCTION (Algorithm 1)
    # ----------------------
    model_start = time.time()

    H_max = construct_hypergraph_max(target_cell, rdrs, rdr_weights)
    H_actual = construct_hypergraph_actual(target_cell, rdrs, rdr_weights)

    inference_zone = H_max.vertices - {target_cell}
    candidates = compute_possible_mask_set_str(target_cell, H_max)
    if not candidates:
        candidates = [set()]

    # ----------------------
    # Exponential mechanism
    # ----------------------
    final_mask, util_val, _L_selected_quick = exponential_mechanism_sample(
        candidates,
        target_cell=target_cell,
        hypergraph=H_actual,
        epsilon=epsilon,
        lam=lam,
        rho=rho,
        tau=tau,
        leakage_method=leakage_method,
    )

    # ----------------------
    # Leakage baseline + final
    # ----------------------
    leakage_base = compute_leakage(
        set(),
        target_cell,
        H_actual,
        tau=tau,
        leakage_method=leakage_method,
        return_counts=False,
    )

    leakage_final = compute_leakage(
        final_mask,
        target_cell,
        H_actual,
        tau=tau,
        leakage_method=leakage_method,
        return_counts=False,
    )

    # Count channel edges (edges containing target)
    num_channel_edges = sum(1 for vertices, _ in H_actual.edges if target_cell in vertices)

    paths_proxy = estimate_paths_proxy_from_channels(
        num_channel_edges=num_channel_edges,
        L_empty=float(leakage_base),
        L_mask=float(leakage_final),
    )

    model_time = time.time() - model_start

    # ----------------------
    # Memory estimate
    # ----------------------
    cand_members = sum(len(s) for s in candidates)
    memory_overhead = estimate_memory_overhead_bytes_delexp(
        hypergraph=H_actual,
        mask_size=len(final_mask),
        num_candidate_masks=len(candidates),
        candidate_mask_members=cand_members,
        includes_channel_map=True,
    )

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": 0.0,

        "dataset": str(dataset),
        "target_cell": str(target_cell),

        "leakage": float(leakage_final),
        "baseline_leakage_empty_mask": float(leakage_base),
        "utility": float(util_val),

        "mask_size": int(len(final_mask)),
        "mask": set(final_mask),

        "num_paths": int(paths_proxy["num_paths_est"]),
        "paths_blocked": int(paths_proxy["paths_blocked_est"]),

        "memory_overhead_bytes": int(memory_overhead),
        "num_instantiated_cells": int(num_instantiated_cells),
        "num_channel_edges": int(num_channel_edges),

        "num_rdrs": int(len(rdrs)),
        "tau": None if tau is None else float(tau),
        "rho": float(rho),
        "leakage_method": str(leakage_method),
    }


if __name__ == "__main__":
    # Example (adjust to your dataset/target):
    out = exponential_deletion_main("hospital", key=500, target_cell="ProviderNumber", epsilon=10, lam=0.67, rho=0.9)
    print(out)
    pass

#change paths to inference chains rather
