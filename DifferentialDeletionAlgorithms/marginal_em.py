#!/usr/bin/env python3
# marignal_em.py
"""
Algorithm 1: MarginalEM (2n leakage calls, ε-DP, responsive)

Build hypergraph exactly like greedy_gumbel.py:
  - load denial constraints module for dataset
  - load weights via leakage.get_dataset_weights
  - convert each DC into an attribute-hyperedge (tuple of attrs)
  - build local hypergraph via leakage.construct_local_hypergraph(target_cell, hyperedges, edge_weights, mode="MAX")

Then:
  Phase 1: compute L(∅), then L({c}) for each c in Z to get ΔL
  Phase 2: build nested candidates M_k = {π_1..π_k}
  Phase 3: compute true L(M_k) for each k
  Phase 4: single Exponential Mechanism over U(M_k) = -|M_k|/|Z| - λ·1[L(M_k)>L0]
"""

from __future__ import annotations

import os
import pickle
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from leakage import (
    get_dataset_weights,
    construct_local_hypergraph,
    leakage as chain_leakage,
    compute_utility_em,
)


def _safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_") or "obj"


def _build_hypergraph(dataset: str, target_cell: str, *, mode: str = "MAX"):
    """
    Mirrors greedy_gumbel.py hypergraph construction.
    """
    dataset = str(dataset)
    target_cell = str(target_cell)

    # Load DCs for dataset (same pattern as greedy_gumbel.py)
    try:
        mod = dataset.capitalize()
        dc_mod = __import__(
            f"DCandDelset.dc_configs.top{mod}DCs_parsed",
            fromlist=["denial_constraints"],
        )
        dcs = dc_mod.denial_constraints
    except Exception:
        dcs = []

    weights = get_dataset_weights(dataset)

    hyperedges: List[Tuple[str, ...]] = []
    edge_weights: List[float] = []

    for i, dc in enumerate(dcs):
        attrs = set()
        for pred in dc:
            for tok in (pred[0], pred[2]):
                if isinstance(tok, str) and "." in tok:
                    attrs.add(tok.split(".")[-1])
        if len(attrs) >= 2:
            hyperedges.append(tuple(sorted(attrs)))
            edge_weights.append(float(weights[i]) if i < len(weights) else 1.0)

    H = construct_local_hypergraph(target_cell, hyperedges, edge_weights, mode=mode)
    return H


def _inference_zone_union(target: str, hypergraph: Any) -> Set[str]:
    return {v for v in getattr(hypergraph, "vertices", []) if v != target}


def _em_sample_index(utilities: np.ndarray, epsilon: float, delta_u: float) -> int:
    # stable softmax over log-weights
    logw = (epsilon * utilities) / (2.0 * delta_u)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    p = w / np.sum(w)
    return int(np.random.choice(len(utilities), p=p))


def marginal_em_main(
    dataset: str,
    target_cell: str,
    epsilon: float,
    lam: float,
    L0: float,
    leakage_method: str = "greedy_disjoint",
) -> Dict[str, Any]:
    init_start = time.time()

    H = _build_hypergraph(dataset, target_cell, mode="MAX")
    Z = list(_inference_zone_union(target_cell, H))
    n = len(Z)

    # Phase 1 — Marginal ordering (n+1 calls including empty)
    L_empty, num_chains, active_chains, blocked_chains = chain_leakage(
        set(),
        target_cell,
        H,
        return_counts=True,
        leakage_method=leakage_method,
    )

    delta_L: Dict[str, float] = {}
    for c in Z:
        L_c = chain_leakage(
            {c},
            target_cell,
            H,
            return_counts=False,
            leakage_method=leakage_method,
        )
        delta_L[c] = float(L_empty) - float(L_c)

    ordering = sorted(Z, key=lambda c: delta_L[c], reverse=True)

    init_time = time.time() - init_start

    # Phase 2 + 3 — candidates + TRUE leakage calls for each candidate (n+1 calls)
    model_start = time.time()

    candidates: List[Set[str]] = []
    utilities: List[float] = []
    leakages: List[float] = []

    current_mask: Set[str] = set()
    for k in range(n + 1):
        if k > 0:
            current_mask = current_mask | {ordering[k - 1]}

        L_k = chain_leakage(
            current_mask,
            target_cell,
            H,
            return_counts=False,
            leakage_method=leakage_method,
        )

        U_k = compute_utility_em(
            leakage=float(L_k),
            mask_size=len(current_mask),
            zone_size=n,
            L0=float(L0),
            lambda_penalty=float(lam),
        )

        candidates.append(set(current_mask))
        utilities.append(float(U_k))
        leakages.append(float(L_k))

    utilities_arr = np.asarray(utilities, dtype=float)
    idx = _em_sample_index(utilities_arr, float(epsilon), float(lam))

    chosen_mask = candidates[idx]
    chosen_leakage = leakages[idx]
    chosen_utility = utilities[idx]

    model_time = time.time() - model_start

    # For compatibility with other outputs: include a PKL-based memory metric
    out_dir = "hypergraphs_pkl_marginal_em"
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(
        out_dir,
        f"hg_{_safe_filename(dataset)}_{_safe_filename(target_cell)}_MAX.pkl",
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(H, f, protocol=pickle.HIGHEST_PROTOCOL)
    memory_overhead_bytes = int(os.path.getsize(pkl_path) + sys.getsizeof(0.0))

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        # experiment_1.py overwrites del_time with update-to-null time when verbose=True,
        # so we set it to 0.0 here (same pattern used by some baselines).
        "del_time": 0.0,
        "leakage": float(chosen_leakage),
        "utility": float(chosen_utility),
        "mask": chosen_mask,
        "mask_size": int(len(chosen_mask)),
        "num_paths": int(num_chains),
        "num_instantiated_cells": int(n),
        "memory_overhead_bytes": int(memory_overhead_bytes),
    }
