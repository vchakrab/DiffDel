#!/usr/bin/env python3
# surrogate_em.py
"""
Algorithm 2: SurrogateEM (n leakage calls, ε-DP, NOT responsive)

Same Phase 1 ordering as MarginalEM, but does NOT compute true L(M_k).
Instead uses:
  L_hat(M_k) = max(0, L(∅) - sum_{i<=k} ΔL(π_i))
and applies EM to surrogate utility.
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
    dataset = str(dataset)
    target_cell = str(target_cell)

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
    logw = (epsilon * utilities) / (2.0 * delta_u)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    p = w / np.sum(w)
    return int(np.random.choice(len(utilities), p=p))


def surrogate_em_main(
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

    # Phase 2 — surrogate leakage + utilities (NO extra leakage calls)
    model_start = time.time()

    candidates: List[Set[str]] = []
    utilities: List[float] = []
    leakages_hat: List[float] = []

    current_mask: Set[str] = set()
    cum_reduction = 0.0

    for k in range(n + 1):
        if k > 0:
            c = ordering[k - 1]
            current_mask = current_mask | {c}
            cum_reduction += float(delta_L[c])

        L_hat = max(0.0, float(L_empty) - cum_reduction)

        U_k = compute_utility_em(
            leakage=float(L_hat),
            mask_size=len(current_mask),
            zone_size=n,
            L0=float(L0),
            lambda_penalty=float(lam),
        )

        candidates.append(set(current_mask))
        utilities.append(float(U_k))
        leakages_hat.append(float(L_hat))

    utilities_arr = np.asarray(utilities, dtype=float)
    idx = _em_sample_index(utilities_arr, float(epsilon), float(lam))

    chosen_mask = candidates[idx]
    chosen_leakage_hat = leakages_hat[idx]
    chosen_utility = utilities[idx]

    model_time = time.time() - model_start

    # Memory overhead metric
    out_dir = "hypergraphs_pkl_surrogate_em"
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
        "del_time": 0.0,
        "leakage": float(chosen_leakage_hat),
        "utility": float(chosen_utility),
        "mask": chosen_mask,
        "mask_size": int(len(chosen_mask)),
        "num_paths": int(num_chains),
        "num_instantiated_cells": int(n),
        "memory_overhead_bytes": int(memory_overhead_bytes),
    }
