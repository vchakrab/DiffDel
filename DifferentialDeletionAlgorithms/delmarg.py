#!/usr/bin/env python3
# delmarg.py
"""
DelMarg: Marginal ordering + candidate evaluation + one-shot Exponential Mechanism,
with leakage computed using greedy mask-disjoint selection.

This is the DelMarg algorithm described by the user, instantiated using the
existing leakage.py implementation:
  - leakage(..., leakage_method="greedy_disjoint")
which already performs:
  - chain enumeration (iter_chains_with_masked)
  - greedy disjoint selection (greedy_mask_disjoint)
  - noisy-or aggregation over selected disjoint chains

So in this code:
  - "ComputeLeakage" / "MaskDisjointSelect" == leakage(..., leakage_method="greedy_disjoint")
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
    leakage as compute_leakage,
    compute_utility_em,
)


def _safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_") or "obj"


def _build_hypergraph(dataset: str, target_cell: str, *, mode: str = "MAX"):
    """
    Mirrors the baseline construction used elsewhere:
      - load denial constraints
      - map each DC to an attribute hyperedge
      - use dataset weights
      - construct local hypergraph around target_cell
    """
    dataset = str(dataset).lower()
    target_cell = str(target_cell)

    # Load DC module the same way as other baselines do
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
            # pred is (a, op, b) where a/b might look like "t.attr"
            for tok in (pred[0], pred[2]):
                if isinstance(tok, str) and "." in tok:
                    attrs.add(tok.split(".", 1)[1])
        if len(attrs) >= 2:
            hyperedges.append(tuple(sorted(attrs)))
            edge_weights.append(float(weights[i]) if i < len(weights) else 1.0)

    H = construct_local_hypergraph(target_cell, hyperedges, edge_weights, mode=mode)
    return H


def _inference_zone(target: str, H: Any) -> List[str]:
    # inference zone Z = all vertices except target
    verts = list(getattr(H, "vertices", []))
    return [v for v in verts if v != target]


def _em_sample_index(utilities: np.ndarray, epsilon: float, delta_u: float) -> int:
    # stable EM sampling
    logw = (epsilon * utilities) / (2.0 * delta_u)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    p = w / np.sum(w)
    return int(np.random.choice(len(utilities), p=p))


def delmarg_main(
    dataset: str,
    target_cell: str,
    epsilon: float,
    lam: float,
    L0: float,
    leakage_method: str = "greedy_disjoint",
) -> Dict[str, Any]:
    """
    Returns a dict compatible with experiment_1.py's standardize_row pipeline.

    Leakage calls:
      1   : L(∅)
      n   : L({c}) for each c in Z
      n+1 : L(M_k) for k=0..n
      -----------------------
      2n + 2 total leakage calls

    Uses leakage_method="greedy_disjoint" by default (DelMarg spec).
    """

    init_start = time.time()

    H = _build_hypergraph(dataset, target_cell, mode="MAX")
    Z = _inference_zone(target_cell, H)
    n = len(Z)

    # Phase 0 + Phase 1 base leakage
    # (We rely on leakage.py's internal chain enumeration + greedy disjoint selection.)
    L_base, num_paths, _active, _blocked = compute_leakage(
        set(),
        target_cell,
        H,
        return_counts=True,
        leakage_method=leakage_method,
    )

    # Phase 1 — marginal ordering
    delta_L: Dict[str, float] = {}
    for c in Z:
        L_i = compute_leakage(
            {c},
            target_cell,
            H,
            return_counts=False,
            leakage_method=leakage_method,
        )
        delta_L[c] = float(L_base) - float(L_i)

    ordering = sorted(Z, key=lambda c: delta_L[c], reverse=True)

    init_time = time.time() - init_start

    # Phase 2 — candidate evaluation
    model_start = time.time()

    candidates: List[Set[str]] = []
    utilities: List[float] = []
    leakages: List[float] = []

    current_mask: Set[str] = set()
    for k in range(n + 1):
        if k > 0:
            current_mask = current_mask | {ordering[k - 1]}

        L_k = compute_leakage(
            current_mask,
            target_cell,
            H,
            return_counts=False,
            leakage_method=leakage_method,
        )

        U_k = compute_utility_em(
            leakage=float(L_k),
            mask_size=int(len(current_mask)),
            zone_size=int(n),
            L0=float(L0),
            lambda_penalty=float(lam),
        )

        candidates.append(set(current_mask))
        leakages.append(float(L_k))
        utilities.append(float(U_k))

    # Phase 3 — private selection (single EM)
    delta_u = float(lam)
    utilities_arr = np.asarray(utilities, dtype=float)
    idx = _em_sample_index(utilities_arr, float(epsilon), delta_u)

    chosen_mask = candidates[idx]
    chosen_leakage = leakages[idx]
    chosen_utility = utilities[idx]

    model_time = time.time() - model_start

    # Memory overhead metric: store hypergraph snapshot similarly to other baselines
    out_dir = "hypergraphs_pkl_delmarg"
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(
        out_dir, f"hg_{_safe_filename(dataset)}_{_safe_filename(target_cell)}_MAX.pkl"
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(H, f, protocol=pickle.HIGHEST_PROTOCOL)

    memory_overhead_bytes = int(os.path.getsize(pkl_path) + sys.getsizeof(0.0))

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        # experiment_1.py overwrites del_time with update-to-NULL time when verbose=True
        "del_time": 0.0,
        "leakage": float(chosen_leakage),
        "utility": float(chosen_utility),
        "mask": chosen_mask,
        "mask_size": int(len(chosen_mask)),
        "num_paths": int(num_paths),
        "num_instantiated_cells": int(n),
        "memory_overhead_bytes": int(memory_overhead_bytes),
    }
