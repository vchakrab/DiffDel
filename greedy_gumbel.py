#!/usr/bin/env python3
# greedy_gumbel.py
"""
Top-Down Gumbel Deletion (exact LaTeX match) + PKL memory metric (2PH-compatible)

Implements Algorithm: Top-Down Gumbel Deletion (optimized leakage calls variant)

Semantics:
  - M is the DELETION mask (cells in M are nulled)
  - Start with FULL deletion mask: M <- I(c*) (here: inference_zone_union)
  - For each cell c, compare keep vs remove using pairwise Gumbel noise,
    but SKIP ComputeLeakage when removal cannot possibly win given the
    best-case utility bound.

Optimization idea:
  Let delta = g_keep - g_remove, and let U_high = -(|M|-1)/n (best-case remove utility
  ignoring leakage penalty). If U_high <= U_keep + delta then even the best possible
  U_remove cannot beat keep, so we can skip leakage computation.

We still report EM utility for compatibility with your pipeline/2PH plots.
"""

from __future__ import annotations

import math
import os
import pickle
import random
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

# -------------------- imports from your repo --------------------
from leakage import (
    get_dataset_weights,
    construct_local_hypergraph,
    leakage as chain_leakage,
)

from leakage import compute_utility_em
# ---------------------------------------------------------------


# ===============================================================
# PKL helpers (copied from your 2PH semantics)
# ===============================================================

def _safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_") or "obj"


def save_hypergraph_pkl(
    hg: Any,
    *,
    dataset: str,
    target_cell: str,
    mode: str,
    out_dir: str,
    prefix: str = "hg",
) -> Tuple[str, int, bool]:
    """
    Save hypergraph to pickle; if it fails, save a snapshot dict.
    Returns (path, size_bytes, pickled_ok).
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{prefix}_{_safe_filename(dataset)}_{_safe_filename(target_cell)}_{_safe_filename(mode)}.pkl"
    path = os.path.join(out_dir, fname)

    pickled_ok = True
    try:
        with open(path, "wb") as f:
            pickle.dump(hg, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pickled_ok = False
        snapshot = {
            "__class__": type(hg).__name__ if hg is not None else None,
            "__module__": type(hg).__module__ if hg is not None else None,
            "__dict__": getattr(hg, "__dict__", None) if hg is not None else None,
            "vertices": getattr(hg, "vertices", None) if hasattr(hg, "vertices") else None,
            "edges": getattr(hg, "edges", None) if hasattr(hg, "edges") else None,
            "hyperedges": getattr(hg, "hyperedges", None) if hasattr(hg, "hyperedges") else None,
            "out": getattr(hg, "out", None) if hasattr(hg, "out") else None,
            "adj": getattr(hg, "adj", None) if hasattr(hg, "adj") else None,
        }
        with open(path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    return path, int(size_bytes), bool(pickled_ok)


# ===============================================================
# Gumbel + inference zone helpers
# ===============================================================

def gumbel(scale: float) -> float:
    """Sample Gumbel(0, scale)."""
    if scale <= 0.0:
        return 0.0
    u = random.random()
    u = max(1e-12, min(1.0 - 1e-12, u))
    return -scale * math.log(-math.log(u))


def inference_zone_union(target: str, hypergraph: Any) -> Set[str]:
    """
    Inference zone = all vertices except target.
    NOTE: This matches your earlier helper. If your leakage.py defines
          a different inference zone notion, update this to match.
    """
    return {v for v in getattr(hypergraph, "vertices", []) if v != target}


# ===============================================================
# Core algorithm (optimized leakage calls variant you specified)
# ===============================================================

def top_down_gumbel_deletion(
    *,
    hypergraph: Any,
    target_cell: str,
    epsilon: float,
    L0: float,
    lambda_penalty: float,
    leakage_method: str = "greedy_disjoint",
    shuffle: bool = True,
) -> Tuple[Set[str], float]:
    """
    Optimized Top-Down Gumbel Deletion.

    Matches your specified pseudocode:
      - maintain U_keep = -|M|/n initially (since L_curr=0)
      - sample g_keep, g_remove then delta = g_keep - g_remove
      - bound check using U_high = -(|M|-1)/n
      - only compute leakage if removal could possibly win
      - update U_keep only upon successful removal
    Returns: (mask M, sampler_time_seconds)
    """
    t0 = time.time()

    # Inference zone I(c*) \ {c*} (since our helper excludes target)
    Z = inference_zone_union(target_cell, hypergraph)
    n = max(1, len(Z))

    # Start with full deletion mask
    M: Set[str] = set(Z)

    # Per-decision privacy budget and Gumbel scale
    eps_prime = float(epsilon) / float(n)
    b = (2.0 * float(lambda_penalty)) / max(1e-12, eps_prime)

    def indicator(x: float) -> float:
        return 1.0 if float(x) > float(L0) else 0.0

    # Full mask has zero leakage (per your algorithm)
    L_curr = 0.0

    # Line 7: initialize U_keep without needing leakage
    U_keep = -(float(len(M)) / float(n))  # since indicator(0) = 0

    cells = list(Z)
    if shuffle:
        random.shuffle(cells)

    for c in cells:
        # Lines 9-12
        g_keep = gumbel(b)
        g_remove = gumbel(b)
        delta = g_keep - g_remove

        # Line 13: best-case remove utility ignoring leakage penalty
        # (i.e., if leakage penalty = 0)
        U_high = -(float(len(M) - 1) / float(n))

        # Lines 15-17: if even best-case remove can't beat keep under noise gap, skip
        if U_high <= U_keep + delta:
            continue

        # Lines 18-21: now we actually need leakage
        M_removed = set(M)
        if c not in M_removed:
            continue
        M_removed.remove(c)

        L_remove = float(chain_leakage(
            M_removed,
            target_cell,
            hypergraph,
            leakage_method=leakage_method,
            return_counts=False,
        ))

        U_remove = U_high - float(lambda_penalty) * indicator(L_remove)

        # Lines 22-25: accept removal if it truly beats keep
        if U_remove > U_keep + delta:
            M = M_removed
            L_curr = L_remove
            U_keep = -(float(len(M)) / float(n)) - float(lambda_penalty) * indicator(L_curr)

    return M, time.time() - t0


# ===============================================================
# Main entry (runner compatible) -- unchanged return schema
# ===============================================================

def gumbel_deletion_main(
    dataset: str,
    target_cell: str,
    *,
    epsilon: float = 1.0,
    lam: float = 100.0,
    L0: float = 0.25,
    K: int = 100,  # kept for signature compatibility (unused here)
    leakage_method: str = "greedy_disjoint",
) -> Dict[str, Any]:

    init_start = time.time()

    # Load parsed DCs
    try:
        if dataset.lower() == "ncvoter":
            mod = "NCVoter"
        else:
            mod = dataset.capitalize()
        dc_mod = __import__(
            f"DCandDelset.dc_configs.top{mod}DCs_parsed",
            fromlist=["denial_constraints"],
        )
        dcs = dc_mod.denial_constraints
    except Exception:
        dcs = []

    weights = get_dataset_weights(dataset)

    # Convert DCs → hyperedges
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

    # Build hypergraph
    H = construct_local_hypergraph(
        target_cell, hyperedges, edge_weights, mode="MAX"
    )

    # Save hypergraph to PKL and compute memory_overhead_bytes (2PH-compatible key name)
    _hg_pkl_path, hg_pkl_bytes, _hg_pickled_ok = save_hypergraph_pkl(
        H,
        dataset=dataset,
        target_cell=target_cell,
        mode="MAX",
        out_dir= "DifferentialDeletionAlgorithms/hypergraphs_pkl_gumbel",
        prefix="hggum",
    )
    py_float_bytes = int(sys.getsizeof(0.0))
    memory_overhead_bytes = int(hg_pkl_bytes + py_float_bytes)

    init_time = time.time() - init_start

    # Run algorithm
    model_start = time.time()

    M, _sampler_time = top_down_gumbel_deletion(
        hypergraph=H,
        target_cell=target_cell,
        epsilon=epsilon,
        L0=L0,
        lambda_penalty=lam,
        leakage_method=leakage_method,
    )

    L, num_chains, _act_ch, _blk_ch = chain_leakage(
        M,
        target_cell,
        H,
        leakage_method=leakage_method,
        return_counts=True,
    )

    inference_zone = inference_zone_union(target_cell, H)

    # Keep EM utility for comparability in your pipeline (as you requested)
    utility = compute_utility_em(
        leakage=float(L),
        mask_size=len(M),
        zone_size=len(inference_zone),
        L0=float(L0),
        lambda_penalty=float(lam),
    )

    model_time = time.time() - model_start

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": 0.0,
        "leakage": float(L),
        "utility": float(utility),
        "mask": M,
        "mask_size": int(len(M)),
        "num_paths": int(num_chains),
        "num_instantiated_cells": int(len(inference_zone)),
        "memory_overhead_bytes": int(memory_overhead_bytes),
    }


# ===============================================================
# Manual smoke test
# ===============================================================

if __name__ == "__main__":
    print(gumbel_deletion_main("adult", "education"))
    print(gumbel_deletion_main("flight", "FlightNum"))
    print(gumbel_deletion_main("airport", "scheduled_service"))
    print(gumbel_deletion_main("hospital", "ProviderNumber"))
    print(gumbel_deletion_main("tax", "marital_status"))
