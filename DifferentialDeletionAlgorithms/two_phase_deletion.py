#!/usr/bin/env python3
"""
del2ph.py (two-phase deletion) that IMPORTS EVERYTHING from your leakage module.

You said: don't re-define leakage / hypergraph / map_dc_to_weight / etc here.
So this file only:
  - loads parsed DCs
  - calls leakage.dc_to_rdrs_and_weights(...)
  - calls leakage.construct_local_hypergraph(...)
  - calls leakage.leakage(...)
  - calls leakage.compute_utility(...)
  - does exp-mech + caching + sampling
"""

from __future__ import annotations

import os
import time
import pickle
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, FrozenSet
from itertools import chain, combinations
from sys import getsizeof

import numpy as np

# ============================================================
# IMPORT *ALL* CORE METHODS FROM leakage.py
# (this is exactly the big blob you pasted)
# ============================================================

from leakage import (
    # weights + DC -> (rdrs, weights)
    get_dataset_weights,
    map_dc_to_weight,
    dc_to_rdrs_and_weights,

    # hypergraph (Algorithm 1)
    Hypergraph,
    construct_local_hypergraph,
    construct_hypergraph_max,
    construct_hypergraph_actual,

    # utility
    compute_utility,

    # leakage (Algorithm 2/5)
    leakage as leakage_model,
)

# ============================================================
# Helpers (non-core)
# ============================================================

def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.array([], dtype=float)
    m = float(np.max(scores))
    ex = np.exp(scores - m)
    z = float(np.sum(ex))
    if z <= 0.0 or not np.isfinite(z):
        return np.ones_like(scores, dtype=float) / max(1, scores.size)
    return ex / z


def exp_mech_probs(utilities: np.ndarray, epsilon: float, lam: float) -> np.ndarray:
    # scores = (ε * u) / (2λ)
    sens = max(1e-12, float(lam))
    scores = (float(epsilon) * utilities.astype(float)) / (2.0 * sens)
    return stable_softmax(scores)


def deep_sizeof(obj: Any, *, seen: Optional[Set[int]] = None) -> int:
    """Rough recursive memory estimate."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)

    size = getsizeof(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_sizeof(k, seen=seen)
            size += deep_sizeof(v, seen=seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for x in obj:
            size += deep_sizeof(x, seen=seen)

    return int(size)


# ============================================================
# Loading parsed DCs
# ============================================================

def load_parsed_dcs_for_dataset(dataset: str) -> List[List[Tuple[str, str, str]]]:
    """
    Loads:
      DCandDelset.dc_configs.top{Dataset}DCs_parsed.denial_constraints

    Mirrors your existing naming:
      ncvoter -> topNCVoterDCs_parsed
      else   -> top{Dataset.capitalize()}DCs_parsed
    """
    ds = str(dataset).lower()
    mod_name = "NCVoter" if ds == "ncvoter" else ds.capitalize()
    dc_module_path = f"DCandDelset.dc_configs.top{mod_name}DCs_parsed"
    try:
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        return getattr(dc_module, "denial_constraints", []) or []
    except Exception:
        return []


# ============================================================
# OFFLINE: build + cache template (now uses leakage.py end-to-end)
# ============================================================

def build_template_two_phase(
    dataset: str,
    target_attribute: str,
    *,
    save_dir: str = "templates_2ph",
    epsilon: float = 50.0,
    lam: float = 0.5,
    tau: Optional[float] = None,
    leakage_method: str = "noisy_or",   # or "greedy_disjoint"
    hypergraph_mode: str = "MAX",       # "MAX" or "ACTUAL"
) -> Dict[str, Any]:
    # 1) load DCs
    raw_dcs = load_parsed_dcs_for_dataset(dataset)

    # 2) minimal init_manager wrapper for dc_to_rdrs_and_weights
    class _Init:
        def __init__(self, dataset: str, denial_constraints):
            self.dataset = dataset
            self.denial_constraints = denial_constraints

    init_manager = _Init(dataset, raw_dcs)

    # 3) DCs -> rdrs + weights (imported)
    rdrs, rdr_weights = dc_to_rdrs_and_weights(init_manager)

    # 4) Local hypergraph around target (imported)
    if str(hypergraph_mode).upper() == "ACTUAL":
        H: Hypergraph = construct_hypergraph_actual(target_attribute, rdrs, rdr_weights)
    else:
        H = construct_hypergraph_max(target_attribute, rdrs, rdr_weights)

    # 5) Inference zone I(c*) = direct neighbors of target (edges incident to target)
    zone_set: Set[str] = set()
    for (edge_verts, _w) in getattr(H, "edges", []):
        if target_attribute in edge_verts:
            for v in edge_verts:
                if v != target_attribute:
                    zone_set.add(v)

    zone: List[str] = sorted(zone_set)
    zone_size = len(zone)

    # 6) Candidate masks
    candidate_masks: List[FrozenSet[str]] = [frozenset(m) for m in powerset(zone)]
    if not candidate_masks:
        candidate_masks = [frozenset()]

    Leakage: Dict[FrozenSet[str], float] = {}
    Utility: Dict[FrozenSet[str], float] = {}
    Probability: Dict[FrozenSet[str], float] = {}

    # Optional diagnostics from leakage.py
    NumChains: Dict[FrozenSet[str], int] = {}
    ActiveChains: Dict[FrozenSet[str], int] = {}
    BlockedChains: Dict[FrozenSet[str], int] = {}

    utilities_arr = np.empty(len(candidate_masks), dtype=float)

    for i, m in enumerate(candidate_masks):
        L, num_ch, act_ch, blk_ch = leakage_model(
            mask=set(m),
            target_cell=target_attribute,
            hypergraph=H,
            tau=tau,
            return_counts=True,
            leakage_method=leakage_method,
        )

        U = compute_utility(leakage=float(L), mask_size=len(m), lam=float(lam), zone_size=zone_size)

        Leakage[m] = float(L)
        Utility[m] = float(U)
        NumChains[m] = int(num_ch)
        ActiveChains[m] = int(act_ch)
        BlockedChains[m] = int(blk_ch)

        utilities_arr[i] = float(U)

    probs_arr = exp_mech_probs(utilities_arr, epsilon=float(epsilon), lam=float(lam))
    for m, p in zip(candidate_masks, probs_arr):
        Probability[m] = float(p)

    L_empty = float(Leakage.get(frozenset(), 0.0))

    T: Dict[str, Any] = {
        "dataset": str(dataset),
        "target": str(target_attribute),

        "epsilon": float(epsilon),
        "lam": float(lam),
        "tau": tau,
        "leakage_method": str(leakage_method),
        "hypergraph_mode": str(hypergraph_mode),

        "zone": zone,
        "R_intra": candidate_masks,
        "Leakage": Leakage,
        "Utility": Utility,
        "Probability": Probability,

        "baseline_leakage_empty_mask": float(L_empty),

        # diagnostics
        "NumChains": NumChains,
        "ActiveChains": ActiveChains,
        "BlockedChains": BlockedChains,

        "num_instantiated_cells": int(len(zone)),
    }

    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f"{dataset}_{target_attribute}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(T, f)

    return T


def load_template_two_phase(dataset: str, target_attribute: str, *, save_dir: str = "templates_2ph") -> Dict[str, Any]:
    pkl_path = os.path.join(save_dir, f"{dataset}_{target_attribute}.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ============================================================
# ONLINE: sample mask
# ============================================================

def two_phase_deletion_main(
    dataset: str,
    key: int,
    target_cell: str,
    *,
    epsilon: float = 50.0,
    lam: float = 0.5,
    tau: Optional[float] = None,
    leakage_method: str = "noisy_or",
    hypergraph_mode: str = "MAX",
    template_dir: str = "templates_2ph",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()

    init_time = 0.0

    # load / build
    try:
        T = load_template_two_phase(dataset, target_cell, save_dir=template_dir)
    except FileNotFoundError:
        T = build_template_two_phase(
            dataset,
            target_cell,
            save_dir=template_dir,
            epsilon=epsilon,
            lam=lam,
            tau=tau,
            leakage_method=leakage_method,
            hypergraph_mode=hypergraph_mode,
        )

    # ensure cache matches params
    if (
        float(T.get("epsilon", -1.0)) != float(epsilon)
        or float(T.get("lam", -1.0)) != float(lam)
        or (T.get("tau", None) != tau)
        or str(T.get("leakage_method", "")) != str(leakage_method)
        or str(T.get("hypergraph_mode", "")) != str(hypergraph_mode)
    ):
        T = build_template_two_phase(
            dataset,
            target_cell,
            save_dir=template_dir,
            epsilon=epsilon,
            lam=lam,
            tau=tau,
            leakage_method=leakage_method,
            hypergraph_mode=hypergraph_mode,
        )

    model_start = time.time()

    masks: List[FrozenSet[str]] = T["R_intra"]
    probs_dict: Dict[FrozenSet[str], float] = T["Probability"]
    probs = np.array([probs_dict[m] for m in masks], dtype=float)

    s = float(probs.sum())
    if s <= 0.0 or not np.isfinite(s):
        probs = np.ones_like(probs) / max(1, probs.size)
    else:
        probs /= s

    idx = int(rng.choice(len(masks), p=probs))
    chosen = masks[idx]

    model_time = float(time.time() - model_start)

    leakage_val = float(T["Leakage"][chosen])
    utility_val = float(T["Utility"][chosen])
    mask_set = set(chosen)

    # Keep these fields compatible with your runner
    num_paths = int(T.get("NumChains", {}).get(chosen, -1))      # proxy: #chains
    paths_blocked = int(T.get("BlockedChains", {}).get(chosen, 0))

    del_time = 0.0  # runner measures actual update-to-null time elsewhere

    memory_overhead_bytes = deep_sizeof(T) + deep_sizeof(mask_set)
    num_instantiated_cells = int(T.get("num_instantiated_cells", len(T.get("zone", []))))

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": float(del_time),

        "leakage": float(leakage_val),
        "utility": float(utility_val),
        "mask_size": int(len(mask_set)),
        "mask": set(mask_set),

        "num_paths": int(num_paths),
        "paths_blocked": int(paths_blocked),

        "memory_overhead_bytes": int(memory_overhead_bytes),
        "num_instantiated_cells": int(num_instantiated_cells),
    }


# ============================================================
# Optional CLI smoke test
# ============================================================

if __name__ == "__main__":
    # quick sanity run:
    # python del2ph.py
    ds = "airport"
    tgt = "iso_country"
    out = two_phase_deletion_main(ds, key=0, target_cell=tgt)
    print(out)
