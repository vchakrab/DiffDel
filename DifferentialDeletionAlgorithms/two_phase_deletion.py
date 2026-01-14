#!/usr/bin/env python3
"""
del2ph.py (two-phase deletion) that IMPORTS EVERYTHING from your leakage module.

Memory metric CHANGE (per your request):
  - This file already saves the TEMPLATE as a pkl.
  - We now ALSO save the ONLINE hypergraph (H_now) as a pkl.
  - We report memory as:
        memory_overhead_bytes = (template_pkl_size_bytes) + (hypergraph_pkl_size_bytes) + sizeof(py_float)
    where sizeof(py_float) = sys.getsizeof(0.0)

Notes:
  - Template is saved in build_template_two_phase(...) as before.
  - We ensure the template file size is read from disk (actual bytes).
  - We pickle H_now (or a snapshot fallback if needed) and add its file size too.
  - We keep your existing fields, but set "memory_overhead_bytes" to the new definition.
"""

from __future__ import annotations

import os
import time
import pickle
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, FrozenSet
from itertools import chain, combinations
from sys import getsizeof

import numpy as np

from DifferentialDeletionAlgorithms.leakage import compute_utility, compute_utility_hinge

# ============================================================
# IMPORT *ALL* CORE METHODS FROM leakage.py
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
    hypergraph_to_edge_dict,
    iter_chains,
)

# ============================================================
# Helpers (non-core)
# ============================================================

def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def compute_candidate_masks(
    target_cell: str,
    H_max: Hypergraph,
    *,
    tau: Optional[float] = None,
    mask_space: Optional[str] = None,   # None => full powerset; "canonical" => inference-derived subset
    canonical_max_chains: int = 2000,
    canonical_max_union_chains: int = 200,
) -> List[FrozenSet[str]]:
    """
    Candidate masks for Del2Ph (and DelExp), matching the same mask-space semantics as exponential_deletion:

      - mask_space=None: all masks (full powerset of I(c*) \ {c*})
      - mask_space="canonical": inference-derived subset built from inference chains in H_max under empty mask

    Returns a list of frozensets (so they can be dict keys / cached).
    """
    zone_set = H_max.vertices - {target_cell}
    zone = sorted(zone_set)
    print(zone)

    # NOTE: your original code returns here unconditionally (so canonical is unreachable).
    # Keeping your exact behavior as given.
    return [frozenset(m) for m in powerset(zone)] or [frozenset()]

    # --- unreachable canonical branch (kept for completeness) ---
    if str(mask_space).lower() != "canonical":
        raise ValueError(f"mask_space must be None or 'canonical' (got {mask_space!r}).")

    edges = hypergraph_to_edge_dict(H_max, tau=tau)

    candidates: Set[FrozenSet[str]] = {frozenset()}
    for v in zone_set:
        candidates.add(frozenset([v]))

    chain_infos: List[Tuple[float, FrozenSet[str]]] = []
    for ch in iter_chains(set(), target_cell, edges):
        supp: Set[str] = set()
        w_prod = 1.0
        for eid in ch:
            verts, w = edges[eid]
            supp |= set(verts)
            w_prod *= float(w)
        supp.discard(target_cell)
        if not supp:
            continue
        chain_infos.append((float(w_prod), frozenset(supp)))

    chain_infos.sort(key=lambda t: t[0], reverse=True)

    seen_supports: Set[FrozenSet[str]] = set()
    top_supports: List[FrozenSet[str]] = []
    for _w, supp_fs in chain_infos:
        if supp_fs in seen_supports:
            continue
        seen_supports.add(supp_fs)
        top_supports.append(supp_fs)
        candidates.add(supp_fs)
        if len(top_supports) >= int(canonical_max_chains):
            break

    topN = top_supports[: int(canonical_max_union_chains)]
    for i in range(len(topN)):
        for j in range(i + 1, len(topN)):
            candidates.add(topN[i] | topN[j])

    return list(candidates)


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
    sens = float(lam)
    scores = (float(epsilon) * utilities.astype(float)) / (2.0 * sens)
    return stable_softmax(scores)


def deep_sizeof(obj: Any, *, seen: Optional[Set[int]] = None) -> int:
    """Rough recursive memory estimate (unused in the new memory metric)."""
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


def template_pkl_path(dataset: str, target_attribute: str, save_dir: str) -> str:
    return os.path.join(save_dir, f"{dataset}_{target_attribute}.pkl")


def get_template_pkl_size_bytes(dataset: str, target_attribute: str, save_dir: str) -> int:
    p = template_pkl_path(dataset, target_attribute, save_dir)
    try:
        return int(os.path.getsize(p))
    except Exception:
        return 0


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
# OFFLINE: build + cache template (leakage.py end-to-end)
# ============================================================

def build_template_two_phase(
    dataset: str,
    target_attribute: str,
    *,
    save_dir: str = "templates_2ph",
    epsilon: float = 50.0,
    lam: float = 0.5,
    tau: Optional[float] = None,
    leakage_method: str = "greedy_disjoint",
    hypergraph_mode: str = "MAX",
    mask_space=None,  # None / "canonical"
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

    # 4) Hypergraph around target (imported)
    if str(hypergraph_mode).upper() == "ACTUAL":
        H: Hypergraph = construct_hypergraph_actual(target_attribute, rdrs, rdr_weights)
    else:
        H = construct_hypergraph_max(target_attribute, rdrs, rdr_weights)


    # 5) Inference zone I(c*) = neighbors of target (edges incident to target)
    zone_set: Set[str] = set()
    for (edge_verts, _w) in getattr(H, "edges", []):
        for v in edge_verts:
            if v != target_attribute:
                zone_set.add(v)

    zone: List[str] = sorted(zone_set)
    zone_size = len(zone)

    # 6) Candidate masks
    candidate_masks = compute_candidate_masks(target_attribute, H, mask_space=mask_space)
    if not candidate_masks:
        candidate_masks = [frozenset()]

    Leakage: Dict[FrozenSet[str], float] = {}
    Utility: Dict[FrozenSet[str], float] = {}
    Probability: Dict[FrozenSet[str], float] = {}

    # Diagnostics
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

    L_empty = leakage_model(mask=set(), target_cell=target_attribute, hypergraph=H,leakage_method=leakage_method, tau=tau)

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
    pkl_path = template_pkl_path(dataset, target_attribute, save_dir)
    with open(pkl_path, "wb") as f:
        pickle.dump(T, f, protocol=pickle.HIGHEST_PROTOCOL)

    return T


def load_template_two_phase(dataset: str, target_attribute: str, *, save_dir: str = "templates_2ph") -> Dict[str, Any]:
    pkl_path = template_pkl_path(dataset, target_attribute, save_dir)
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
    leakage_method: str = "greedy_disjoint",
    hypergraph_mode: str = "MAX",
    template_dir: str = "templates",
    mask_method: str = "None",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()

    init_time = 0.0

    # load / build template
    try:
        T = load_template_two_phase(dataset, target_cell, save_dir=template_dir)
    except FileNotFoundError:
        print(f"Building Template {dataset}_{target_cell}")
        start_time = time.time()
        T = build_template_two_phase(
            dataset,
            target_cell,
            save_dir=template_dir,
            epsilon=epsilon,
            lam=lam,
            tau=tau,
            leakage_method=leakage_method,
            hypergraph_mode=hypergraph_mode,
            mask_space=mask_method,
        )
        init_time = float(time.time() - start_time)

    # ensure cache matches params
    if (
        float(T.get("epsilon", -1.0)) != float(epsilon)
        or float(T.get("lam", -1.0)) != float(lam)
        or (T.get("tau", None) != tau)
        or str(T.get("leakage_method", "")) != str(leakage_method)
        or str(T.get("hypergraph_mode", "")) != str(hypergraph_mode)
    ):
        print(f"Building Template -  {dataset}_{target_cell}")
        start_time = time.time()
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
        init_time = float(time.time() - start_time)

    # template file size (now that it exists / refreshed)
    template_bytes = get_template_pkl_size_bytes(dataset, target_cell, template_dir)
    py_float_bytes = int(sys.getsizeof(0.0))

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
    leakage_empty = float(T["baseline_leakage_empty_mask"])
    mask_set = set(chosen)

    # Recompute inference-chain counts for THIS sampled mask
    H_now: Optional[Hypergraph] = None
    hg_pkl_path = ""
    hg_pkl_bytes = 0
    hg_pickled_ok = False

    try:
        raw_dcs_now = load_parsed_dcs_for_dataset(dataset)

        class _InitNow:
            def __init__(self, dataset: str, denial_constraints):
                self.dataset = dataset
                self.denial_constraints = denial_constraints

        init_manager_now = _InitNow(dataset, raw_dcs_now)
        rdrs_now, rdr_weights_now = dc_to_rdrs_and_weights(init_manager_now)

        if str(hypergraph_mode).upper() == "ACTUAL":
            H_now = construct_hypergraph_actual(target_cell, rdrs_now, rdr_weights_now)
        else:
            H_now = construct_hypergraph_max(target_cell, rdrs_now, rdr_weights_now)

        _L_tmp, num_paths, _act_ch, paths_blocked = leakage_model(
            mask_set,
            target_cell,
            H_now,
            tau=tau,
            return_counts=True,
            leakage_method=leakage_method,
        )

        # ✅ save H_now as pkl and get file size
        hg_pkl_path, hg_pkl_bytes, hg_pickled_ok = save_hypergraph_pkl(
            H_now,
            dataset=dataset,
            target_cell=target_cell,
            mode=str(hypergraph_mode),
            out_dir="hypergraphs_pkl_2ph",
            prefix="hg2ph",
        )

    except Exception:
        # fallback: use precomputed template diagnostics if recomputation fails
        num_paths = int(T.get("NumChains", {}).get(chosen, -1))
        paths_blocked = int(T.get("BlockedChains", {}).get(chosen, 0))

        # still try to pickle whatever we have (likely None -> small snapshot)
        hg_pkl_path, hg_pkl_bytes, hg_pickled_ok = save_hypergraph_pkl(
            H_now,
            dataset=dataset,
            target_cell=target_cell,
            mode=str(hypergraph_mode),
            out_dir="hypergraphs_pkl_2ph",
            prefix="hg2ph",
        )

    del_time = 0.0  # runner measures actual update-to-null time elsewhere

    # ✅ NEW memory metric:
    # template pkl bytes + hypergraph pkl bytes + sizeof(py_float)
    memory_overhead_bytes = int(template_bytes + int(hg_pkl_bytes) + py_float_bytes)

    num_instantiated_cells = int(T.get("num_instantiated_cells", len(T.get("zone", []))))

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": float(del_time),
        "leakage": float(leakage_val),
        "utility": float(utility_val),
        "mask_size": int(len(mask_set)),
        "mask": mask_set,
        "baseline_leakage": float(leakage_empty),
        "num_paths": int(num_paths),
        "memory_overhead_bytes": int(memory_overhead_bytes),
        "num_instantiated_cells": int(num_instantiated_cells),


    }


# ============================================================
# Optional CLI smoke test
# ============================================================

if __name__ == "__main__":
    # print(two_phase_deletion_main("adult", key=1, target_cell="education"))
    # print(two_phase_deletion_main("flight", key = 1, target_cell = "FlightNum"))
    print(two_phase_deletion_main("tax", key = 1, target_cell = "marital_status"))
    print(two_phase_deletion_main("airport", key = 1, target_cell = "scheduled_service"))
    print(two_phase_deletion_main("hospital", key = 1, target_cell = "ProviderNumber"))
