#!/usr/bin/env python3
"""
two_phase_deletion.py  (NEW, λ/ε version)

Two-phase "delexp split":
- OFFLINE: build a cached template per (dataset, target_attribute)
    - loads DC hyperedges
    - applies dataset-specific edge weights (same idea as delexp)
    - defines inference zone I(c*) = direct neighbors of target in the hypergraph
    - enumerates masks over I(c*)
    - computes leakage L(M) using InferableLeakageModel fixed-point
    - computes utility u(M) = -(λ * L(M)) - (1-λ) * (|M|/(|I(c*)|-1))
    - computes exp-mech probabilities: Pr[M] ∝ exp( ε*u(M) / (2λ) )
    - stores: masks, leakage, utility, probs, plus diagnostics (num_paths, L_empty)

- ONLINE: sample a mask from cached probabilities (fast), return standardized metrics
    - DOES NOT update DB here (runner measures update_to_null separately)

Defaults requested by you:
- epsilon = 50
- lam = 0.5
"""

from __future__ import annotations

import os
import time
import pickle
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, FrozenSet
from itertools import chain, combinations
from collections import deque
from sys import getsizeof

import numpy as np


Cell = Any


# ============================================================
# Helpers
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


def _normalize_hyperedges(hyperedges: Sequence[Iterable[str]]) -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    for e in hyperedges:
        s = tuple(sorted(set(e)))
        if len(s) >= 2:
            out.append(s)
    return out


def get_dataset_weights(dataset: str):
    """
    Matches your delexp behavior:
      module_name = weights.weights_corrected.<dataset>_weights
      returns WEIGHTS or None
    """
    dataset = dataset.lower()
    module_name = f"weights.weights_corrected.{dataset}_weights"
    try:
        weights_module = importlib.import_module(module_name)
        return getattr(weights_module, "WEIGHTS", None)
    except ModuleNotFoundError:
        print(f"[WARN] No weights module found for dataset: {dataset}")
        return None


def _normalize_edge_weights(hyperedges: Sequence[Iterable[str]], edge_weights: Optional[Any]) -> List[float]:
    m = len(hyperedges)
    if edge_weights is None:
        return [0.8] * m
    if isinstance(edge_weights, dict):
        return [float(edge_weights.get(i, 0.8)) for i in range(m)]
    if isinstance(edge_weights, (list, tuple, np.ndarray)):
        w = list(edge_weights)
        if len(w) != m:
            w = (w + [0.8] * m)[:m]
        return [float(x) for x in w]
    return [0.8] * m


def clean_raw_dcs(raw_dcs: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
    """
    Best-effort extraction of attribute tokens from parsed DCs.
    Pulls suffix after '.' from things like "t1.education".
    """
    cleaned: List[Tuple[str, ...]] = []
    for dc in raw_dcs:
        attrs: Set[str] = set()
        for pred in dc:
            if not isinstance(pred, (list, tuple)) or len(pred) < 3:
                continue
            for item in (pred[0], pred[2]):
                if isinstance(item, str) and "." in item:
                    attrs.add(item.split(".")[-1])
        if attrs:
            cleaned.append(tuple(sorted(attrs)))
    return cleaned


# ============================================================
# Inferable leakage model (fixed-point, NO PATH enumeration)
# ============================================================

@dataclass(frozen=True)
class Edge:
    verts: Tuple[int, ...]
    w: float


class InferableLeakageModel:
    """
    Same semantics as your delexp:
      - vertices are attributes/cells
      - an edge can infer a vertex if all other vertices in that edge are inferable
      - each edge has probability weight w
    Leakage of target is:
      L = 1 - Π_{edges containing target}(1 - w * Π_{others} p[other])
    """

    def __init__(self, hyperedges: Sequence[Iterable[str]], weights: Sequence[float], target: str):
        H = _normalize_hyperedges(hyperedges)
        W = list(weights)
        if len(H) != len(W):
            raise ValueError("hyperedges and weights must have same length")

        V: Set[str] = {target}
        hedges: List[Set[str]] = []
        for e in H:
            s = set(e)
            if len(s) < 2:
                continue
            V |= s
            hedges.append(s)

        ordered = sorted(V)
        self.cell_to_id: Dict[str, int] = {c: i for i, c in enumerate(ordered)}
        self.id_to_cell: List[str] = ordered
        self.n = len(ordered)

        self.target = target
        self.tid = self.cell_to_id[target]

        self.edges: List[Edge] = []
        for s, w in zip(hedges, W):
            ww = float(w)
            if not (0.0 < ww <= 1.0):
                ww = min(1.0, max(1e-12, ww))
            verts = tuple(sorted(self.cell_to_id[c] for c in s))
            self.edges.append(Edge(verts=verts, w=ww))

        self.inc: List[List[int]] = [[] for _ in range(self.n)]
        for ei, e in enumerate(self.edges):
            for v in e.verts:
                self.inc[v].append(ei)

        neigh_sets: List[Set[int]] = [set() for _ in range(self.n)]
        for e in self.edges:
            for v in e.verts:
                neigh_sets[v].update(e.verts)
        for v in range(self.n):
            neigh_sets[v].discard(v)
        self.neigh: List[Tuple[int, ...]] = [tuple(sorted(s)) for s in neigh_sets]

        self.channel_edges: List[int] = [ei for ei, e in enumerate(self.edges) if self.tid in e.verts]

    def _attempt(self, e: Edge, infer_v: int, p: List[float]) -> float:
        prod = 1.0
        for u in e.verts:
            if u == infer_v:
                continue
            prod *= p[u]
            if prod == 0.0:
                return 0.0
        a = e.w * prod
        if a <= 0.0:
            return 0.0
        if a >= 1.0:
            return 1.0
        return a

    def _recompute_pv(self, v: int, observed: List[bool], p: List[float]) -> float:
        if observed[v]:
            return 1.0
        prod_fail = 1.0
        for ei in self.inc[v]:
            e = self.edges[ei]
            a = self._attempt(e, v, p)
            if a == 0.0:
                continue
            if a == 1.0:
                return 1.0
            prod_fail *= (1.0 - a)
            if prod_fail == 0.0:
                return 1.0
        return 1.0 - prod_fail

    def _compute_L(self, p: List[float]) -> float:
        t = self.tid
        prod_fail = 1.0
        for ei in self.channel_edges:
            e = self.edges[ei]
            prod = 1.0
            for u in e.verts:
                if u == t:
                    continue
                prod *= p[u]
                if prod == 0.0:
                    break
            q = e.w * prod
            if q <= 0.0:
                continue
            if q >= 1.0:
                return 1.0
            prod_fail *= (1.0 - q)
            if prod_fail == 0.0:
                return 1.0
        return 1.0 - prod_fail

    def leakage(self, mask: Set[str], *, tau: float = 1e-10, max_updates: int = 2_000_000) -> float:
        mask_ids = {self.cell_to_id[c] for c in mask if c in self.cell_to_id}
        if self.tid in mask_ids:
            raise ValueError("mask includes target")

        observed = [True] * self.n
        observed[self.tid] = False
        for mid in mask_ids:
            observed[mid] = False

        p = [1.0 if observed[v] else 0.0 for v in range(self.n)]
        Q = deque(range(self.n))
        in_q = [True] * self.n
        pops = 0

        while Q and pops < max_updates:
            v = Q.popleft()
            in_q[v] = False
            pops += 1
            if observed[v]:
                continue
            new_p = self._recompute_pv(v, observed, p)
            if abs(new_p - p[v]) > tau:
                p[v] = new_p
                for u in self.neigh[v]:
                    if not in_q[u]:
                        Q.append(u)
                        in_q[u] = True

        L = float(self._compute_L(p))
        return float(max(0.0, min(1.0, L)))


# ============================================================
# Utility + exp mechanism (NEW λ version)
# ============================================================

def compute_utility(mask_size: int, leakage: float, lam: float, zone_size: int) -> float:
    """
    u(M) = -(λ * L(M)) - (1-λ) * (|M| / (|I(c*)|-1))
    where I(c*) = inference zone (direct neighbors of target).
    """
    denom = max(1, int(zone_size) - 1)
    norm = float(mask_size) / float(denom) if zone_size > 1 else 0.0
    return float(-(lam * float(leakage)) - ((1.0 - lam) * norm))


def exp_mech_probs(utilities: np.ndarray, epsilon: float, lam: float) -> np.ndarray:
    # scores = (ε * u) / (2λ)   (sensitivity = λ)
    sens = max(1e-12, float(lam))
    scores = (float(epsilon) * utilities.astype(float)) / (2.0 * sens)
    return stable_softmax(scores)


# ============================================================
# OFFLINE: build + cache template
# ============================================================

def build_template_two_phase(
    dataset: str,
    target_attribute: str,
    *,
    save_dir: str = "templates_2ph",
    epsilon: float = 50.0,
    lam: float = 0.5,
) -> Dict[str, Any]:
    """
    Builds and caches template for (dataset, target_attribute).

    Template stores:
      - hyperedges + weights
      - zone (direct neighbors)
      - masks list (frozensets)
      - Leakage[mask], Utility[mask], Probability[mask]
      - num_paths proxy (= #target-incident hyperedges)
      - baseline leakage of empty mask (for blocked proxy)
    """

    ds = dataset
    attr = target_attribute

    # Load parsed DCs
    try:
        if ds.lower() == "ncvoter":
            mod_name = "NCVoter"
        else:
            mod_name = ds.capitalize()
        dc_module_path = f"DCandDelset.dc_configs.top{mod_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        raw_dcs = getattr(dc_module, "denial_constraints", [])
    except Exception:
        raw_dcs = []

    hyperedges_raw = clean_raw_dcs(raw_dcs)
    H = _normalize_hyperedges(hyperedges_raw)

    W = _normalize_edge_weights(H, get_dataset_weights(ds))

    model = InferableLeakageModel(H, W, target=attr)

    # Inference zone I(c*) = direct neighbors of target (matches your delexp candidate construction)
    zone_set: Set[str] = set()
    for e in H:
        if attr in e:
            for v in e:
                if v != attr:
                    zone_set.add(v)
    zone: List[str] = sorted(zone_set)

    # Candidate masks: all subsets of zone (if no neighbors, still include empty)
    candidate_masks: List[FrozenSet[str]] = [frozenset(m) for m in powerset(zone)]
    if not candidate_masks:
        candidate_masks = [frozenset()]

    Leakage: Dict[FrozenSet[str], float] = {}
    Utility: Dict[FrozenSet[str], float] = {}
    utilities_arr = np.empty(len(candidate_masks), dtype=float)

    zone_size = len(zone)

    for i, m in enumerate(candidate_masks):
        L = float(model.leakage(set(m)))
        U = compute_utility(mask_size=len(m), leakage=L, lam=float(lam), zone_size=zone_size)
        Leakage[m] = L
        Utility[m] = U
        utilities_arr[i] = U

    probs_arr = exp_mech_probs(utilities_arr, epsilon=float(epsilon), lam=float(lam))
    Probability: Dict[FrozenSet[str], float] = {m: float(p) for m, p in zip(candidate_masks, probs_arr)}

    # proxies/diagnostics
    num_paths = int(len(model.channel_edges))
    L_empty = float(Leakage.get(frozenset(), float(model.leakage(set()))))

    T_attr = {
        "dataset": ds,
        "target": attr,

        "hyperedges": H,
        "weights": W,

        "zone": zone,                   # direct neighbors of target
        "R_intra": candidate_masks,      # list[frozenset]
        "Leakage": Leakage,              # dict[frozenset]->float
        "Utility": Utility,              # dict[frozenset]->float
        "Probability": Probability,      # dict[frozenset]->float

        "epsilon": float(epsilon),
        "lam": float(lam),

        "num_paths": num_paths,
        "baseline_leakage_empty_mask": float(L_empty),
    }

    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f"{ds}_{attr}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(T_attr, f)

    return T_attr


def load_template_two_phase(
    dataset: str,
    target_attribute: str,
    *,
    save_dir: str = "templates_2ph",
) -> Dict[str, Any]:
    pkl_path = os.path.join(save_dir, f"{dataset}_{target_attribute}.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ============================================================
# ONLINE: sample mask + return standardized fields (NO DB writes)
# ============================================================

def two_phase_deletion_main(
    dataset: str,
    key: int,
    target_cell: str,
    *,
    epsilon: float = 50.0,
    lam: float = 0.5,
    template_dir: str = "templates_2ph",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Returns keys compatible with your runner's standardize_row():
      init_time, model_time, del_time,
      leakage, utility, mask_size, mask,
      num_paths, paths_blocked,
      memory_overhead_bytes,
      num_instantiated_cells
    """
    if rng is None:
        rng = np.random.default_rng()

    # init_time: 0 here (offline already done)
    init_time = 0.0

    # load template; if not found, build it
    try:
        T = load_template_two_phase(dataset, target_cell, save_dir=template_dir)
    except FileNotFoundError:
        T = build_template_two_phase(
            dataset, target_cell,
            save_dir=template_dir,
            epsilon=epsilon,
            lam=lam
        )

    # If caller passes different epsilon/lam than cached, rebuild to match requested settings
    if float(T.get("epsilon", -1.0)) != float(epsilon) or float(T.get("lam", -1.0)) != float(lam):
        T = build_template_two_phase(
            dataset, target_cell,
            save_dir=template_dir,
            epsilon=epsilon,
            lam=lam
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

    leakage = float(T["Leakage"][chosen])
    utility = float(T["Utility"][chosen])
    mask_set = set(chosen)

    # proxies
    num_paths = int(T.get("num_paths", -1))
    L_empty = float(T.get("baseline_leakage_empty_mask", 0.0))
    paths_blocked = 0
    if num_paths > 0 and np.isfinite(L_empty) and L_empty > 1e-15 and np.isfinite(leakage):
        frac = 1.0 - (float(leakage) / float(L_empty))
        frac = float(max(0.0, min(1.0, frac)))
        paths_blocked = int(round(frac * num_paths))

    # del_time is 0 here (runner measures update_time)
    del_time = 0.0

    memory_overhead_bytes = deep_sizeof(T) + deep_sizeof(mask_set)

    # instantiated cells proxy: size of zone (direct neighbors)
    num_instantiated_cells = int(len(T.get("zone", [])))

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": float(del_time),

        "leakage": float(leakage),
        "utility": float(utility),
        "mask_size": int(len(mask_set)),
        "mask": set(mask_set),

        "num_paths": int(num_paths),
        "paths_blocked": int(paths_blocked),

        "memory_overhead_bytes": int(memory_overhead_bytes),
        "num_instantiated_cells": int(num_instantiated_cells),
    }
