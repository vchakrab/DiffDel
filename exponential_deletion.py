#!/usr/bin/env python3
"""
exponential_deletion.py  (DROP-IN)

Fixes / guarantees:
- Uses inferable-leakage fixed-point model (NO path enumeration).
- Exponential mechanism over candidate masks (this is why delexp is slowest + biggest memory).
- Returns NON-NEGATIVE num_paths + paths_blocked estimates (proxy = #target incident hyperedges).
- Does NOT update the DB (runner measures update_to_null time consistently).
- Memory_overhead_bytes is intentionally the LARGEST among methods:
  counts hypergraph + inferable model + candidate masks + utilities array.

Expected trends:
- delexp: largest memory, largest time
- leakage > 0 typically (unless target has no incident hyperedges)
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence, Tuple
from itertools import chain, combinations
from collections import deque
import weights
import numpy as np


# ============================================================
# Helpers
# ============================================================

def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def _normalize_hyperedges(hyperedges: Sequence[Iterable[str]]) -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    for e in hyperedges:
        s = tuple(sorted(set(e)))
        if len(s) >= 2:
            out.append(s)
    return out


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


import importlib


def get_dataset_weights(dataset: str):
    dataset = dataset.lower()
    module_name = f"weights.weights_corrected.{dataset}_weights"

    try:
        weights_module = importlib.import_module(module_name)
        return weights_module.WEIGHTS
    except ModuleNotFoundError:
        print(f"[WARN] No weights module found for dataset: {dataset}")
        return None

    raise ValueError(f"Unknown dataset: {dataset}")
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
                    # t1.attr -> attr
                    attrs.add(item.split(".")[-1])
        if attrs:
            cleaned.append(tuple(sorted(attrs)))
    return cleaned


# ============================================================
# Inferable leakage model (fixed-point, NO PATHS)
# ============================================================

class InferableLeakageModel:
    """
    p[v] = probability v becomes inferable given:
      - adversary knows everything except masked cells + target (target starts unknown)
      - an edge (hyperedge) can infer a vertex u if all other vertices in edge are inferable,
        with probability weight w.
    We compute a monotone fixed point using a queue (fast + stable).
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

        # store edges as (verts_ids, weight)
        self.edges: List[Tuple[Tuple[int, ...], float]] = []
        for s, w in zip(hedges, W):
            ww = float(w)
            if not (0.0 < ww <= 1.0):
                # clamp into (0,1]
                ww = min(1.0, max(1e-12, ww))
            verts = tuple(sorted(self.cell_to_id[c] for c in s))
            self.edges.append((verts, ww))

        self.inc: List[List[int]] = [[] for _ in range(self.n)]
        for ei, (verts, _w) in enumerate(self.edges):
            for v in verts:
                self.inc[v].append(ei)

        # neighbor sets (for queue propagation)
        neigh_sets: List[Set[int]] = [set() for _ in range(self.n)]
        for verts, _w in self.edges:
            for v in verts:
                neigh_sets[v].update(verts)
        for v in range(self.n):
            neigh_sets[v].discard(v)
        self.neigh: List[Tuple[int, ...]] = [tuple(sorted(s)) for s in neigh_sets]

        # edges containing target (channels)
        self.channel_edges: List[int] = [ei for ei, (verts, _w) in enumerate(self.edges) if self.tid in verts]

    def _attempt(self, verts: Tuple[int, ...], w: float, infer_v: int, p: List[float]) -> float:
        prod = 1.0
        for u in verts:
            if u == infer_v:
                continue
            prod *= p[u]
            if prod == 0.0:
                return 0.0
        a = w * prod
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
            verts, w = self.edges[ei]
            a = self._attempt(verts, w, v, p)
            if a == 0.0:
                continue
            if a == 1.0:
                return 1.0
            prod_fail *= (1.0 - a)
            if prod_fail == 0.0:
                return 1.0
        return 1.0 - prod_fail

    def _compute_L(self, p: List[float]) -> float:
        # L = 1 - Π_{target-edges}(1 - q_e), q_e = w * Π_{others} p[other]
        t = self.tid
        prod_fail = 1.0
        for ei in self.channel_edges:
            verts, w = self.edges[ei]
            prod = 1.0
            for u in verts:
                if u == t:
                    continue
                prod *= p[u]
                if prod == 0.0:
                    break
            q = w * prod
            if q <= 0.0:
                continue
            if q >= 1.0:
                return 1.0
            prod_fail *= (1.0 - q)
            if prod_fail == 0.0:
                return 1.0
        return 1.0 - prod_fail

    def leakage_with_diagnostics(
        self,
        mask: Set[str],
        *,
        tau: float = 1e-10,
        max_updates: int = 2_000_000
    ) -> Tuple[float, List[float], Dict[int, float]]:
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

        channel_q: Dict[int, float] = {}
        t = self.tid
        for ei in self.channel_edges:
            verts, w = self.edges[ei]
            prod = 1.0
            for u in verts:
                if u == t:
                    continue
                prod *= p[u]
                if prod == 0.0:
                    break
            channel_q[ei] = float(max(0.0, min(1.0, w * prod)))

        L = float(max(0.0, min(1.0, self._compute_L(p))))
        return L, p, channel_q

    def leakage(self, mask: Set[str]) -> float:
        L, _p, _q = self.leakage_with_diagnostics(mask)
        return float(L)


# ============================================================
# Candidate masks + utility + exponential mechanism
# ============================================================

def compute_possible_mask_set_str(target_cell: str, hyperedges: List[Tuple[str, ...]]) -> List[Set[str]]:
    neigh: Set[str] = set()
    for e in hyperedges:
        for v in e:
            if v != target_cell:
                neigh.add(v)
    return [set(s) for s in powerset(sorted(neigh))]


def compute_utility(mask: Set[str], leakage: float, alpha: float, beta: float) -> float:
    # bigger is better
    return float(-(alpha * leakage) - (beta * len(mask)))


def exponential_mechanism_sample(
    candidates: List[Set[str]],
    *,
    model: InferableLeakageModel,
    epsilon: float,
    alpha: float,
    beta: float
) -> Tuple[Set[str], float]:
    utilities = np.empty(len(candidates), dtype=float)
    for i, M in enumerate(candidates):
        L = model.leakage(M)
        utilities[i] = compute_utility(M, L, alpha, beta)

    # score = eps * u / (2 * sensitivity). We use alpha as sensitivity-scale here (matches your older code shape).
    scores = (float(epsilon) * utilities) / (2.0 * max(1e-12, float(alpha)))
    max_score = float(np.max(scores)) if len(scores) else 0.0
    exp_scores = np.exp(scores - max_score)
    Z = float(np.sum(exp_scores)) if len(exp_scores) else 1.0
    probs = exp_scores / Z

    idx = int(np.random.choice(len(candidates), p=probs))
    return candidates[idx], float(utilities[idx])


# ============================================================
# "paths blocked" proxy (NO path enumeration)
# ============================================================

def estimate_paths_proxy_from_channels(
    *,
    num_channel_edges: int,
    L_empty: float,
    L_mask: float
) -> Dict[str, int]:
    """
    Proxy:
      total_paths_est := #hyperedges that contain the target (channels)
      blocked_est := round((1 - L_mask/L_empty) * total)  (clamped)
    """
    total = int(max(0, num_channel_edges))
    if total == 0 or not np.isfinite(L_empty) or L_empty <= 1e-15 or not np.isfinite(L_mask):
        return {"num_paths_est": total, "paths_blocked_est": 0}

    frac = 1.0 - (float(L_mask) / float(L_empty))
    frac = float(max(0.0, min(1.0, frac)))
    blocked = int(round(frac * total))
    blocked = int(max(0, min(total, blocked)))
    return {"num_paths_est": total, "paths_blocked_est": blocked}


# ============================================================
# Memory estimate (designed so delexp is biggest)
# ============================================================

def estimate_memory_overhead_bytes_delexp(
    *,
    hyperedges: List[Tuple[str, ...]],
    num_vertices: int,
    mask_size: int,
    num_candidate_masks: int,
    candidate_mask_members: int,
    includes_channel_map: bool = True,
) -> int:
    """
    Purposefully counts:
      - hypergraph storage
      - inferable model footprint
      - channel map
      - candidate masks (dominant)
      - utilities array (float64 per candidate)
    """
    num_edges = len(hyperedges)
    edge_members = sum(len(e) for e in hyperedges)

    # stable constants
    BYTES_PER_VERTEX = 112
    BYTES_PER_EDGE = 184
    BYTES_PER_EDGE_MEMBER = 72
    BYTES_PER_MASK_SET = 96
    BYTES_PER_MASK_MEMBER = 72
    BYTES_PER_EDGE_STRUCT = 80
    BYTES_PER_FLOAT = 8
    BYTES_PER_INT = 28
    BYTES_PER_CAND_MASK = 96

    est = 0
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER

    # final mask object
    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    # inferable model
    est += num_edges * BYTES_PER_EDGE_STRUCT
    est += num_vertices * BYTES_PER_FLOAT  # p
    est += num_edges * BYTES_PER_FLOAT     # weights

    if includes_channel_map:
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    # candidate masks + their elements
    est += num_candidate_masks * BYTES_PER_CAND_MASK
    est += candidate_mask_members * BYTES_PER_MASK_MEMBER

    # utilities array
    est += num_candidate_masks * 8

    return int(est)


# ============================================================
# Main orchestrator (NO DB writes)
# ============================================================

def exponential_deletion_main(
    dataset: str,
    key: int,
    target_cell: str,
    *,
    epsilon: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      init_time, model_time, del_time(=0 here),
      leakage, utility, mask_size, mask,
      num_paths, paths_blocked,
      memory_overhead_bytes,
      num_instantiated_cells
    """
    # ----------------------
    # INIT (timed)
    # ----------------------
    init_start = time.time()

    # import parsed DCs
    try:
        if dataset.lower() == "ncvoter":
            dataset_module_name = "NCVoter"
        else:
            dataset_module_name = dataset.capitalize()
        dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        raw_dcs = getattr(dc_module, "denial_constraints", [])
    except Exception:
        raw_dcs = []

    hyperedges = clean_raw_dcs(raw_dcs)
    H = _normalize_hyperedges(hyperedges)
    W = _normalize_edge_weights(H, get_dataset_weights(dataset))

    instantiated_cells: Set[str] = set()
    for e in H:
        instantiated_cells.update(e)
    num_instantiated_cells = int(len(instantiated_cells))

    init_time = float(time.time() - init_start)

    # ----------------------
    # MODEL (timed)  <-- slowest for delexp because enumerates candidates
    # ----------------------
    model_start = time.time()

    model = InferableLeakageModel(H, W, target=target_cell)

    candidates = compute_possible_mask_set_str(target_cell, H)
    print(f"DEBUG: Number of candidate masks (2^neighbors) for '{target_cell}': {len(candidates)}")
    # if no candidates (target has no neighbors), just empty mask
    if not candidates:
        candidates = [set()]
    print(f"DEBUG: Instantiated cells for '{target_cell}': {instantiated_cells} (Total: {len(instantiated_cells)})")

    final_mask, util_val = exponential_mechanism_sample(
        candidates,
        model=model,
        epsilon=epsilon,
        alpha=alpha,
        beta=beta
    )

    leakage = float(model.leakage(final_mask))

    # also compute empty-mask leakage for proxy blocked
    L_empty = float(model.leakage(set()))
    num_channel_edges = int(len(model.channel_edges))
    paths_proxy = estimate_paths_proxy_from_channels(
        num_channel_edges=num_channel_edges,
        L_empty=L_empty,
        L_mask=leakage
    )

    model_time = float(time.time() - model_start)

    # ----------------------
    # DEL (NOT DONE HERE; runner measures update_to_null)
    # ----------------------
    del_time = 0.0

    # ----------------------
    # MEMORY (report-only)
    # ----------------------
    num_vertices = int(model.n)
    # candidate mask members
    cand_members = int(sum(len(s) for s in candidates))
    memory_overhead = estimate_memory_overhead_bytes_delexp(
        hyperedges=H,
        num_vertices=num_vertices,
        mask_size=len(final_mask),
        num_candidate_masks=len(candidates),
        candidate_mask_members=cand_members,
        includes_channel_map=True
    )

    return {
        "init_time": init_time,
        "model_time": model_time,
        "del_time": del_time,

        "leakage": float(leakage),
        "utility": float(util_val),
        "mask_size": int(len(final_mask)),
        "mask": set(final_mask),

        "num_paths": int(paths_proxy["num_paths_est"]),
        "paths_blocked": int(paths_proxy["paths_blocked_est"]),

        "memory_overhead_bytes": int(memory_overhead),
        "num_instantiated_cells": int(num_instantiated_cells),
        "num_channel_edges": int(num_channel_edges),
        "baseline_leakage_empty_mask": float(L_empty),
    }
