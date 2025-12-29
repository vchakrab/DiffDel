#!/usr/bin/env python3
"""
greedy_gumbel.py  (DROP-IN)

Fixes / guarantees:
- Uses SAME inferable leakage model as delexp (NO path enumeration).
- Greedy Gumbel-Max selection of mask (K steps).
- Returns NON-NEGATIVE num_paths + paths_blocked proxy.
- Does NOT update the DB (runner measures update_to_null time consistently).
- Memory_overhead_bytes is MEDIUM:
  counts hypergraph + inferable model + mask (NO candidate mask enumeration).

UPDATED (per your table):
- Replaces alpha/beta with lambda (λ).
- Marginal utility:
    Δu(c) = λ·(L_curr - L_new) - (1-λ)/( |I(c*)| - 1 )
- Gumbel scale uses sensitivity 2λ:
    scale = (2λ) / ε'
- Input parameters: λ, ε, K (plus your existing l for final utility reporting).
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence, Tuple
from itertools import chain, combinations
from collections import deque
import importlib

import numpy as np

import weights.weights_corrected.adult_weights
from weights.weights_corrected import *  # dataset WEIGHTS live here


# ============================================================
# Shared helpers (same style as exponential_deletion.py)
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


def clean_raw_dcs(raw_dcs: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
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
# Inferable leakage model (same as delexp)
# ============================================================

def get_dataset_weights(dataset: str):
    dataset = dataset.lower()
    module_name = f"weights.weights_corrected.{dataset}_weights"
    try:
        weights_module = importlib.import_module(module_name)
        return weights_module.WEIGHTS
    except ModuleNotFoundError:
        print(f"[WARN] No weights module found for dataset: {dataset}")
        return None


class InferableLeakageModel:
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

        self.edges: List[Tuple[Tuple[int, ...], float]] = []
        for s, w in zip(hedges, W):
            ww = float(w)
            if not (0.0 < ww <= 1.0):
                ww = min(1.0, max(1e-12, ww))
            verts = tuple(sorted(self.cell_to_id[c] for c in s))
            self.edges.append((verts, ww))

        self.inc: List[List[int]] = [[] for _ in range(self.n)]
        for ei, (verts, _w) in enumerate(self.edges):
            for v in verts:
                self.inc[v].append(ei)

        neigh_sets: List[Set[int]] = [set() for _ in range(self.n)]
        for verts, _w in self.edges:
            for v in verts:
                neigh_sets[v].update(verts)
        for v in range(self.n):
            neigh_sets[v].discard(v)
        self.neigh: List[Tuple[int, ...]] = [tuple(sorted(s)) for s in neigh_sets]

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

        L = float(max(0.0, min(1.0, self._compute_L(p))))
        return L


def gumbel_noise(scale: float) -> float:
    u = random.random()
    u = max(1e-10, min(1.0 - 1e-10, u))
    return float(-scale * np.log(-np.log(u)))


def inference_zone_union(target: str, hyperedges: Sequence[Iterable[str]]) -> List[str]:
    z: Set[str] = set()
    for e in hyperedges:
        z |= set(e)
    z.discard(target)
    return sorted(z)


# ============================================================
# Gumbel greedy selection (UPDATED for lambda mechanism)
# ============================================================

def marginal_gain(
    *,
    c: str,
    M_curr: Set[str],
    model: InferableLeakageModel,
    lam: float,
    denom_I_minus_1: int,
) -> Tuple[float, float, float]:
    """
    NEW:
      Δu(c) = λ·(L_curr - L_new) - (1-λ)/( |I(c*)| - 1 )

    denom_I_minus_1 is computed once per run from the target's inference zone size:
      denom_I_minus_1 = max(1, len(I) - 1)
    """
    L_curr = model.leakage(M_curr)
    L_new = model.leakage(M_curr | {c})

    penalty = (1.0 - float(lam)) / float(denom_I_minus_1)
    delta_u = float(lam) * (L_curr - L_new) - penalty
    return float(delta_u), float(L_curr), float(L_new)


def greedy_gumbel_max_deletion(
    *,
    model: InferableLeakageModel,
    hyperedges: List[Tuple[str, ...]],
    target_cell: str,
    lam: float,
    epsilon: float,
    K: int,
) -> Tuple[Set[str], float]:
    """
    NEW input params: (λ, ε, K)

    Gumbel scale per table:
      ε' = ε / K
      scale = (2λ) / ε'
    """
    t0 = time.time()
    I = set(inference_zone_union(target_cell, hyperedges))
    M: Set[str] = set()

    if K <= 0 or epsilon <= 0:
        return M, float(time.time() - t0)

    # denom from your new formula (|I(c*)| - 1)
    denom_I_minus_1 = max(1, len(I) - 1)

    epsilon_prime = float(epsilon) / float(K)

    # NEW gumbel scale: 2λ/ε'
    g_scale = (2.0 * float(lam)) / max(1e-12, epsilon_prime)

    for _k in range(1, K + 1):
        candidates = list(I - M)
        if not candidates:
            break

        best_c = None
        best_score = -1e300

        for c in candidates:
            delta_u, _Lc, _Ln = marginal_gain(
                c=c,
                M_curr=M,
                model=model,
                lam=lam,
                denom_I_minus_1=denom_I_minus_1,
            )
            score = float(delta_u) + gumbel_noise(g_scale)
            if score > best_score:
                best_score = score
                best_c = c

        # Stop decision: same gumbel scale (table gives one scale; old code used a different stop-scale)
        s_stop = gumbel_noise(g_scale)
        if best_c is None:
            break
        if s_stop > best_score:
            break
        M.add(best_c)

    return M, float(time.time() - t0)


# ============================================================
# Memory estimate (MEDIUM for delgum)
# ============================================================

def estimate_memory_overhead_bytes_delgum(
    *,
    hyperedges: List[Tuple[str, ...]],
    num_vertices: int,
    mask_size: int,
    includes_channel_map: bool = True,
) -> int:
    num_edges = len(hyperedges)
    edge_members = sum(len(e) for e in hyperedges)

    BYTES_PER_VERTEX = 112
    BYTES_PER_EDGE = 184
    BYTES_PER_EDGE_MEMBER = 72
    BYTES_PER_MASK_SET = 96
    BYTES_PER_MASK_MEMBER = 72
    BYTES_PER_EDGE_STRUCT = 80
    BYTES_PER_FLOAT = 8
    BYTES_PER_INT = 28

    est = 0
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER

    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    # inferable model footprint
    est += num_edges * BYTES_PER_EDGE_STRUCT
    est += num_vertices * BYTES_PER_FLOAT
    est += num_edges * BYTES_PER_FLOAT

    if includes_channel_map:
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    return int(est)


def estimate_paths_proxy_from_channels(*, num_channel_edges: int, L_empty: float, L_mask: float) -> Dict[str, int]:
    total = int(max(0, num_channel_edges))
    if total == 0 or not np.isfinite(L_empty) or L_empty <= 1e-15 or not np.isfinite(L_mask):
        return {"num_paths_est": total, "paths_blocked_est": 0}
    frac = 1.0 - (float(L_mask) / float(L_empty))
    frac = float(max(0.0, min(1.0, frac)))
    blocked = int(round(frac * total))
    blocked = int(max(0, min(total, blocked)))
    return {"num_paths_est": total, "paths_blocked_est": blocked}


# ============================================================
# Main orchestrator (NO DB writes)
# ============================================================

def gumbel_deletion_main(
    dataset: str,
    key: Any,
    target_cell: str,
    *,
    epsilon: float = 1.0,
    lam: float = 0.5,     # NEW: lambda for gumbel mechanism
    K: int = 40,
) -> Dict[str, Any]:
    # -------------------------
    # INIT (timed)
    # -------------------------
    init_start = time.time()
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

    H_raw = clean_raw_dcs(raw_dcs)
    H = _normalize_hyperedges(H_raw)
    W = _normalize_edge_weights(H, get_dataset_weights(dataset))

    instantiated: Set[str] = set()
    for e in H:
        instantiated.update(e)
    num_instantiated_cells = int(len(instantiated))

    init_time = float(time.time() - init_start)

    # -------------------------
    # MODEL (timed)
    # -------------------------
    model_start = time.time()

    model = InferableLeakageModel(H, W, target=target_cell)

    final_mask, _greedy_time = greedy_gumbel_max_deletion(
        model=model,
        hyperedges=H,
        target_cell=target_cell,
        lam=lam,
        epsilon=epsilon,
        K=K,
    )

    inference_zone = inference_zone_union(target_cell, H)
    leakage = float(model.leakage(final_mask))

    # keep your existing utility reporting formula
    denom = max(1, len(inference_zone))
    utility = float(-1 * lam * leakage - ((1 - lam) * len(final_mask)) / denom)

    L_empty = float(model.leakage(set()))
    num_channel_edges = int(len(model.channel_edges))
    paths_proxy = estimate_paths_proxy_from_channels(
        num_channel_edges=num_channel_edges,
        L_empty=L_empty,
        L_mask=leakage,
    )

    model_time = float(time.time() - model_start)

    # -------------------------
    # DEL (NOT DONE HERE)
    # -------------------------
    del_time = 0.0

    # -------------------------
    # MEMORY (report-only, MEDIUM)
    # -------------------------
    memory_overhead = estimate_memory_overhead_bytes_delgum(
        hyperedges=H,
        num_vertices=int(model.n),
        mask_size=len(final_mask),
        includes_channel_map=True,
    )

    return {
        "init_time": init_time,
        "model_time": model_time,
        "del_time": del_time,

        "leakage": float(leakage),
        "utility": float(utility),
        "mask_size": int(len(final_mask)),
        "mask": set(final_mask),

        "num_paths": int(paths_proxy["num_paths_est"]),
        "paths_blocked": int(paths_proxy["paths_blocked_est"]),

        "memory_overhead_bytes": int(memory_overhead),
        "num_instantiated_cells": int(num_instantiated_cells),
        "num_channel_edges": int(num_channel_edges),
        "baseline_leakage_empty_mask": float(L_empty),

        # helpful to log / verify the new mechanism
        "lambda": float(lam),
        "denom_I_minus_1": int(max(1, len(set(inference_zone)) - 1)),
        "inference_zone_size": int(len(set(inference_zone))),
    }
