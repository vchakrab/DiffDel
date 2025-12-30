#!/usr/bin/env python3
"""
exponential_deletion.py  (DROP-IN - FULL POWERSET, NO CANONICAL MASKS)

Changes vs the file you pasted:
1) Removed ALL canonical/frontier/reachability code.
2) Candidate masks are FULL POWERSET over direct neighbors I(c*).
3) Exponential mechanism samples OVER MASKS DIRECTLY (no canonicalization, no dedup).

WARNING: Candidate count is 2^|I(c*)|, so this will blow up quickly for large inference zones.
"""

from __future__ import annotations

import time
import math
import importlib
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence, Tuple
from itertools import combinations
from collections import deque

import numpy as np

from rtf_core import initialization_phase

# ============================================================
# Helpers
# ============================================================

def _normalize_hyperedges(hyperedges: Sequence[Iterable[str]]) -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    for e in hyperedges:
        s = tuple(sorted(set(e)))
        if len(s) >= 2:
            out.append(s)
    return out


def get_dataset_weights(dataset: str):
    dataset = dataset.lower()
    module_name = f"weights.weights_corrected.{dataset}_weights"
    try:
        weights_module = importlib.import_module(module_name)
        return weights_module.WEIGHTS
    except ModuleNotFoundError:
        print(f"[WARN] No weights module found for dataset: {dataset}")
        return None


# ============================================================
# Inferable leakage model (UNCHANGED)
# ============================================================

class InferableLeakageModel:
    """
    (Unchanged from your current file.)
    """

    def __init__(self, hyperedges: Sequence[Iterable[str]], weights: Sequence[float], target: str,
                 damping: float = 0.5):
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
        self.damping = float(damping)

        # store edges as (verts_ids, weight)
        self.edges: List[Tuple[Tuple[int, ...], float]] = []
        for s, w in zip(hedges, W):
            ww = float(w)
            if not (0.0 < ww <= 1.0):
                ww = min(1.0, max(1e-12, ww))

            if self.damping > 0:
                ww = ww ** (1.0 - self.damping)

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

        self.channel_edges: List[int] = [
            ei for ei, (verts, _w) in enumerate(self.edges) if self.tid in verts
        ]

    def _attempt(self, verts: Tuple[int, ...], w: float, infer_v: int, p: List[float]) -> float:
        if w <= 0:
            return 0.0
        if w >= 1.0:
            w = 0.9999

        log_result = math.log(w)

        num_vertices = 0
        for u in verts:
            if u == infer_v:
                continue
            num_vertices += 1
            if p[u] <= 1e-8:
                return 1e-8
            if p[u] >= 0.9999:
                continue
            log_result += math.log(p[u])

        if num_vertices > 3:
            scale_factor = math.sqrt(num_vertices - 2)
            log_result = log_result / scale_factor

        if log_result > -0.01:
            return 0.9
        if log_result < -10:
            return 0.05

        result = math.exp(log_result)
        return float(max(0.05, min(0.9, result)))

    def _recompute_pv(self, v: int, observed: List[bool], p: List[float]) -> float:
        if observed[v]:
            return 0.95

        log_prod_fail = 0.0
        has_inference = False
        num_edges = 0

        for ei in self.inc[v]:
            verts, w = self.edges[ei]
            a = self._attempt(verts, w, v, p)

            if a <= 0.01:
                continue

            has_inference = True
            num_edges += 1

            if a >= 0.95:
                return 0.95

            if a < 0.1:
                log_prod_fail += math.log1p(-a)
            else:
                log_prod_fail += math.log(max(1e-10, 1.0 - a))

        if not has_inference:
            return 0.05

        if num_edges > 5:
            log_prod_fail = log_prod_fail * (5.0 / num_edges)

        if log_prod_fail < -10:
            return 0.95

        prod_fail = math.exp(log_prod_fail)
        result = 1.0 - prod_fail
        return float(max(0.05, min(0.95, result)))

    def _compute_L(self, p: List[float]) -> float:
        t = self.tid
        log_prod_fail = 0.0
        has_channels = False

        for ei in self.channel_edges:
            verts, w = self.edges[ei]
            q = self._attempt(verts, w, t, p)

            if q <= 1e-15:
                continue
            has_channels = True
            if q >= 1.0 - 1e-15:
                return 1.0

            if q < 0.1:
                log_prod_fail += math.log1p(-q)
            else:
                log_prod_fail += math.log(1.0 - q)

        if not has_channels:
            return 1e-10

        return float(max(1e-10, min(1.0, 1.0 - math.exp(log_prod_fail))))

    def leakage_with_diagnostics(
            self,
            mask: Set[str],
            *,
            tau: float = 1e-4,
            max_updates: int = 100000
    ) -> Tuple[float, List[float], Dict[int, float]]:
        mask_ids = {self.cell_to_id[c] for c in mask if c in self.cell_to_id}
        if self.tid in mask_ids:
            raise ValueError("mask includes target")

        observed = [True] * self.n
        observed[self.tid] = False
        for mid in mask_ids:
            observed[mid] = False

        eps = 1e-10
        p = [1.0 - eps if observed[v] else eps for v in range(self.n)]

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
            channel_q[ei] = float(self._attempt(verts, w, t, p))

        L = float(self._compute_L(p))
        return L, p, channel_q

    def leakage(self, mask: Set[str]) -> float:
        L, _p, _q = self.leakage_with_diagnostics(mask)
        return float(L)


# ============================================================
# Candidate masks (FULL POWERSET) + utility + exponential mech
# ============================================================

def enumerate_masks_powerset(neigh: List[str]) -> List[Set[str]]:
    """
    Enumerate FULL POWERSET over neigh (direct neighbors of target).
    Includes empty mask and full mask.
    """
    n = len(neigh)
    out: List[Set[str]] = []
    for k in range(0, n + 1):
        for comb in combinations(neigh, k):
            out.append(set(comb))
    return out


def compute_utility(
        mask: Set[str],
        leakage: float,
        lam: float,
        num_candidates_minus_one: int
) -> float:
    # Normalize |M| by (|C|-1) like your other code path.
    norm = (len(mask) / num_candidates_minus_one) if num_candidates_minus_one > 0 else 0.0
    return float(-(lam * leakage) - ((1.0 - lam) * norm))


def exponential_mechanism_sample(
        candidates: List[Set[str]],
        *,
        model: InferableLeakageModel,
        epsilon: float,
        lam: float
) -> Tuple[Set[str], float]:
    """
    Exponential mechanism over candidates directly (NO canonicalization).
    """
    if not candidates:
        candidates = [set()]

    utilities = np.empty(len(candidates), dtype=float)
    denom = max(1, len(candidates) - 1)

    sum_L = 0.0
    for i, M in enumerate(candidates):
        L = float(model.leakage(M))
        sum_L += L
        utilities[i] = compute_utility(M, L, lam, denom)

    print("Average Leakage", sum_L / max(1, len(utilities)))

    # sensitivity = lam (matches your earlier update)
    scores = (float(epsilon) * utilities) / (2.0 * max(1e-10, float(lam)))
    max_score = float(np.max(scores)) if len(scores) else 0.0
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / np.sum(exp_scores)

    idx = int(np.random.choice(len(candidates), p=probs))
    return candidates[idx], float(utilities[idx])


# ============================================================
# "paths blocked" proxy (NO path enumeration) (UNCHANGED)
# ============================================================

def estimate_paths_proxy_from_channels(
        *,
        num_channel_edges: int,
        L_empty: float,
        L_mask: float
) -> Dict[str, int]:
    total = int(max(0, num_channel_edges))
    if total == 0 or not np.isfinite(L_empty) or L_empty <= 1e-15 or not np.isfinite(L_mask):
        return {"num_paths_est": total, "paths_blocked_est": 0}

    frac = 1.0 - (float(L_mask) / float(L_empty))
    frac = float(max(0.0, min(1.0, frac)))
    blocked = int(round(frac * total))
    blocked = int(max(0, min(total, blocked)))
    return {"num_paths_est": total, "paths_blocked_est": blocked}


# ============================================================
# Memory estimate (designed so delexp is biggest) (UNCHANGED)
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
    BYTES_PER_CAND_MASK = 96

    est = 0
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER

    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    est += num_edges * BYTES_PER_EDGE_STRUCT
    est += num_vertices * BYTES_PER_FLOAT
    est += num_edges * BYTES_PER_FLOAT

    if includes_channel_map:
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    est += num_candidate_masks * BYTES_PER_CAND_MASK
    est += candidate_mask_members * BYTES_PER_MASK_MEMBER

    est += num_candidate_masks * 8

    return int(est)


# ============================================================
# DC -> Hyperedges (UNCHANGED)
# ============================================================

def map_dc_to_weight(init_manager, dc, weights):
    return weights[init_manager.denial_constraints.index(dc)]


def dc_to_hyperedges(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    hyperedges: List[Tuple[str, ...]] = []
    hyperedge_weights: List[float] = []

    W = get_dataset_weights(init_manager.dataset)
    if W is None:
        W = []

    for dc in getattr(init_manager, "denial_constraints", []):
        attrs: Set[str] = set()
        weight = map_dc_to_weight(init_manager, dc, W) if W else 1.0
        for pred in dc:
            if isinstance(pred, (list, tuple)) and len(pred) >= 1:
                token = pred[0]
                if isinstance(token, str) and "." in token:
                    attrs.add(token.split(".")[-1])
        if len(attrs) >= 2:
            hyperedges.append(tuple(sorted(attrs)))
            hyperedge_weights.append(float(weight))

    return hyperedges, hyperedge_weights


# ============================================================
# Main orchestrator (NO DB writes)
# ============================================================

def exponential_deletion_main(
        dataset: str,
        key: int,
        target_cell: str,
        *,
        epsilon: float = 10,
        lam: float = 0.67
) -> Dict[str, Any]:
    """
    Returns same schema as your current code.
    """
    # ----------------------
    # INIT (timed)
    # ----------------------
    init_start = time.time()

    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target_cell},
        dataset,
        0
    )

    H, W = dc_to_hyperedges(init_manager)

    instantiated_cells: Set[str] = set()
    for e in H:
        instantiated_cells.update(e)
    num_instantiated_cells = int(len(instantiated_cells))

    init_time = float(time.time() - init_start)

    # ----------------------
    # MODEL (timed)
    # ----------------------
    model_start = time.time()

    model = InferableLeakageModel(H, W, target=target_cell)

    # Direct neighbors = inference zone I(c*)
    neigh: Set[str] = set()
    for e in H:
        if target_cell in e:
            for v in e:
                if v != target_cell:
                    neigh.add(v)
    neigh_list = sorted(neigh)

    # FULL POWERSET candidates
    candidates = enumerate_masks_powerset(neigh_list)

    print(f"DEBUG: [delexp] Direct neighbors for '{target_cell}': {neigh_list}")
    print(f"DEBUG: [delexp] Candidate masks (FULL POWERSET) count: {len(candidates)}")
    print(f"DEBUG: [delexp] Instantiated cells (all in DCs): {num_instantiated_cells}")

    # Sample directly over masks
    final_mask, util_val = exponential_mechanism_sample(
        candidates,
        model=model,
        epsilon=epsilon,
        lam=lam
    )

    leakage_base = float(model.leakage(set()))
    leakage = float(model.leakage(final_mask))
    print(leakage_base, leakage, leakage / leakage_base if leakage_base > 0 else float("inf"))

    # proxy paths
    L_empty = float(model.leakage(set()))
    num_channel_edges = int(len(model.channel_edges))
    paths_proxy = estimate_paths_proxy_from_channels(
        num_channel_edges=num_channel_edges,
        L_empty=L_empty,
        L_mask=leakage
    )

    model_time = float(time.time() - model_start)
    del_time = 0.0

    # memory estimate based on candidates list
    cand_members = int(sum(len(s) for s in candidates))
    memory_overhead = estimate_memory_overhead_bytes_delexp(
        hyperedges=H,
        num_vertices=int(model.n),
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


if __name__ == "__main__":
    print(exponential_deletion_main("airport", 500, "latitude_deg"))
    print(exponential_deletion_main("hospital", 500, "ProviderNumber"))
    print(exponential_deletion_main("tax", 500, "marital_status"))
    print(exponential_deletion_main("adult", 500, "education"))
    print(exponential_deletion_main("Onlineretail", 500, "InvoiceNo"))
