#!/usr/bin/env python3
"""
exponential_deletion.py (EXACT SCHEMA - UPDATED LEAKAGE LOGIC)

Matches the full diagnostic output of your original script while
fixing the Leakage Model to strictly follow Algorithm 1 (MPE/Max inference).
"""

from __future__ import annotations

import time
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
# Leakage model EXACTLY as specified in Algorithm 1
# ============================================================

class MaskedMPELeakageModel:
    """
    Implements the leakage computation exactly as described in the user's text:
    - Observations: O(M) = V \ (M ∪ {c*})
    - Inferability: Pr(x) = max_{e ∋ x, e ∉ E*} w(e) * Π_{y in e\{x}} Pr(y)
    - Monotone fixpoint iteration with a worklist.
    - Leakage: L(M) = (1/|E*|) Σ_{e in E*} w*_e(M) / w*_e(∅)
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
            # Clamp to [0, 1] as per model assumptions
            ww = max(0.0, min(1.0, ww))
            verts = tuple(sorted(self.cell_to_id[c] for c in s))
            self.edges.append((verts, ww))

        self.inc: List[List[int]] = [[] for _ in range(self.n)]
        neigh_sets: List[Set[int]] = [set() for _ in range(self.n)]
        for ei, (verts, _w) in enumerate(self.edges):
            for v in verts:
                self.inc[v].append(ei)
                neigh_sets[v].update(verts)

        self.neigh: List[Tuple[int, ...]] = [tuple(sorted(s - {i})) for i, s in
                                             enumerate(neigh_sets)]

        self.channel_edges: List[int] = [
            ei for ei, (verts, _w) in enumerate(self.edges) if self.tid in verts
        ]
        self._channel_edge_set: Set[int] = set(self.channel_edges)

    def _prod_except(self, verts: Tuple[int, ...], skip_v: int, pr: List[float]) -> float:
        out = 1.0
        for y in verts:
            if y == skip_v: continue
            out *= pr[y]
            if out == 0.0: return 0.0
        return out

    def leakage_with_diagnostics(
            self,
            mask: Set[str],
            *,
            max_updates: int = 10_000_000,
    ) -> Tuple[float, List[float], Dict[int, float]]:
        mask_ids = {self.cell_to_id[c] for c in mask if c in self.cell_to_id}

        # O(M) = V \ (M ∪ {c*})
        observed = [True] * self.n
        observed[self.tid] = False
        for mid in mask_ids:
            observed[mid] = False

        # Pr(x) = 1 if x in O else 0
        pr = [1.0 if observed[v] else 0.0 for v in range(self.n)]
        Q = deque(range(self.n))
        in_q = [True] * self.n
        pops = 0

        while Q and pops < max_updates:
            v = Q.popleft()
            in_q[v] = False
            pops += 1
            if observed[v]: continue

            # Pr_new(v) = max_{e ∋ v, e ∉ E*} w(e) * Π Pr(y)
            best = 0.0
            for ei in self.inc[v]:
                if ei in self._channel_edge_set: continue
                verts, w = self.edges[ei]
                candidate = w * self._prod_except(verts, v, pr)
                if candidate > best:
                    best = candidate

            if best > pr[v]:
                pr[v] = best
                for u in self.neigh[v]:
                    if not in_q[u]:
                        Q.append(u)
                        in_q[u] = True

        channel_wstar: Dict[int, float] = {}
        if not self.channel_edges:
            return 0.0, pr, {}

        acc = 0.0
        for ei in self.channel_edges:
            verts, w = self.edges[ei]
            # w*_e(M) = w(e) * Π Pr(y)
            prod = self._prod_except(verts, self.tid, pr)
            w_star_M = w * prod
            channel_wstar[ei] = w_star_M

            # Ratio = w*_e(M) / w*_e(empty). Since w*_e(empty) = w(e):
            if w > 0:
                acc += (w_star_M / w)
            else:
                acc += 0.0

        L = acc / len(self.channel_edges)
        return float(L), pr, channel_wstar

    def leakage(self, mask: Set[str]) -> float:
        L, _, _ = self.leakage_with_diagnostics(mask)
        return L


# ============================================================
# Utilities, Paths Proxy, Memory Estimate (ORIGINAL LOGIC)
# ============================================================

def enumerate_masks_powerset(neigh: List[str]) -> List[Set[str]]:
    out: List[Set[str]] = []
    for k in range(len(neigh) + 1):
        for comb in combinations(neigh, k):
            out.append(set(comb))
    return out


def compute_utility(mask: Set[str], leakage: float, lam: float,
                    num_candidates_minus_one: int) -> float:
    norm = (len(mask) / num_candidates_minus_one) if num_candidates_minus_one > 0 else 0.0
    return float(-(lam * leakage) - ((1.0 - lam) * norm))


def estimate_paths_proxy_from_channels(num_channel_edges: int, L_empty: float, L_mask: float) -> \
Dict[str, int]:
    total = int(max(0, num_channel_edges))
    if total == 0 or L_empty <= 1e-15:
        return {"num_paths_est": total, "paths_blocked_est": 0}
    frac = max(0.0, min(1.0, 1.0 - (L_mask / L_empty)))
    blocked = int(round(frac * total))
    return {"num_paths_est": total, "paths_blocked_est": blocked}


def estimate_memory_overhead_bytes_delexp(
        hyperedges: List[Tuple[str, ...]],
        num_vertices: int,
        mask_size: int,
        num_candidates: int,
        candidate_mask_members: int,  # Ensure this matches the keyword in the call
        includes_channel_map: bool = True
) -> int:
    num_edges = len(hyperedges)
    edge_members = sum(len(e) for e in hyperedges)

    # Constants for object sizes in bytes
    B_V, B_E, B_EM, B_MS, B_MM, B_ES, B_F, B_I, B_CM = 112, 184, 72, 96, 72, 80, 8, 28, 96

    est = (num_vertices * B_V) + (num_edges * B_E) + (edge_members * B_EM)
    est += B_MS + (mask_size * B_MM)
    est += (num_edges * B_ES) + (num_vertices * B_F) + (num_edges * B_F)

    if includes_channel_map:
        est += num_edges * (B_I + B_F)

    est += (num_candidates * B_CM) + (candidate_mask_members * B_MM) + (num_candidates * 8)
    return int(est)


# ============================================================
# Main orchestrator
# ============================================================

def map_dc_to_weight(init_manager, dc, weights):
    return weights[init_manager.denial_constraints.index(dc)]


def dc_to_hyperedges(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    hyperedges, weights = [], []
    W = get_dataset_weights(init_manager.dataset)
    for dc in getattr(init_manager, "denial_constraints", []):
        attrs = set()
        weight = map_dc_to_weight(init_manager, dc, W) if W else 1.0
        for pred in dc:
            if isinstance(pred, (list, tuple)) and len(pred) >= 1:
                token = pred[0]
                if isinstance(token, str) and "." in token:
                    attrs.add(token.split(".")[-1])
        if len(attrs) >= 2:
            hyperedges.append(tuple(sorted(attrs)))
            weights.append(float(weight))
    return hyperedges, weights


def exponential_deletion_main(dataset: str, key: int, target_cell: str, epsilon: float = 10,
                              lam: float = 0.67) -> Dict[str, Any]:
    init_start = time.time()
    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target_cell}, dataset, 0)
    H, W = dc_to_hyperedges(init_manager)

    instantiated_cells = set()
    for e in H: instantiated_cells.update(e)
    num_instantiated_cells = len(instantiated_cells)
    init_time = time.time() - init_start

    model_start = time.time()
    model = MaskedMPELeakageModel(H, W, target = target_cell)

    # I(c*) = direct neighbors
    neigh = set()
    for e in H:
        if target_cell in e:
            for v in e:
                if v != target_cell: neigh.add(v)
    neigh_list = sorted(neigh)

    candidates = enumerate_masks_powerset(neigh_list)

    # Exponential mechanism
    denom = max(1, len(neigh_list))
    utilities = []
    for M in candidates:
        L = model.leakage(M)
        utilities.append(compute_utility(M, L, lam, denom))

    utilities = np.array(utilities)
    scores = (epsilon * utilities) / (2.0 * lam)
    probs = np.exp(scores - np.max(scores))
    probs /= probs.sum()

    idx = np.random.choice(len(candidates), p = probs)
    final_mask = candidates[idx]
    util_val = utilities[idx]

    leakage_base = model.leakage(set())
    leakage_final = model.leakage(final_mask)

    paths_proxy = estimate_paths_proxy_from_channels(
        num_channel_edges = len(model.channel_edges),
        L_empty = leakage_base,
        L_mask = leakage_final
    )

    model_time = time.time() - model_start
    cand_members = sum(len(s) for s in candidates)

    # FIXED: Keywords now match the definition signature exactly
    memory_overhead = estimate_memory_overhead_bytes_delexp(
        hyperedges = H,
        num_vertices = model.n,
        mask_size = len(final_mask),
        num_candidates = len(candidates),
        candidate_mask_members = cand_members
    )

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": 0.0,
        "leakage": float(leakage_final),
        "utility": float(util_val),
        "mask_size": int(len(final_mask)),
        "mask": set(final_mask),
        "num_paths": int(paths_proxy["num_paths_est"]),
        "paths_blocked": int(paths_proxy["paths_blocked_est"]),
        "memory_overhead_bytes": int(memory_overhead),
        "num_instantiated_cells": int(num_instantiated_cells),
        "num_channel_edges": int(len(model.channel_edges)),
        "baseline_leakage_empty_mask": float(leakage_base),
    }



if __name__ == "__main__":
    print(exponential_deletion_main("airport", 500, "latitude_deg"))
    print(exponential_deletion_main("hospital", 500, "ProviderNumber"))
    print(exponential_deletion_main("tax", 500, "marital_status"))
    print(exponential_deletion_main("adult", 500, "education"))
    print(exponential_deletion_main("Onlineretail", 500, "InvoiceNo"))
