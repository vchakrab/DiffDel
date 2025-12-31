#!/usr/bin/env python3
"""
exponential_deletion.py (OR-BLOCKING LEAKAGE MODEL) + WEIGHT DAMPENING

Replaces MaskedMPELeakageModel with OR-blocking semantics:
- A channel is blocked if ANY prerequisite has Pr = 0
- Degradation: masked cells with inference paths have 0 < Pr < 1
- Leakage = mean of (w*_e / w_e) across all channels

NEW:
- Weight dampening: compress weights toward a center (default 0.5)
  to reduce brittleness where products go to 0 too easily.
"""

from __future__ import annotations

import time
import importlib
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence, Tuple
from itertools import combinations
from collections import defaultdict

import numpy as np

from rtf_core import initialization_phase


# ============================================================
# USER TUNABLES (Weight Dampening)
# ============================================================

# Dampening strength in [0, 1].
# 0.0 => no change. 1.0 => all weights become WEIGHT_DAMPEN_CENTER.
WEIGHT_DAMPEN_ALPHA = 0.35

# The "center" we dampen toward (usually 0.5).
WEIGHT_DAMPEN_CENTER = 0.5

# Clip weights to avoid exact 0 or 1, which can make products collapse / saturate.
WEIGHT_CLIP_MIN = 1e-6
WEIGHT_CLIP_MAX = 1.0 - 1e-6


def dampen_weight(w: float) -> float:
    """
    Compress weight toward a center:
        w' = (1-a)*w + a*center
    then clip away from {0,1}.
    """
    ww = float(w)
    # First clamp to [0,1] (raw safety)
    ww = max(0.0, min(1.0, ww))

    a = float(WEIGHT_DAMPEN_ALPHA)
    c = float(WEIGHT_DAMPEN_CENTER)

    if a > 0.0:
        ww = (1.0 - a) * ww + a * c

    # Final clip away from exact 0/1
    ww = max(WEIGHT_CLIP_MIN, min(WEIGHT_CLIP_MAX, ww))
    return ww


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
# OR-Blocking Leakage Model
# ============================================================

class ORBlockingLeakageModel:
    """
    OR-blocking leakage computation:
    - Channels: edges containing target c*
    - Prerequisites: other cells in each channel
    - Inferability via fixpoint iteration (product semantics)
    - Leakage: mean ratio of effective/baseline weights across channels
    """

    def __init__(self, hyperedges: Sequence[Iterable[str]], weights: Sequence[float], target: str):
        H = _normalize_hyperedges(hyperedges)
        W = list(weights)
        if len(H) != len(W):
            raise ValueError("hyperedges and weights must have same length")

        # Build cell set and normalize
        V: Set[str] = {target}
        hedges: List[Set[str]] = []
        for e in H:
            s = set(e)
            V |= s
            hedges.append(s)

        self.target = target
        self.edges: List[Tuple[Set[str], float]] = []

        for s, w in zip(hedges, W):
            # Apply dampening + clipping
            ww = dampen_weight(w)
            self.edges.append((s, ww))

        # Identify channels (edges containing target)
        self.channels: List[Tuple[Set[str], float]] = [
            (e, w) for e, w in self.edges if target in e
        ]
        self.non_channels: List[Tuple[Set[str], float]] = [
            (e, w) for e, w in self.edges if target not in e
        ]

        # Build index: cell -> list of (edge, weight) for non-channel edges
        self.cell_to_edges: Dict[str, List[Tuple[Set[str], float]]] = defaultdict(list)
        for e, w in self.non_channels:
            for cell in e:
                self.cell_to_edges[cell].append((e, w))

        # All cells in the hypergraph
        self.all_cells: Set[str] = set()
        for e, _w in self.edges:
            self.all_cells.update(e)

    def _product(self, values) -> float:
        """Compute product of values."""
        result = 1.0
        for v in values:
            result *= v
            if result == 0.0:
                return 0.0
        return result

    def _compute_inferability(self, mask: Set[str], relevant_cells: Set[str]) -> Dict[str, float]:
        """
        Compute inferability Pr(x) for all cells via fixpoint iteration.

        Args:
            mask: set of masked cells
            relevant_cells: cells we need inferability for (kept for interface; we compute all)

        Returns:
            dict mapping cell -> inferability in [0, 1]
        """
        # Initialize: visible cells have Pr=1, masked cells have Pr=0
        pr: Dict[str, float] = {}
        for cell in self.all_cells:
            pr[cell] = 0.0 if cell in mask else 1.0

        # Fixpoint iteration for masked cells
        changed = True
        max_iterations = len(self.all_cells) * len(self.all_cells) + 1  # Safety bound
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for x in mask:
                best_prob = pr.get(x, 0.0)

                # Try to infer x via each non-channel edge containing x
                for e, w in self.cell_to_edges.get(x, []):
                    other_cells = e - {x}
                    prob = w * self._product(pr.get(y, 0.0) for y in other_cells)

                    if prob > best_prob:
                        best_prob = prob

                if best_prob > pr.get(x, 0.0):
                    pr[x] = best_prob
                    changed = True

        return pr

    def leakage_with_diagnostics(
            self,
            mask: Set[str],
            *,
            max_updates: int = 10_000_000,
    ) -> Tuple[float, Dict[str, float], Dict[int, float]]:
        """
        Compute leakage with full diagnostics.

        Returns:
            (leakage, inferability_dict, channel_weights_dict)
        """
        if self.target in mask:
            raise ValueError("Target cell cannot be in mask")

        if not self.channels:
            # No channels = no leakage
            return 0.0, {}, {}

        # Get all prerequisites across all channels
        all_prerequisites: Set[str] = set()
        for channel, _ in self.channels:
            prerequisites = channel - {self.target}
            all_prerequisites.update(prerequisites)

        pr = self._compute_inferability(mask, all_prerequisites)

        total_ratio = 0.0
        channel_wstar: Dict[int, float] = {}

        for channel_idx, (channel, channel_weight) in enumerate(self.channels):
            prerequisites = channel - {self.target}
            w_star = channel_weight * self._product(pr.get(y, 0.0) for y in prerequisites)
            channel_wstar[channel_idx] = w_star

            w_baseline = channel_weight
            ratio = w_star / w_baseline if w_baseline > 0 else 0.0
            total_ratio += ratio

        leakage = total_ratio / len(self.channels)
        return float(leakage), pr, channel_wstar

    def leakage(self, mask: Set[str]) -> float:
        """Compute leakage for given mask."""
        L, _, _ = self.leakage_with_diagnostics(mask)
        return L


# ============================================================
# Utilities, Paths Proxy, Memory Estimate
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


def estimate_paths_proxy_from_channels(num_channel_edges: int, L_empty: float, L_mask: float) -> Dict[str, int]:
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
        candidate_mask_members: int,
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
    for e in H:
        instantiated_cells.update(e)
    num_instantiated_cells = len(instantiated_cells)
    init_time = time.time() - init_start

    model_start = time.time()
    model = ORBlockingLeakageModel(H, W, target=target_cell)

    # I(c*) = direct neighbors
    neigh = set()
    for e in H:
        if target_cell in e:
            for v in e:
                if v != target_cell:
                    neigh.add(v)
    neigh_list = sorted(neigh)

    candidates = enumerate_masks_powerset(neigh_list)

    # Exponential mechanism
    denom = max(1, len(neigh_list))
    utilities = []
    for M in candidates:
        L = model.leakage(M)
        utilities.append(compute_utility(M, L, lam, denom))

    utilities = np.array(utilities, dtype=float)

    # NOTE: leaving your exponential mechanism math unchanged
    scores = (epsilon * utilities) / (2.0 * lam)
    probs = np.exp(scores - np.max(scores))
    probs /= probs.sum()

    idx = np.random.choice(len(candidates), p=probs)
    final_mask = candidates[idx]
    util_val = utilities[idx]

    leakage_base = model.leakage(set())
    leakage_final = model.leakage(final_mask)

    paths_proxy = estimate_paths_proxy_from_channels(
        num_channel_edges=len(model.channels),
        L_empty=leakage_base,
        L_mask=leakage_final
    )

    model_time = time.time() - model_start
    cand_members = sum(len(s) for s in candidates)

    memory_overhead = estimate_memory_overhead_bytes_delexp(
        hyperedges=H,
        num_vertices=len(model.all_cells),
        mask_size=len(final_mask),
        num_candidates=len(candidates),
        candidate_mask_members=cand_members
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
        "num_channel_edges": int(len(model.channels)),
        "baseline_leakage_empty_mask": float(leakage_base),
        # Helpful debug so you can confirm dampening is active
        "weight_dampen_alpha": float(WEIGHT_DAMPEN_ALPHA),
        "weight_dampen_center": float(WEIGHT_DAMPEN_CENTER),
        "weight_clip_min": float(WEIGHT_CLIP_MIN),
        "weight_clip_max": float(WEIGHT_CLIP_MAX),
    }


if __name__ == "__main__":
    print(exponential_deletion_main("airport", 500, "latitude_deg"))
    print(exponential_deletion_main("hospital", 500, "ProviderNumber"))
    print(exponential_deletion_main("tax", 500, "marital_status"))
    print(exponential_deletion_main("adult", 500, "education"))
    print(exponential_deletion_main("Onlineretail", 500, "InvoiceNo"))
