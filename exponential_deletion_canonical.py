"""
exponential_deletion.py (WITH CANONICAL MASKS OPTIMIZATION)

Adds canonical mask enumeration to reduce search space from 2^|I(c*)| to only
canonical masks where M = F(M) (frontier equality).

Key improvements:
- Layer-based enumeration with frontier pruning
- Dramatically reduced candidate set
- Same optimal result with orders of magnitude speedup
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence, Tuple
from itertools import chain, combinations
from collections import deque, defaultdict
import numpy as np


try:
    from rtf_core import initialization_phase
    from rtf_core.initialization_phase import InitializationManager


    HAVE_RTF = True
except ImportError:
    HAVE_RTF = False
    print("[WARN] rtf_core not available, using standalone mode")

try:
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


    HAVE_WEIGHTS = True
except:
    HAVE_WEIGHTS = False


    def get_dataset_weights(dataset: str):
        return None


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


# ============================================================
# CANONICAL MASK ENUMERATOR
# ============================================================

class CanonicalMaskEnumerator:
    """
    Enumerates only canonical masks using layer structure and frontier pruning.
    Reduces search space from 2^|I(c*)| to canonical masks only.
    """

    def __init__(self, hyperedges: List[Tuple[str, ...]], target: str):
        self.hyperedges = hyperedges
        self.target = target

        # Build cell universe
        self.cells: Set[str] = set()
        for e in hyperedges:
            self.cells.update(e)

        if target not in self.cells:
            raise ValueError(f"Target {target} not in hypergraph")

        # Build inference zone
        self.inference_zone = self._compute_inference_zone()

        # Compute layers
        self.layers, self.cell_to_layer = self._compute_layers()

        # Build adjacency
        self.cell_to_edges = defaultdict(list)
        for i, e in enumerate(hyperedges):
            for c in e:
                self.cell_to_edges[c].append(i)

    def _compute_inference_zone(self) -> Set[str]:
        """Compute I(c*): cells that can reach target through inference."""
        reachable = {self.target}
        changed = True

        while changed:
            changed = False
            for e in self.hyperedges:
                if self.target in e or any(c in reachable for c in e):
                    for c in e:
                        if c not in reachable:
                            reachable.add(c)
                            changed = True

        return reachable - {self.target}

    def _compute_layers(self) -> Tuple[List[Set[str]], Dict[str, int]]:
        """Compute layer structure: L_k = cells at distance k from target."""
        distance = {self.target: 0}
        queue = deque([self.target])

        while queue:
            current = queue.popleft()
            curr_dist = distance[current]

            for e in self.hyperedges:
                if current in e:
                    for c in e:
                        if c != current and c not in distance:
                            distance[c] = curr_dist + 1
                            queue.append(c)

        # Group by distance and reverse (L_1 is closest to target)
        max_dist = max(distance.values()) if distance else 0
        layers = [set() for _ in range(max_dist + 1)]

        for cell, dist in distance.items():
            if cell != self.target:
                layers[dist].add(cell)

        layers = [layer for layer in reversed(layers) if layer]

        # Rebuild cell_to_layer
        cell_to_layer = {}
        for i, layer in enumerate(layers, 1):
            for cell in layer:
                cell_to_layer[cell] = i

        return layers, cell_to_layer

    def is_reachable(self, cell: str, observed: Set[str]) -> bool:
        """Check if cell is reachable from observed set via inference."""
        if cell in observed:
            return True

        reachable = set(observed)
        queue = deque(observed)

        while queue:
            current = queue.popleft()

            for edge_idx in self.cell_to_edges[current]:
                edge = self.hyperedges[edge_idx]

                for target_cell in edge:
                    if target_cell not in reachable:
                        if all(c in reachable for c in edge if c != target_cell):
                            reachable.add(target_cell)
                            queue.append(target_cell)

                            if target_cell == cell:
                                return True

        return False

    def compute_frontier(self, mask: Set[str]) -> Set[str]:
        """Compute F(M): masked cells that are directly attackable."""
        observed = self.cells - mask - {self.target}
        frontier = set()

        for c in mask:
            if self.is_reachable(c, observed):
                frontier.add(c)

        return frontier

    def is_canonical(self, mask: Set[str]) -> bool:
        """Check if mask is canonical (M = F(M))."""
        return mask == self.compute_frontier(mask)

    def enumerate_canonical_masks(
            self,
            max_masks: Optional[int] = None,
            verbose: bool = True
    ) -> List[Set[str]]:
        """
        Enumerate canonical masks using layer-by-layer traversal.
        """
        canonical_masks = [set()]  # Empty mask is always canonical

        if not self.layers:
            return canonical_masks

        if verbose:
            print(f"\n[CANONICAL] Layer structure:")
            for i, layer in enumerate(self.layers, 1):
                print(f"  L_{i}: {len(layer)} cells - {sorted(layer)}")

        # Layer-by-layer enumeration
        for layer_idx, layer in enumerate(self.layers):
            new_masks = []
            layer_list = sorted(layer)

            # For each existing mask, try adding subsets of current layer
            for base_mask in canonical_masks:
                for mask_bits in range(1 << len(layer_list)):
                    layer_subset = {layer_list[i] for i in range(len(layer_list))
                                    if mask_bits & (1 << i)}

                    candidate = base_mask | layer_subset

                    # Check canonicity
                    if self.is_canonical(candidate):
                        if candidate not in new_masks:
                            new_masks.append(candidate)

                            if max_masks and len(new_masks) >= max_masks:
                                if verbose:
                                    print(f"[CANONICAL] Reached max_masks limit: {max_masks}")
                                return new_masks

            canonical_masks = new_masks
            if verbose:
                print(f"  After layer {layer_idx + 1}: {len(canonical_masks)} canonical masks")

        return canonical_masks


# ============================================================
# FIXED Inferable leakage model
# ============================================================

class InferableLeakageModel:
    """Numerically stable inferable leakage model with log-space computation."""

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

        self.channel_edges: List[int] = [ei for ei, (verts, _w) in enumerate(self.edges) if
                                         self.tid in verts]

    def _attempt(self, verts: Tuple[int, ...], w: float, infer_v: int, p: List[float]) -> float:
        """Stable computation using log-space."""
        if w <= 0:
            return 0.0
        if w >= 1.0:
            w = 1.0

        log_result = math.log(w)

        for u in verts:
            if u == infer_v:
                continue
            if p[u] <= 1e-12:
                return 0.0
            if p[u] >= 1.0 - 1e-12:
                continue
            log_result += math.log(p[u])

        if log_result > 0:
            return 0.9999
        if log_result < -20:
            return 1e-8

        result = math.exp(log_result)
        return float(max(1e-8, min(0.9999, result)))

    def _recompute_pv(self, v: int, observed: List[bool], p: List[float]) -> float:
        """Stable recomputation with regularization."""
        if observed[v]:
            return 0.9999

        log_prod_fail = 0.0
        has_inference = False

        for ei in self.inc[v]:
            verts, w = self.edges[ei]
            a = self._attempt(verts, w, v, p)

            if a <= 1e-12:
                continue

            has_inference = True

            if a >= 0.9999:
                return 0.9999

            if a < 0.1:
                log_prod_fail += math.log1p(-a)
            else:
                log_prod_fail += math.log(1.0 - a)

            if log_prod_fail < -20:
                return 0.9999

        if not has_inference:
            return 1e-6

        if log_prod_fail < -20:
            return 0.9999

        prod_fail = math.exp(log_prod_fail)
        result = 1.0 - prod_fail

        return float(max(1e-6, min(0.9999, result)))

    def _compute_L(self, p: List[float]) -> float:
        """Compute leakage in log-space."""
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

        if log_prod_fail < -30:
            return 1.0

        return float(max(1e-10, min(1.0, 1.0 - math.exp(log_prod_fail))))

    def leakage_with_diagnostics(
            self,
            mask: Set[str],
            *,
            tau: float = 1e-6,
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
# Exponential mechanism
# ============================================================

def compute_utility(
        mask: Set[str],
        leakage: float,
        lam: float,
        num_candidates_minus_one: int
) -> float:
    if num_candidates_minus_one <= 0:
        norm = 0.0
    else:
        norm = len(mask) / num_candidates_minus_one
    return float(-(lam * leakage) - ((1.0 - lam) * norm))


def exponential_mechanism_sample(
        candidates: List[Set[str]],
        *,
        model: InferableLeakageModel,
        epsilon: float,
        lam: float
) -> Tuple[Set[str], float]:
    utilities = np.empty(len(candidates), dtype = float)
    denom = max(1, len(candidates) - 1)

    sum_L = 0
    for i, M in enumerate(candidates):
        L = model.leakage(M)
        sum_L += L
        utilities[i] = compute_utility(M, L, lam, denom)
    print(
        f"[EXPOMECH] Average Leakage across {len(candidates)} candidates: {sum_L / len(utilities):.6f}")

    scores = (float(epsilon) * utilities) / (2.0 * max(1e-10, float(lam)))
    max_score = float(np.max(scores)) if len(scores) else 0.0
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / np.sum(exp_scores)

    idx = int(np.random.choice(len(candidates), p = probs))
    return candidates[idx], float(utilities[idx])


# ============================================================
# Paths proxy
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
# Memory estimate
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
# Data loading helpers
# ============================================================

def load_parsed_dcs(dataset: str) -> List[List[Tuple[str, str, str]]]:
    if not HAVE_RTF:
        return []
    try:
        dataset_module_name = "NCVoter" if dataset.lower() == "ncvoter" else dataset.capitalize()
        dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist = ["denial_constraints"])
        return getattr(dc_module, "denial_constraints", [])
    except Exception:
        return []


def map_dc_to_weight(init_manager, dc, weights):
    if weights is None:
        return 0.5
    return weights[init_manager.denial_constraints.index(dc)]


def dc_to_hyperedges(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    hyperedges: List[Tuple[str, ...]] = []
    hyperedge_weights: List[float] = []

    weights = get_dataset_weights(init_manager.dataset) if HAVE_WEIGHTS else None

    for dc in getattr(init_manager, "denial_constraints", []):
        attrs: Set[str] = set()
        weight = map_dc_to_weight(init_manager, dc, weights)
        for pred in dc:
            if isinstance(pred, (list, tuple)) and len(pred) >= 1:
                token = pred[0]
                if isinstance(token, str) and "." in token:
                    attrs.add(token.split(".")[-1])
        if len(attrs) >= 2:
            hyperedges.append(tuple(sorted(attrs)))
            hyperedge_weights.append(weight)
    return hyperedges, hyperedge_weights


# ============================================================
# Main orchestrator WITH CANONICAL MASKS
# ============================================================

def exponential_deletion_main(
        dataset: str,
        key: int,
        target_cell: str,
        *,
        epsilon: float = 10,
        lam: float = 0.67,
        use_canonical: bool = True,  # NEW FLAG
        max_canonical_masks: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main entry point with canonical mask optimization.

    Args:
        use_canonical: If True, use canonical mask enumeration (much faster)
                       If False, use full powerset (original behavior)
    """
    print(f"\n{'=' * 60}")
    print(f"Exponential Deletion: {dataset}, key={key}, target={target_cell}")
    print(f"Mode: {'CANONICAL MASKS' if use_canonical else 'FULL POWERSET'}")
    print(f"{'=' * 60}")

    # ----------------------
    # INIT
    # ----------------------
    init_start = time.time()

    if HAVE_RTF:
        raw_dcs = load_parsed_dcs(dataset)
        init_manager = initialization_phase.InitializationManager(
            {"key": key, "attribute": target_cell},
            dataset,
            0
        )
        hyperedges, weight = dc_to_hyperedges(init_manager)
    else:
        # Standalone mode for testing
        hyperedges = [
            (target_cell, "attr1", "attr2"),
            (target_cell, "attr2", "attr3"),
            (target_cell, "attr1", "attr3")
        ]
        weight = [0.3, 0.5, 0.7]

    H = hyperedges
    W = weight

    instantiated_cells: Set[str] = set()
    for e in H:
        instantiated_cells.update(e)
    num_instantiated_cells = int(len(instantiated_cells))

    init_time = float(time.time() - init_start)

    # ----------------------
    # MODEL
    # ----------------------
    model_start = time.time()

    model = InferableLeakageModel(H, W, target = target_cell)

    # Get neighbors
    neigh: Set[str] = set()
    for e in H:
        if target_cell in e:
            for v in e:
                if v != target_cell:
                    neigh.add(v)

    # CANONICAL MASK ENUMERATION OR POWERSET
    if use_canonical and len(neigh) > 0:
        print(f"\n[CANONICAL] Enumerating canonical masks...")
        enum_start = time.time()

        enumerator = CanonicalMaskEnumerator(H, target_cell)
        candidates = enumerator.enumerate_canonical_masks(
            max_masks = max_canonical_masks,
            verbose = True
        )

        enum_time = time.time() - enum_start

        full_space_size = 2 ** len(neigh)
        reduction_factor = full_space_size / max(1, len(candidates))

        print(f"\n[CANONICAL] Enumeration complete in {enum_time:.3f}s")
        print(f"[CANONICAL] Inference zone: {len(enumerator.inference_zone)} cells")
        print(f"[CANONICAL] Full search space: {full_space_size}")
        print(f"[CANONICAL] Canonical masks: {len(candidates)}")
        print(f"[CANONICAL] Reduction factor: {reduction_factor:.2e}x")
    else:
        print(f"\n[POWERSET] Using full powerset enumeration...")
        candidates = [set(s) for s in powerset(sorted(neigh))]
        if not candidates:
            candidates = [set()]
        print(f"[POWERSET] Total candidate masks: {len(candidates)}")

    print(f"\nDirect neighbors for '{target_cell}': {sorted(list(neigh))}")

    # Exponential mechanism
    final_mask, util_val = exponential_mechanism_sample(
        candidates,
        model = model,
        epsilon = epsilon,
        lam = lam
    )

    leakage = float(model.leakage(final_mask))
    L_empty = float(model.leakage(set()))

    print(f"\nResults:")
    print(f"  Selected mask: {sorted(final_mask) if final_mask else '∅'}")
    print(f"  Baseline leakage (empty): {L_empty:.6f}")
    print(f"  Final leakage: {leakage:.6f}")
    print(f"  Leakage reduction: {(1 - leakage / L_empty) * 100:.2f}%")

    num_channel_edges = int(len(model.channel_edges))
    paths_proxy = estimate_paths_proxy_from_channels(
        num_channel_edges = num_channel_edges,
        L_empty = L_empty,
        L_mask = leakage
    )

    model_time = float(time.time() - model_start)

    # ----------------------
    # MEMORY
    # ----------------------
    num_vertices = int(model.n)
    cand_members = int(sum(len(s) for s in candidates))
    memory_overhead = estimate_memory_overhead_bytes_delexp(
        hyperedges = H,
        num_vertices = num_vertices,
        mask_size = len(final_mask),
        num_candidate_masks = len(candidates),
        candidate_mask_members = cand_members,
        includes_channel_map = True
    )

    print(f"\nTiming:")
    print(f"  Init: {init_time:.3f}s")
    print(f"  Model: {model_time:.3f}s")
    print(f"  Total: {init_time + model_time:.3f}s")

    return {
        "init_time": init_time,
        "model_time": model_time,
        "del_time": 0.0,

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

        # NEW: canonical mask stats
        "num_candidates": len(candidates),
        "used_canonical": use_canonical,
    }


if __name__ == "__main__":
    # Test with canonical masks (default)
    print("\n" + "=" * 60)
    print("TESTING WITH CANONICAL MASKS")
    print("=" * 60)

    if HAVE_RTF:
        result1 = exponential_deletion_main("airport", 500, "latitude_deg", use_canonical = True)
        result2 = exponential_deletion_main("hospital", 500, "ProviderNumber", use_canonical = True)
        result3 = exponential_deletion_main("tax", 500, "marital_status", use_canonical = True)
    else:
        # Standalone test
        result1 = exponential_deletion_main("test", 500, "target", use_canonical = True)

    # Optionally test without canonical (for comparison)
    # print("\n" + "="*60)
    # print("TESTING WITH FULL POWERSET (for comparison)")
    # print("="*60)
    # result_full = exponential_deletion_main("airport", 500, "latitude_deg", use_canonical=False)