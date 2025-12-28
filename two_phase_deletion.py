import math
import os
import pickle
import time
from dataclasses import dataclass
from itertools import chain, combinations
from collections import deque
from sys import getsizeof
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple, FrozenSet, Optional

import numpy as np

Cell = Any


# ============================================================
# 0) Small helpers (exponential deletion essentials)
# ============================================================

def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    """All subsets of iterable, as tuples."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    if scores.size == 0:
        return np.array([], dtype=float)
    m = float(np.max(scores))
    ex = np.exp(scores - m)
    z = float(np.sum(ex))
    if z <= 0.0 or not np.isfinite(z):
        # fallback uniform
        return np.ones_like(scores, dtype=float) / max(1, scores.size)
    return ex / z


def exponential_mechanism_sample(
    candidates: Sequence[FrozenSet[Cell]],
    utilities: Sequence[float],
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[FrozenSet[Cell], np.ndarray]:
    """
    Exponential mechanism:
      Pr[M] ∝ exp( (epsilon * u(M)) / (2 * sensitivity) )
    Returns (chosen_mask, probabilities)
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(candidates) != len(utilities):
        raise ValueError("candidates and utilities must have same length")

    if len(candidates) == 0:
        return frozenset(), np.array([], dtype=float)

    # scores = (eps * utility) / (2 * Δu)
    scores = (epsilon * np.array(utilities, dtype=float)) / (2.0 * float(sensitivity))
    probs = stable_softmax(scores)

    idx = int(rng.choice(len(candidates), p=probs))
    return candidates[idx], probs


def deep_sizeof(obj: Any, *, seen: Optional[Set[int]] = None) -> int:
    """
    Rough recursive memory estimate.
    Good enough for standardized 'memory_overhead_bytes' across methods.
    """
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
# 1) Your leakage engine (kept)
# ============================================================

def inference_zone_union(target: Cell, hyperedges: Sequence[Iterable[Cell]]) -> List[Cell]:
    z: Set[Cell] = set()
    for e in hyperedges:
        z |= set(e)
    z.discard(target)
    return sorted(list(z), key=lambda x: repr(x))


@dataclass(frozen=True)
class Edge:
    verts: Tuple[int, ...]
    w: float


class InferableLeakageModel:
    def __init__(self, hyperedges: Sequence[Iterable[Cell]], weights: Sequence[float], target: Cell):
        if len(hyperedges) != len(weights):
            raise ValueError("hyperedges and weights must have the same length")

        V: Set[Cell] = {target}
        hedges: List[Set[Cell]] = []
        for e in hyperedges:
            s = set(e)
            if len(s) < 2:
                raise ValueError(f"Each hyperedge must contain >=2 cells, got {s}")
            hedges.append(s)
            V |= s

        ordered = sorted(list(V), key=lambda x: repr(x))
        self.cell_to_id: Dict[Cell, int] = {c: i for i, c in enumerate(ordered)}
        self.id_to_cell: List[Cell] = ordered
        self.n = len(ordered)
        self.target = target
        self.tid = self.cell_to_id[target]

        self.edges: List[Edge] = []
        for i, (s, w) in enumerate(zip(hedges, weights)):
            w = float(w)
            if not (0.0 < w <= 1.0):
                raise ValueError(f"Edge {i} weight must be in (0,1], got {w}")
            verts = tuple(sorted(self.cell_to_id[c] for c in s))
            self.edges.append(Edge(verts=verts, w=w))

        self.inc: List[List[int]] = [[] for _ in range(self.n)]
        for ei, e in enumerate(self.edges):
            for v in e.verts:
                self.inc[v].append(ei)

        neigh_sets: List[Set[int]] = [set() for _ in range(self.n)]
        for e in self.edges:
            vs = e.verts
            for v in vs:
                neigh_sets[v].update(vs)
        for v in range(self.n):
            neigh_sets[v].discard(v)
        self.neigh: List[Tuple[int, ...]] = [tuple(sorted(s)) for s in neigh_sets]

        self.channel_edges: List[int] = [ei for ei, e in enumerate(self.edges) if self.tid in e.verts]
        self.zone: List[Cell] = inference_zone_union(target, hyperedges)

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

    def leakage(self, mask: Set[Cell], *, tau: float = 1e-10, max_updates: int = 2_000_000) -> float:
        mask_ids = {self.cell_to_id[c] for c in mask if c in self.cell_to_id}
        if self.tid in mask_ids:
            raise ValueError("mask includes target")
        observed = [True] * self.n
        observed[self.tid] = False
        for v in mask_ids:
            observed[v] = False

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
        return self._compute_L(p)


# ============================================================
# 2) DC parsing -> hyperedges (same as you had)
# ============================================================

def clean_raw_dcs(raw_dcs: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
    cleaned_hyperedges: List[Tuple[str, ...]] = []
    for dc in raw_dcs:
        attributes: Set[str] = set()
        for pred in dc:
            for item in (pred[0], pred[2]):
                if isinstance(item, str) and (("t1." in item) or ("t2." in item)):
                    attributes.add(item.split(".")[-1])
                else:
                    attributes.add(str(item))
        if attributes:
            cleaned_hyperedges.append(tuple(sorted(attributes)))
    return cleaned_hyperedges


# ============================================================
# 3) OFFLINE phase (template build) using exp-deletion essentials
# ============================================================

def build_template_two_phase(
    dataset: str,
    attribute: str,
    *,
    save_dir: str = "templates",
    alpha: float = 1.0,
    beta: float = 0.5,
    epsilon: float = 1.0,
    edge_weight: float = 0.8,
) -> Dict[str, Any]:
    """
    Offline (schema/attribute-level) template:
      - build hyperedges from DCs
      - build inference zone
      - enumerate masks over zone
      - compute leakage + utility for each mask
      - compute exp-mech probabilities
      - save T_attr
    """
    # Load parsed DCs
    dataset_module_name = "NCVoter" if dataset == "ncvoter" else dataset.capitalize()
    dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
    dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
    raw_dcs = dc_module.denial_constraints

    hyperedges = [h for h in clean_raw_dcs(raw_dcs) if len(h) >= 2]
    weights = [float(edge_weight)] * len(hyperedges)

    model = InferableLeakageModel(hyperedges, weights, attribute)
    zone = model.zone  # inference zone (excludes target)

    # Candidates: all subsets of zone
    candidate_masks: List[FrozenSet[Cell]] = [frozenset(m) for m in powerset(zone)]

    Leakage: Dict[FrozenSet[Cell], float] = {}
    Utility: Dict[FrozenSet[Cell], float] = {}
    utilities: List[float] = []

    for m in candidate_masks:
        L = model.leakage(set(m))
        U = -float(alpha) * float(L) - float(beta) * float(len(m))
        Leakage[m] = float(L)
        Utility[m] = float(U)
        utilities.append(float(U))

    # Probabilities via exp-mech
    _, probs = exponential_mechanism_sample(
        candidates=candidate_masks,
        utilities=utilities,
        epsilon=float(epsilon),
        sensitivity=float(alpha),  # matches your prior "2*alpha" normalization idea
    )
    Probability: Dict[FrozenSet[Cell], float] = {m: float(p) for m, p in zip(candidate_masks, probs)}

    T_attr = {
        "hyperedges": hyperedges,
        "weights": weights,
        "target": attribute,
        "zone": zone,
        "R_intra": candidate_masks,     # list[frozenset]
        "Leakage": Leakage,             # dict[frozenset]->float
        "Utility": Utility,             # dict[frozenset]->float
        "Probability": Probability,     # dict[frozenset]->float
        "alpha": float(alpha),
        "beta": float(beta),
        "epsilon": float(epsilon),
    }

    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f"{dataset}_{attribute}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(T_attr, f)

    return T_attr


# ============================================================
# 4) ONLINE phase (sample mask + apply update_to_null)
# ============================================================

def two_phase_deletion_main(
    dataset: str,
    key: int,
    target_attribute: str,
    templates: Dict[str, Dict[str, Any]],
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Online:
      - sample a mask using precomputed Probability
      - apply update_mask_to_null(dataset, key, mask)
      - return standardized metrics
    """
    if rng is None:
        rng = np.random.default_rng()

    if target_attribute not in templates:
        raise ValueError(f"No precomputed template for attribute '{target_attribute}'")
    T_attr = templates[target_attribute]

    # init_time is 0 for offline/online split (unless you do DB copy etc.)
    init_time = 0.0

    model_start = time.time()

    mask_list: List[FrozenSet[Cell]] = T_attr["R_intra"]
    probs_dict: Dict[FrozenSet[Cell], float] = T_attr["Probability"]
    probabilities = np.array([probs_dict[m] for m in mask_list], dtype=float)

    # Safety: normalize if drift
    s = float(probabilities.sum())
    if s <= 0.0 or not np.isfinite(s):
        probabilities = np.ones_like(probabilities) / max(1, probabilities.size)
    else:
        probabilities /= s

    selected_idx = int(rng.choice(len(mask_list), p=probabilities))
    chosen_mask = mask_list[selected_idx]           # frozenset
    final_mask_cols = set(chosen_mask)              # set[str] for update_to_null

    model_time = float(time.time() - model_start)

    leakage = float(T_attr["Leakage"][chosen_mask])
    utility = float(T_attr["Utility"][chosen_mask])

    # “num_paths” proxy: count channel hyperedges touching target
    model = InferableLeakageModel(T_attr["hyperedges"], T_attr["weights"], T_attr["target"])
    num_paths = int(len(model.channel_edges))

    unmasked_L = float(T_attr["Leakage"].get(frozenset(), 0.0))
    paths_blocked = 0
    if unmasked_L > 0.0:
        reduction = max(0.0, min(1.0, (unmasked_L - leakage) / unmasked_L))
        paths_blocked = int(round(num_paths * reduction))

    # Apply deletion = update masked columns to NULL
    # Uses YOUR provided function.
    update_time, num_cells_updated = update_mask_to_null(dataset, key, final_mask_cols)
    del_time = float(update_time)  # if you want separate "del_time", keep it same here

    # Memory: measure template object (roughly) + mask we sampled
    memory_overhead_bytes = deep_sizeof(T_attr) + deep_sizeof(final_mask_cols)

    num_instantiated_cells = int(len(model.zone))  # zone size (attribute-level instantiated cells)

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": float(del_time),
        "update_time": float(update_time),
        "leakage": float(leakage),
        "utility": float(utility),
        "mask_size": int(len(final_mask_cols)),
        "final_mask": final_mask_cols,
        "num_paths": int(num_paths),
        "paths_blocked": int(paths_blocked),
        "memory_overhead_bytes": int(memory_overhead_bytes),
        "num_instantiated_cells": int(num_instantiated_cells),
        "num_cells_updated": int(num_cells_updated),
    }
