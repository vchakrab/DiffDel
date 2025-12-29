#!/usr/bin/env python3
"""
two_phase_delexp.py  (DROP-IN)

This is "delexp broken into two":
  (A) OFFLINE: build + cache attribute templates (candidate masks, leakage, utility, exp-mech probs)
  (B) ONLINE:  load template, sample a mask, call update_mask_to_null(), return standardized metrics

New (λ, ε) spec:
  Utility:        u(M) = -λ * L(M) - (1-λ) * |M| / (|I(c*)| - 1)
                  where I(c*) is the inference zone (neighbors of target in DC hyperedges)
  Sensitivity Δu: λ
  Sampling:       Pr[M] ∝ exp( ε * u(M) / (2λ) )

Notes:
- Uses inferable fixed-point leakage model (NO path enumeration).
- Candidates are all subsets of inference-zone neighbors of the target attribute.
- paths proxy: total_paths := #hyperedges containing target; blocked := round((1 - L_mask/L_empty)*total)

Based on your pasted two-phase template builder and leakage engine. :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import time
from dataclasses import dataclass
from itertools import chain, combinations
from collections import deque
from sys import getsizeof
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, FrozenSet

import numpy as np

Cell = Any


# ============================================================
# 0) Small helpers
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
        return np.ones_like(scores, dtype=float) / max(1, scores.size)
    return ex / z


def deep_sizeof(obj: Any, *, seen: Optional[Set[int]] = None) -> int:
    """
    Rough recursive memory estimate.
    Useful for standardized 'memory_overhead_bytes' across methods.
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


def get_dataset_weights(dataset: str) -> Optional[Any]:
    """
    Optional: if you have per-dataset edge weights modules, load them.
    Expected module: weights.weights_corrected.<dataset>_weights with WEIGHTS inside.
    Returns None if missing, which falls back to a constant edge_weight.
    """
    dataset = dataset.lower()
    module_name = f"weights.weights_corrected.{dataset}_weights"
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "WEIGHTS", None)
    except ModuleNotFoundError:
        return None


def normalize_edge_weights(num_edges: int, edge_weights: Optional[Any], default_w: float) -> List[float]:
    if edge_weights is None:
        return [float(default_w)] * num_edges
    if isinstance(edge_weights, dict):
        return [float(edge_weights.get(i, default_w)) for i in range(num_edges)]
    if isinstance(edge_weights, (list, tuple, np.ndarray)):
        w = list(edge_weights)
        if len(w) != num_edges:
            w = (w + [default_w] * num_edges)[:num_edges]
        return [float(x) for x in w]
    return [float(default_w)] * num_edges


# ============================================================
# 1) Inferable leakage model (fixed point)
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
    """
    p[v] = probability v becomes inferable given:
      - adversary knows everything except masked cells + target (target starts unknown)
      - a hyperedge can infer a vertex u if all other vertices are inferable, with prob w
    We compute the monotone fixed point using a queue.
    """

    def __init__(self, hyperedges: Sequence[Iterable[Cell]], weights: Sequence[float], target: Cell):
        if len(hyperedges) != len(weights):
            raise ValueError("hyperedges and weights must have the same length")

        V: Set[Cell] = {target}
        hedges: List[Set[Cell]] = []
        for e in hyperedges:
            s = set(e)
            if len(s) < 2:
                continue
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

        L = self._compute_L(p)
        return float(max(0.0, min(1.0, L)))


# ============================================================
# 2) DC parsing -> hyperedges
# ============================================================

def clean_raw_dcs(raw_dcs: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
    """
    Best-effort extraction of attribute tokens from parsed DCs.
    Pulls suffix after '.' from things like "t1.education".
    """
    cleaned_hyperedges: List[Tuple[str, ...]] = []
    for dc in raw_dcs:
        attributes: Set[str] = set()
        for pred in dc:
            if not isinstance(pred, (list, tuple)) or len(pred) < 3:
                continue
            for item in (pred[0], pred[2]):
                if isinstance(item, str) and "." in item:
                    attributes.add(item.split(".")[-1])
        if attributes:
            cleaned_hyperedges.append(tuple(sorted(attributes)))
    return cleaned_hyperedges


def load_parsed_dcs(dataset: str) -> List[List[Tuple[str, str, str]]]:
    """
    Imports DCandDelset.dc_configs.top{Dataset}DCs_parsed and returns denial_constraints (or []).
    """
    try:
        dataset_module_name = "NCVoter" if dataset.lower() == "ncvoter" else dataset.capitalize()
        dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        return getattr(dc_module, "denial_constraints", [])
    except Exception:
        return []


# ============================================================
# 3) New (λ, ε) utility + exp-mech scoring
# ============================================================

def compute_utility_new(
    *,
    leakage: float,
    mask_size: int,
    lam: float,
    zone_size: int,
) -> float:
    """
    u(M) = -λ L(M) - (1-λ) * |M|/(|I(c*)|-1)
    where I(c*) is inference zone (neighbors of target).
    """
    denom = max(1, int(zone_size) - 1)
    norm = float(mask_size) / float(denom)
    return float(-(lam * float(leakage)) - ((1.0 - lam) * norm))


def exp_mech_probs_from_utilities(
    utilities: Sequence[float],
    *,
    epsilon: float,
    lam: float,
) -> np.ndarray:
    """
    Pr[M] ∝ exp( ε * u(M) / (2λ) )
    """
    lam_eff = max(1e-12, float(lam))
    scores = (float(epsilon) * np.array(list(utilities), dtype=float)) / (2.0 * lam_eff)
    return stable_softmax(scores)


def estimate_paths_proxy(
    *,
    num_channel_edges: int,
    L_empty: float,
    L_mask: float
) -> Tuple[int, int]:
    """
    total_paths_est := #hyperedges containing target (channels)
    blocked_est := round((1 - L_mask/L_empty) * total) (clamped)
    """
    total = int(max(0, num_channel_edges))
    if total == 0 or not np.isfinite(L_empty) or L_empty <= 1e-15 or not np.isfinite(L_mask):
        return total, 0
    frac = 1.0 - (float(L_mask) / float(L_empty))
    frac = float(max(0.0, min(1.0, frac)))
    blocked = int(round(frac * total))
    blocked = int(max(0, min(total, blocked)))
    return total, blocked


# ============================================================
# 4) OFFLINE: build template
# ============================================================

def build_template_two_phase(
    dataset: str,
    attribute: str,
    *,
    save_dir: str = "templates",
    lam: float = 0.5,
    epsilon: float = 1.0,
    default_edge_weight: float = 0.8,
) -> Dict[str, Any]:
    """
    Offline template:
      - build hyperedges from DCs
      - define inference zone as neighbors of target (direct neighbors in incident hyperedges)
      - enumerate candidate masks over that zone
      - compute leakage + utility + exp-mech probabilities
      - save as pickle
    """
    raw_dcs = load_parsed_dcs(dataset)
    hyperedges = [h for h in clean_raw_dcs(raw_dcs) if len(h) >= 2]

    # optional per-edge weights from your weights module
    edge_weights_obj = get_dataset_weights(dataset)
    weights = normalize_edge_weights(len(hyperedges), edge_weights_obj, default_edge_weight)

    model = InferableLeakageModel(hyperedges, weights, attribute)

    # IMPORTANT: inference zone I(c*) = direct neighbors of target in incident hyperedges
    zone_set: Set[str] = set()
    for e in hyperedges:
        if attribute in e:
            for v in e:
                if v != attribute:
                    zone_set.add(v)
    zone = sorted(zone_set)

    candidate_masks: List[FrozenSet[str]] = [frozenset(m) for m in powerset(zone)]
    if not candidate_masks:
        candidate_masks = [frozenset()]

    Leakage: Dict[FrozenSet[str], float] = {}
    Utility: Dict[FrozenSet[str], float] = {}
    utilities: List[float] = []

    for m in candidate_masks:
        L = model.leakage(set(m))
        U = compute_utility_new(leakage=L, mask_size=len(m), lam=float(lam), zone_size=len(zone))
        Leakage[m] = float(L)
        Utility[m] = float(U)
        utilities.append(float(U))

    probs = exp_mech_probs_from_utilities(utilities, epsilon=float(epsilon), lam=float(lam))
    Probability: Dict[FrozenSet[str], float] = {m: float(p) for m, p in zip(candidate_masks, probs)}

    # baseline leakage (empty mask) for paths proxy (handy later)
    L_empty = float(Leakage.get(frozenset(), model.leakage(set())))

    T_attr = {
        "dataset": dataset,
        "target": attribute,

        "hyperedges": hyperedges,
        "weights": weights,

        "zone": zone,                          # I(c*)
        "R_intra": candidate_masks,            # list[frozenset]
        "Leakage": Leakage,                    # dict[frozenset]->float
        "Utility": Utility,                    # dict[frozenset]->float
        "Probability": Probability,            # dict[frozenset]->float

        "lam": float(lam),
        "epsilon": float(epsilon),

        "num_channel_edges": int(len(model.channel_edges)),
        "baseline_leakage_empty_mask": float(L_empty),
    }

    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f"{dataset}_{attribute}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(T_attr, f)

    return T_attr


def load_template(dataset: str, attribute: str, *, save_dir: str = "templates") -> Dict[str, Any]:
    pkl_path = os.path.join(save_dir, f"{dataset}_{attribute}.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ============================================================
# 5) ONLINE: sample + apply update_to_null
# ============================================================

def update_mask_to_null(dataset: str, key: int, mask_cols: Set[str]) -> Tuple[float, int]:
    """
    YOU MUST REPLACE THIS with your real DB update.
    Return (update_time_seconds, num_cells_updated).
    """
    # ---- Example stub (no-op) ----
    t0 = time.time()
    _ = (dataset, key, mask_cols)
    time.sleep(0.0)
    return float(time.time() - t0), int(len(mask_cols))


def two_phase_delexp_main(
    dataset: str,
    key: int,
    target_attribute: str,
    *,
    template_dir: str = "templates",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Online:
      - load precomputed template for attribute
      - sample mask using stored Probability
      - apply update_mask_to_null(dataset, key, mask)
      - return standardized metrics
    """
    if rng is None:
        rng = np.random.default_rng()

    # init_time: template load
    init_start = time.time()
    T_attr = load_template(dataset, target_attribute, save_dir=template_dir)
    init_time = float(time.time() - init_start)

    model_start = time.time()

    mask_list: List[FrozenSet[str]] = T_attr["R_intra"]
    probs_dict: Dict[FrozenSet[str], float] = T_attr["Probability"]
    probabilities = np.array([probs_dict[m] for m in mask_list], dtype=float)

    s = float(probabilities.sum())
    if s <= 0.0 or not np.isfinite(s):
        probabilities = np.ones_like(probabilities) / max(1, probabilities.size)
    else:
        probabilities /= s

    selected_idx = int(rng.choice(len(mask_list), p=probabilities))
    chosen_mask = mask_list[selected_idx]
    final_mask_cols = set(chosen_mask)

    model_time = float(time.time() - model_start)

    leakage = float(T_attr["Leakage"][chosen_mask])
    utility = float(T_attr["Utility"][chosen_mask])

    # paths proxy needs num_channel_edges and baseline empty leakage
    num_channel_edges = int(T_attr.get("num_channel_edges", 0))
    L_empty = float(T_attr.get("baseline_leakage_empty_mask", 0.0))
    num_paths, paths_blocked = estimate_paths_proxy(
        num_channel_edges=num_channel_edges,
        L_empty=L_empty,
        L_mask=leakage,
    )

    # Apply deletion = update masked columns to NULL
    upd_time, num_cells_updated = update_mask_to_null(dataset, key, final_mask_cols)
    del_time = float(upd_time)
    update_time = float(upd_time)

    # memory: count template + final mask
    memory_overhead_bytes = deep_sizeof(T_attr) + deep_sizeof(final_mask_cols)

    # instantiated cells proxy: size of inference zone (neighbors)
    num_instantiated_cells = int(len(T_attr.get("zone", [])))

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
    }


# ============================================================
# 6) CLI
# ============================================================

def cmd_build(args: argparse.Namespace) -> None:
    build_template_two_phase(
        dataset=args.dataset,
        attribute=args.attribute,
        save_dir=args.template_dir,
        lam=args.lam,
        epsilon=args.epsilon,
        default_edge_weight=args.edge_weight,
    )
    print(f"[OK] Built template: {args.template_dir}/{args.dataset}_{args.attribute}.pkl")


def cmd_run(args: argparse.Namespace) -> None:
    out = two_phase_delexp_main(
        dataset=args.dataset,
        key=args.key,
        target_attribute=args.attribute,
        template_dir=args.template_dir,
        rng=np.random.default_rng(args.seed) if args.seed is not None else None,
    )
    # pretty print
    print("RESULT:")
    for k in [
        "init_time", "model_time", "del_time", "update_time",
        "leakage", "utility", "mask_size",
        "num_paths", "paths_blocked",
        "memory_overhead_bytes", "num_instantiated_cells",
    ]:
        print(f"  {k}: {out[k]}")
    print(f"  final_mask: {sorted(list(out['final_mask']))}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build/save offline template for one (dataset, attribute)")
    pb.add_argument("--dataset", required=True)
    pb.add_argument("--attribute", required=True)
    pb.add_argument("--template-dir", default="templates")
    pb.add_argument("--lam", type=float, default=0.5)
    pb.add_argument("--epsilon", type=float, default=1.0)
    pb.add_argument("--edge-weight", type=float, default=0.8)
    pb.set_defaults(func=cmd_build)

    pr = sub.add_parser("run", help="Run online phase using a saved template (samples mask + update_to_null)")
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--attribute", required=True)
    pr.add_argument("--key", type=int, required=True)
    pr.add_argument("--template-dir", default="templates")
    pr.add_argument("--seed", type=int, default=None)
    pr.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
