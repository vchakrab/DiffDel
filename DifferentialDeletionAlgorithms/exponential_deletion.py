#!/usr/bin/env python3
"""
exponential_deletion.py - Hypergraph-Based Implementation (Schema-level)

Key addition:
- Canonical frontier logic: replace any mask M by its frontier F(M)
  (drop masked cells that are not directly attackable from reachable region).

Also:
- Change "paths" counters to "inference chains" counters using iter_chains.
"""

from __future__ import annotations

import csv
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, FrozenSet
from itertools import chain, combinations

import numpy as np

from rtf_core import initialization_phase

from leakage import (
    get_dataset_weights,
    Hypergraph,
    construct_hypergraph_max,
    construct_hypergraph_actual,
    leakage as compute_leakage,
    compute_utility,
    map_dc_to_weight,
    iter_chains,
    hypergraph_to_edge_dict,
    compute_utility_log, compute_utility_hinge,
)

# ============================================================
# Utilities
# ============================================================

def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


EdgeDict = Dict[int, Tuple[Tuple[str, ...], float]]  # edges[eid] = (verts, weight)


def _build_incident_index(edges: EdgeDict) -> Dict[str, List[int]]:
    incident: Dict[str, List[int]] = {}
    for eid, (verts, _w) in edges.items():
        for v in verts:
            incident.setdefault(v, []).append(eid)
    return incident


def _reachable_closure_undirected(
    visible: Set[str],
    masked: Set[str],
    edges: EdgeDict,
) -> Set[str]:
    """
    Reachability closure under your (undirected) hyperedge inference rule:
      if all-but-one vertices of an edge are reachable, infer the last one.
    Masked cells are treated as not reachable (hard wall).
    """
    reachable: Set[str] = set(visible)

    changed = True
    while changed:
        changed = False
        for _eid, (verts, _w) in edges.items():
            missing: List[str] = []
            for u in verts:
                # masked cells are treated as "missing" forever
                if u in masked or u not in reachable:
                    missing.append(u)
                    if len(missing) > 1:
                        break

            # If exactly one vertex is missing, it would be inferable
            if len(missing) == 1:
                v = missing[0]
                if v not in masked and v not in reachable:
                    reachable.add(v)
                    changed = True

    return reachable


def canonical_frontier_mask_undirected(
    mask: Set[str],
    target_cell: str,
    vertices: Set[str],
    edges: EdgeDict,
    *,
    incident: Optional[Dict[str, List[int]]] = None,
) -> Set[str]:
    """
    Compute frontier F(M):
      masked cells that are directly attackable from the reachable region.

    Formal (matching the writeup intuition):
      - Let O(M) be visible cells = V \ (M ∪ {c*})
      - Let R(M) be reachable closure from O(M) without crossing masked cells
      - A masked cell c is in the frontier if there exists a hyperedge e containing c
        such that all other vertices in e are reachable.

    Returns F(M) ⊆ M.
    """
    M = set(mask)
    M.discard(target_cell)

    visible = set(vertices) - M - {target_cell}
    reachable = _reachable_closure_undirected(visible=visible, masked=M, edges=edges)

    if incident is None:
        incident = _build_incident_index(edges)

    frontier: Set[str] = set()
    for c in M:
        for eid in incident.get(c, []):
            verts, _w = edges[eid]
            # c is frontier if every other vertex in this hyperedge is reachable
            if all((u == c) or (u in reachable) for u in verts):
                frontier.add(c)
                break

    return frontier


def canonicalize_mask(
    mask: Set[str],
    *,
    target_cell: str,
    hypergraph: Hypergraph,
    incident: Optional[Dict[str, List[int]]] = None,
    edges: Optional[EdgeDict] = None,
    vertices: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Replace any mask M by its canonical representative F(M).
    """
    if edges is None:
        edges = hypergraph_to_edge_dict(hypergraph)
    if vertices is None:
        vertices = set(hypergraph.vertices)
    if incident is None:
        incident = _build_incident_index(edges)

    return canonical_frontier_mask_undirected(
        mask=set(mask),
        target_cell=target_cell,
        vertices=vertices,
        edges=edges,
        incident=incident,
    )


def _chain_vertex_set(chain_edge_ids: Iterable[int], edges: EdgeDict) -> Set[str]:
    """
    Convert a chain (list of edge ids) into the union of vertices used in those edges.
    """
    out: Set[str] = set()
    for eid in chain_edge_ids:
        verts, _w = edges[eid]
        out.update(verts)
    return out


def count_inference_chains(
    *,
    edges: EdgeDict,
    target_cell: str,
    mask: Set[str],
) -> Tuple[int, int, int]:
    """
    Counts inference chains (NOT generic "paths"):
      total_chains: number of chains to target in the hypergraph (empty mask)
      active_chains: chains that do NOT touch any masked vertex (excluding target)
      blocked_chains: total - active

    Uses iter_chains from leakage.py for chain enumeration.
    """
    M = set(mask)
    M.discard(target_cell)

    total = 0
    active = 0

    # Enumerate all chains to target (mask-independent)
    for ch in iter_chains(set(), target_cell, edges):
        total += 1
        vset = _chain_vertex_set(ch, edges)
        vset.discard(target_cell)
        if vset.isdisjoint(M):
            active += 1

    blocked = total - active
    return total, active, blocked


# ============================================================
# Candidate masks
# ============================================================

def compute_possible_mask_set_str(
    target_cell: str,
    hypergraph: Hypergraph,
    *,
    mask_space: Optional[str] = None,   # None => full powerset; "canonical" => canonicalized candidates
    tau: Optional[float] = None,
    canonical_max_chains: int = 2000,
    canonical_max_union_chains: int = 200,
) -> List[Set[str]]:
    """
    Candidates:
      - mask_space=None: full powerset of inference zone
      - mask_space="canonical": generate a reduced set, canonicalized via F(M)
        (dedup by frontier equivalence)
    """
    inference_zone = set(hypergraph.vertices) - {target_cell}

    if mask_space is None:
        return [set(s) for s in powerset(sorted(inference_zone))]

    if str(mask_space).lower() != "canonical":
        raise ValueError(f"mask_space must be None or 'canonical' (got {mask_space!r}).")

    edges: EdgeDict = hypergraph_to_edge_dict(hypergraph)
    vertices = set(hypergraph.vertices)
    incident = _build_incident_index(edges)

    def canon_fs(M: Set[str]) -> FrozenSet[str]:
        F = canonical_frontier_mask_undirected(
            mask=M,
            target_cell=target_cell,
            vertices=vertices,
            edges=edges,
            incident=incident,
        )
        return frozenset(F)

    candidates: Set[FrozenSet[str]] = {frozenset()}  # include empty

    # canonicalized singletons
    for v in inference_zone:
        candidates.add(canon_fs({v}))

    # chain-derived supports + canonicalize
    chain_infos: List[Tuple[float, FrozenSet[str]]] = []
    for ch in iter_chains(set(), target_cell, edges):
        supp: Set[str] = set()
        w_prod = 1.0
        for eid in ch:
            verts_e, w = edges[eid]
            supp |= set(verts_e)
            w_prod *= float(w)
        supp.discard(target_cell)
        if not supp:
            continue
        supp_c = canon_fs(supp)
        if supp_c:
            chain_infos.append((float(w_prod), supp_c))

    chain_infos.sort(key=lambda t: t[0], reverse=True)

    top_supports: List[FrozenSet[str]] = []
    seen: Set[FrozenSet[str]] = set()
    for _w, s in chain_infos:
        if s in seen:
            continue
        seen.add(s)
        top_supports.append(s)
        candidates.add(s)
        if len(top_supports) >= int(canonical_max_chains):
            break

    # unions of top supports, canonicalized
    topN = top_supports[: int(canonical_max_union_chains)]
    for i in range(len(topN)):
        for j in range(i + 1, len(topN)):
            candidates.add(canon_fs(set(topN[i] | topN[j])))

    return [set(fs) for fs in candidates]


# ============================================================
# Exponential mechanism
# ============================================================

def exponential_mechanism_sample(
    candidates: List[Set[str]],
    *,
    target_cell: str,
    hypergraph: Hypergraph,
    epsilon: float,
    lam: float,
    rho: float = 0.9,
    tau: Optional[float] = None,
    leakage_method: str = "greedy_disjoint",
    canonicalize_each: bool = True,
) -> Tuple[Set[str], float, float]:
    """
    Sample a mask using exponential mechanism.

    If canonicalize_each=True, each candidate M is replaced by F(M) before scoring,
    and duplicates are merged (keeping the best utility among duplicates implicitly
    via dedup list order + recompute).
    """

    zone_size = len((hypergraph.vertices - {target_cell}))
    if zone_size <= 4:
        raise ValueError()

    # Precompute canonicalization helpers for this hypergraph
    edges: EdgeDict = hypergraph_to_edge_dict(hypergraph)
    vertices = set(hypergraph.vertices)
    incident = _build_incident_index(edges)

    # Canonicalize + dedup candidates (important so EM probs align to unique masks)
    if canonicalize_each:
        uniq: Dict[FrozenSet[str], Set[str]] = {}
        for M in candidates:
            F = canonical_frontier_mask_undirected(
                mask=set(M),
                target_cell=target_cell,
                vertices=vertices,
                edges=edges,
                incident=incident,
            )
            uniq.setdefault(frozenset(F), set(F))
        candidates = list(uniq.values())
        if not candidates:
            candidates = [set()]

    utilities = np.empty(len(candidates), dtype=float)
    leakages = np.empty(len(candidates), dtype=float)
    candidates = [frozenset(m) for m in powerset(vertices)] or [frozenset()]
    for i, M in enumerate(candidates):
        L = compute_leakage(
            M,
            target_cell,
            hypergraph,
            tau=tau,
            leakage_method=leakage_method,
            return_counts=False,
        )
        writer.writerow(["mask_size", "mask", "leakage"])
        with open(f"{target_cell}_original_utility_mvl.csv", "a", newline = "", encoding = "utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                len(M),  # mask_size
                repr(M),  # mask contents
                L  # leakage
            ])

        leakages[i] = float(L)
        utilities[i] = compute_utility_hinge(
            leakage=float(L),
            mask_size=len(M),
            lam=float(lam),
            zone_size=int(zone_size),
        )

    scores = (float(epsilon) * utilities) / (2.0)
    max_score = float(np.max(scores)) if len(scores) else 0.0
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / np.sum(exp_scores)

    idx = int(np.random.choice(len(candidates), p=probs))
    return candidates[idx], float(utilities[idx]), float(leakages[idx])


# ============================================================
# Memory estimate (unchanged)
# ============================================================

def estimate_memory_overhead_bytes_delexp(
    *,
    hypergraph: Hypergraph,
    mask_size: int,
    num_candidate_masks: int,
    candidate_mask_members: int,
    includes_channel_map: bool = True,
) -> int:
    num_vertices = len(hypergraph.vertices)
    num_edges = len(hypergraph.edges)
    edge_members = sum(len(vertices) for vertices, _ in hypergraph.edges)

    BYTES_PER_VERTEX = 112
    BYTES_PER_EDGE = 184
    BYTES_PER_EDGE_MEMBER = 72
    BYTES_PER_MASK_SET = 96
    BYTES_PER_MASK_MEMBER = 72
    BYTES_PER_FLOAT = 8
    BYTES_PER_INT = 28
    BYTES_PER_CAND_MASK = 96

    est = 0
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER
    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    if includes_channel_map:
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    est += num_candidate_masks * BYTES_PER_CAND_MASK
    est += candidate_mask_members * BYTES_PER_MASK_MEMBER
    est += num_candidate_masks * 8  # utilities array

    return int(est)


# ============================================================
# DC -> RDR hyperedges
# ============================================================

def dc_to_hyperedges(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    rdrs: List[Tuple[str, ...]] = []
    rdr_weights: List[float] = []

    dataset_weights = get_dataset_weights(init_manager.dataset)

    for dc in getattr(init_manager, "denial_constraints", []) or []:
        attrs: Set[str] = set()
        weight = map_dc_to_weight(init_manager, dc, dataset_weights)

        for pred in dc:
            if isinstance(pred, (list, tuple)) and len(pred) >= 1:
                tok0 = pred[0]
                if isinstance(tok0, str) and "." in tok0:
                    attrs.add(tok0.split(".")[-1])

            if isinstance(pred, (list, tuple)) and len(pred) >= 3:
                tok2 = pred[2]
                if isinstance(tok2, str) and "." in tok2:
                    attrs.add(tok2.split(".")[-1])

        if len(attrs) >= 2:
            rdrs.append(tuple(sorted(attrs)))
            rdr_weights.append(float(weight))

    return rdrs, rdr_weights


# ============================================================
# Main orchestrator
# ============================================================

def exponential_deletion_main(
    dataset: str,
    key: int,
    target_cell: str,
    *,
    epsilon: float = 10.0,
    lam: float = 0.67,
    rho: float = 0.9,
    tau: Optional[float] = None,
    leakage_method: str = "greedy_disjoint",
    mask_method: str = None,  # None or "canonical"
) -> Dict[str, Any]:

    # ----------------------
    # INIT
    # ----------------------
    init_start = time.time()

    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target_cell},
        dataset,
        0,
    )

    rdrs, rdr_weights = dc_to_hyperedges(init_manager)

    instantiated_cells: Set[str] = set()
    for rdr in rdrs:
        instantiated_cells.update(rdr)
    num_instantiated_cells = len(instantiated_cells)

    init_time = time.time() - init_start

    # ----------------------
    # HYPERGRAPH CONSTRUCTION
    # ----------------------
    model_start = time.time()

    H_max = construct_hypergraph_max(target_cell, rdrs, rdr_weights)
    H_actual = construct_hypergraph_actual(target_cell, rdrs, rdr_weights)

    candidates = compute_possible_mask_set_str(target_cell, H_max, mask_space=mask_method)
    if not candidates:
        candidates = [set()]

    # ----------------------
    # Exponential mechanism (sample)
    # ----------------------
    final_mask, util_val, _L_selected_quick = exponential_mechanism_sample(
        candidates,
        target_cell=target_cell,
        hypergraph=H_actual,
        epsilon=epsilon,
        lam=lam,
        rho=rho,
        tau=tau,
        leakage_method=leakage_method,
        canonicalize_each=True,  # ensure frontier-only masks scored/sampled
    )

    # Canonicalize the sampled mask too (guarantees output is canonical)
    final_mask = canonicalize_mask(final_mask, target_cell=target_cell, hypergraph=H_actual)

    # ----------------------
    # Leakage baseline + final
    # ----------------------
    leakage_base = compute_leakage(
        set(),
        target_cell,
        H_actual,
        tau=tau,
        leakage_method=leakage_method,
        return_counts=False,
    )

    leakage_final = compute_leakage(
        final_mask,
        target_cell,
        H_actual,
        tau=tau,
        leakage_method=leakage_method,
        return_counts=False,
    )

    # ----------------------
    # Inference chain counts (replaces "paths")
    # ----------------------
    edges_actual: EdgeDict = hypergraph_to_edge_dict(H_actual)
    total_chains, active_chains, chains_blocked = count_inference_chains(
        edges=edges_actual,
        target_cell=target_cell,
        mask=final_mask,
    )

    # Count channel edges (edges containing target)
    num_channel_edges = sum(1 for vertices, _ in H_actual.edges if target_cell in vertices)

    model_time = time.time() - model_start

    # ----------------------
    # Memory estimate
    # ----------------------
    cand_members = sum(len(s) for s in candidates)
    memory_overhead = estimate_memory_overhead_bytes_delexp(
        hypergraph=H_actual,
        mask_size=len(final_mask),
        num_candidate_masks=len(candidates),
        candidate_mask_members=cand_members,
        includes_channel_map=True,
    )

    return {
        "init_time": float(init_time),
        "model_time": float(model_time),
        "del_time": 0.0,

        "dataset": str(dataset),
        "target_cell": str(target_cell),

        "leakage": float(leakage_final),
        "baseline_leakage_empty_mask": float(leakage_base),
        "utility": float(util_val),

        "mask_size": int(len(final_mask)),
        "mask": set(final_mask),

        # CHANGED: inference chains (not generic "paths")
        "total_chains": int(total_chains),
        "active_chains": int(active_chains),
        "chains_blocked": int(chains_blocked),

        "memory_overhead_bytes": int(memory_overhead),
        "num_instantiated_cells": int(num_instantiated_cells),
        "num_channel_edges": int(num_channel_edges),

        "num_rdrs": int(len(rdrs)),
        "tau": None if tau is None else float(tau),
        "rho": float(rho),
        "leakage_method": str(leakage_method),
    }


if __name__ == "__main__":
    out = exponential_deletion_main(
        "adult",
        key=500,
        target_cell="education",
        epsilon=25,
        lam=0.75,
        rho=0.9,
        mask_method="None",  # try canonical
    )
    out = exponential_deletion_main(
        "hospital",
        key = 500,
        target_cell = "ProviderNumber",
        epsilon = 25,
        lam = 0.75,
        rho = 0.9,
        mask_method = "None",  # try canonical
    )
    out = exponential_deletion_main(
        "airport",
        key = 500,
        target_cell = "scheduled_service",
        epsilon = 25,
        lam = 0.75,
        rho = 0.9,
        mask_method = "None",  # try canonical
    )
    out = exponential_deletion_main(
        "flight",
        key = 500,
        target_cell = "FlightNum",
        epsilon = 25,
        lam = 0.75,
        rho = 0.9,
        mask_method = "None",  # try canonical
    )
    out = exponential_deletion_main(
        "tax",
        key = 500,
        target_cell = "marital_status",
        epsilon = 25,
        lam = 0.75,
        rho = 0.9,
        mask_method = "None",  # try canonical
    )


