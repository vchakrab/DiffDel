#!/usr/bin/env python3
"""
BASELINE 3 (ILP) + DELEXP HYPERGRAPH + DELEXP LEAKAGE MODEL (Algorithm 2)

What this file contains:
  - Your Baseline 3 ILP (Gurobi) deletion (unchanged structurally)
  - delexp-style hypergraph construction over schema-level RDRs (Algorithm 1)
  - delexp-style leakage computation with inference chains + rho-safe check (Algorithm 2)
  - Utility: u(M) = -λ·L(M) - (1-λ)·|M|/(|I(c*)|-1)

Key fixes vs your pasted combined script:
  - STRICT per-dataset weight loading (no silent defaults)
  - Proper DC->weight mapping (no "return 1.0" bug)
  - Leakage computed with the SAME model as delexp (not the fixed-point inferable model)
  - No hardcoded target removal ("latitude_deg") — removes {target} correctly
"""

from __future__ import annotations

import importlib
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import deque, defaultdict

import numpy as np

# Optional DB deps
try:
    import mysql.connector
except Exception:
    mysql = None

# Optional config
try:
    import config  # type: ignore
except Exception:
    config = None

# =========================
# USER SETTINGS
# =========================
LAM = 0.5     # set to same λ as delexp
RHO = 0.9     # delexp rho threshold (ρ-safe)
AUTO_ADJUST_RHO = True

DELETION_QUERY = """
UPDATE {table_name}
SET `{column_name}` = NULL
WHERE id = {key};
"""


# ============================================================
# STRICT WEIGHT LOADING (same naming convention as delexp)
# ============================================================

def get_dataset_weights_strict(dataset: str) -> Any:
    """
    Loads edge weights using the same convention as delexp:
      weights.weights_corrected.<dataset>_weights with a WEIGHTS object.

    Raises FileNotFoundError if the module or WEIGHTS is missing.
    """
    ds = str(dataset).lower()
    module_name = f"weights.weights_corrected.{ds}_weights"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise FileNotFoundError(
            f"Missing edge-weight module '{module_name}'. "
            f"Expected: weights/weights_corrected/{ds}_weights.py defining WEIGHTS."
        ) from e

    if not hasattr(mod, "WEIGHTS"):
        raise FileNotFoundError(f"Module '{module_name}' exists but does not define WEIGHTS.")

    weights_obj = getattr(mod, "WEIGHTS")
    if weights_obj is None:
        raise FileNotFoundError(f"Module '{module_name}' defines WEIGHTS=None; expected actual weights.")
    return weights_obj


def map_dc_to_weight_strict(init_manager, dc, weights_obj) -> float:
    """
    delexp-style mapping: use init_manager.denial_constraints ordering.
    """
    try:
        idx = init_manager.denial_constraints.index(dc)
    except ValueError:
        return 1.0
    try:
        return float(weights_obj[idx])
    except Exception:
        # If weights_obj isn't indexable, fail loudly (this is safer than silent defaults)
        raise RuntimeError("WEIGHTS object is not indexable by DC index; check weights module format.")


def dc_to_rdrs_and_weights_strict(init_manager) -> Tuple[List[Tuple[str, ...]], List[float]]:
    """
    Convert denial constraints into RDRs (tuples of attribute names) + aligned weights.
    Schema-level: each DC becomes one hyperedge over the attrs it mentions.
    """
    rdrs: List[Tuple[str, ...]] = []
    rdr_weights: List[float] = []

    weights_obj = get_dataset_weights_strict(init_manager.dataset)

    for dc in getattr(init_manager, "denial_constraints", []) or []:
        attrs: Set[str] = set()
        w = map_dc_to_weight_strict(init_manager, dc, weights_obj)

        for pred in dc:
            if not isinstance(pred, (list, tuple)) or len(pred) < 1:
                continue

            # Your DC format tends to have t1.attr in pred[0] (and sometimes pred[2])
            tok0 = pred[0]
            if isinstance(tok0, str) and "." in tok0:
                attrs.add(tok0.split(".")[-1])

            if len(pred) >= 3:
                tok2 = pred[2]
                if isinstance(tok2, str) and "." in tok2:
                    attrs.add(tok2.split(".")[-1])

        if len(attrs) >= 2:
            rdrs.append(tuple(sorted(attrs)))
            rdr_weights.append(float(w))
    print("RDRs:", rdrs)
    print("RDR_weights:", rdr_weights)
    return rdrs, rdr_weights


# ============================================================
# delexp: Hypergraph construction (Algorithm 1)
# ============================================================

class Hypergraph:
    def __init__(self):
        self.vertices: Set[str] = set()
        self.edges: List[Tuple[Set[str], float]] = []  # (edge_vertices, weight)

    def add_vertex(self, v: str):
        self.vertices.add(v)

    def add_edge(self, vertices: Set[str], weight: float):
        if len(vertices) >= 2:
            self.edges.append((set(vertices), float(weight)))
            self.vertices.update(vertices)


def incident_rdrs(cell: str, rdrs: List[Tuple[str, ...]]) -> List[int]:
    out = []
    for i, rdr in enumerate(rdrs):
        if cell in rdr:
            out.append(i)
    return out


def instantiate_rdr(rdr: Tuple[str, ...], weight: float, mode: str = "MAX") -> List[Tuple[Set[str], float]]:
    # schema-level: one hyperedge per RDR
    verts = set(rdr)
    if len(verts) < 2:
        return []
    return [(verts, float(weight))]


def construct_local_hypergraph(
    target_cell: str,
    rdrs: List[Tuple[str, ...]],
    weights: List[float],
    mode: str = "MAX"
) -> Hypergraph:
    H = Hypergraph()
    H.add_vertex(target_cell)

    added_rdrs: Set[int] = set()
    frontier: Set[str] = {target_cell}
    seen: Set[str] = set()

    while frontier:
        next_frontier: Set[str] = set()
        for c in list(frontier):
            if c in seen:
                continue
            seen.add(c)

            for rdr_idx in incident_rdrs(c, rdrs):
                if rdr_idx in added_rdrs:
                    continue
                added_rdrs.add(rdr_idx)

                rdr = rdrs[rdr_idx]
                w = weights[rdr_idx]
                for edge_verts, edge_w in instantiate_rdr(rdr, w, mode):
                    H.add_edge(edge_verts, edge_w)
                    for v in edge_verts:
                        if v not in seen:
                            next_frontier.add(v)

        frontier = next_frontier

    return H
#test this code thoroughly, with different rdr complexities (edge cases: no rdr, 1 rdr, nested rdr, multiple rdr for the same cell)

def construct_hypergraph_max(target_cell: str, rdrs: List[Tuple[str, ...]], weights: List[float]) -> Hypergraph:
    return construct_local_hypergraph(target_cell, rdrs, weights, mode="MAX")


def construct_hypergraph_actual(target_cell: str, rdrs: List[Tuple[str, ...]], weights: List[float]) -> Hypergraph:
    # schema-level: same as MAX
    return construct_local_hypergraph(target_cell, rdrs, weights, mode="ACTUAL")


# ============================================================
# delexp: Leakage computation (Algorithm 2)
# ============================================================

from collections import deque
from typing import Dict, List, Set

#change this for hypereges (only for reporting)
def get_inference_chains_bfs(
    hypergraph: "Hypergraph",
    target_cell: str,
    masked_cells: Set[str],   # kept for API compatibility; NOT USED
    *,
    max_depth: int = 8,
    max_paths_per_dst: int = 500,
) -> Dict[str, List[List[str]]]:
    """
    Mask-independent chain enumeration (NO filtering):
      - Enumerate candidate simple vertex-paths in the hypergraph.
      - Collect paths for EVERY destination vertex.
      - Mask effects belong in compute_chain_weight(..., mask).

    Returns:
      chains[v] = list of vertex-paths ending at v.
    """
    V = set(hypergraph.vertices)
    chains: Dict[str, List[List[str]]] = {v: [] for v in V}

    # Build undirected adjacency induced by hyperedges
    nbrs: Dict[str, Set[str]] = {v: set() for v in V}
    for edge_verts, _w in hypergraph.edges:
        ev = list(edge_verts)
        for i in range(len(ev)):
            for j in range(len(ev)):
                if i != j:
                    nbrs[ev[i]].add(ev[j])

    for root in V:
        q = deque([(root, [root])])

        while q:
            cur, path = q.popleft()
            if len(path) >= max_depth:
                continue

            for nxt in nbrs.get(cur, ()):
                if nxt in path:
                    continue  # keep simple paths only
                new_path = path + [nxt]

                # record path for destination nxt (cap per-destination)
                if len(chains[nxt]) < max_paths_per_dst:
                    chains[nxt].append(new_path)

                q.append((nxt, new_path))
    print(chains)
    return chains

import math
from typing import List, Set, Tuple, Optional, Iterable

import math
from typing import List, Set, Tuple, Optional, Iterable, FrozenSet

def compute_chain_weight(
    chain: List[str],
    *,
    hypergraph,
    masked_cells: Set[str],
    target_cell: str,
) -> float:
    """
    Corrected per your intended semantics:

    Walk starts at target_cell, then traces along the chain outward from target.
    target_cell may NOT be in chain.

    Hyperedge selection:
      - For each consecutive pair (u, v) along the walk, select the hyperedge e that contains both
        and has maximum weight w(e).
      - If consecutive pairs stay within the SAME selected hyperedge, multiply its weight ONCE.

    Blocking + stopping:
      - Treat target_cell as masked (always).
      - A hyperedge is BLOCKED if it contains >=2 masked vertices.
      - Multiply weights of consecutive BLOCKED hyperedges (counting each hyperedge only once per run of being "current").
      - When the first ACTIVE hyperedge is encountered, multiply it once and STOP.
      - If the chain ends without seeing an active hyperedge, return product of multiplied blocked hyperedges.

    Returns:
      - 0.0 if the walk cannot attach to the target or if a required connecting hyperedge is missing.
      - otherwise product clamped into [1e-12, 1.0].
    """

    if not chain:
        return 1.0

    masked = set(masked_cells)
    print("Masked", masked, "target", target_cell)
    masked.add(target_cell)

    # Find best hyperedge (by max weight) that connects u and v
    def best_edge_between(u: str, v: str) -> Optional[Tuple[FrozenSet[str], float]]:
        best_w = -1.0
        best_verts = None
        for verts, w in hypergraph.edges:
            if u in verts and v in verts:
                fw = float(w)
                if fw > best_w:
                    best_w = fw
                    best_verts = frozenset(verts)
        if best_verts is None:
            return None
        return best_verts, best_w

    def is_blocked(edge_verts: Iterable[str]) -> bool:
        cnt = 0
        for x in edge_verts:
            if x in masked:
                cnt += 1
                if cnt >= 2:
                    return True
        return False

    # Decide which end of chain attaches to target_cell
    can_attach_first = best_edge_between(target_cell, chain[0]) is not None
    can_attach_last  = best_edge_between(target_cell, chain[-1]) is not None

    if can_attach_first and not can_attach_last:
        walk = [target_cell] + list(chain)
    elif can_attach_last and not can_attach_first:
        walk = [target_cell] + list(reversed(chain))
    elif can_attach_first and can_attach_last:
        # Ambiguous: both ends connect. Choose the stronger immediate link.
        e_first = best_edge_between(target_cell, chain[0])
        e_last  = best_edge_between(target_cell, chain[-1])
        w_first = e_first[1] if e_first else -1.0
        w_last  = e_last[1] if e_last else -1.0
        walk = [target_cell] + (list(chain) if w_first >= w_last else list(reversed(chain)))
    else:
        return 0.0  # cannot even start from target into this chain
    # At start of compute_chain_weight
    known = set(hypergraph.vertices) - masked


    # After you decide the walk order
    for i in range(len(walk) - 1):
        u, x = walk[i], walk[i + 1]
        best = best_edge_between(u, x)
        if best is None:
            return 0.0

        edge_verts, _ = best
        # PAPER CONDITION: prerequisites must be known
        if not (set(edge_verts) - {x} <= known):
            return 0.0

        known.add(x)

    logw = 0.0
    current_edge_verts: Optional[FrozenSet[str]] = None
    current_edge_w: Optional[float] = None
    current_edge_blocked: Optional[bool] = None

    def commit_current_edge_and_maybe_stop() -> bool:
        """
        Multiply current edge weight once (if present).
        Return True if we should STOP (i.e., current edge is active).
        """
        nonlocal logw
        if current_edge_verts is None or current_edge_w is None or current_edge_blocked is None:
            return False
        if current_edge_w <= 0.0:
            return True  # treat as dead end
        logw += math.log(min(1.0, current_edge_w))
        return (not current_edge_blocked)

    # Traverse pair-by-pair, but only "commit" an edge when it CHANGES
    for i in range(len(walk) - 1):
        u, v = walk[i], walk[i + 1]
        best = best_edge_between(u, v)
        if best is None:
            return 0.0

        edge_verts, edge_w = best
        edge_blocked = is_blocked(edge_verts)

        if current_edge_verts is None:
            # start tracking first edge
            current_edge_verts = edge_verts
            current_edge_w = edge_w
            current_edge_blocked = edge_blocked
            continue

        if edge_verts == current_edge_verts:
            # still in same hyperedge: DO NOT multiply again
            # (also keep the originally-chosen weight for that hyperedge)
            continue

        # Edge changed: commit previous edge once
        should_stop = commit_current_edge_and_maybe_stop()
        if should_stop:
            # previous edge was active: stop after multiplying it
            w = math.exp(logw)
            return float(min(1.0, max(1e-12, w)))

        # Move to new edge
        current_edge_verts = edge_verts
        current_edge_w = edge_w
        current_edge_blocked = edge_blocked

    # End of walk: commit the last tracked edge once
    commit_current_edge_and_maybe_stop()
    w = math.exp(logw)
    return float(min(1.0, max(1e-12, w)))


def compute_leakage_delexp(mask: Set[str], target_cell: str, hypergraph: Hypergraph, rho: float) -> float:
    # O ← V \ (M ∪ {c*})   (kept; not used for leakage aggregation anymore)
    O = hypergraph.vertices - mask - {target_cell}

    # r(c) ← 1[c ∈ O]      (kept; not used)
    r: Dict[str, float] = {c: (1.0 if c in O else 0.0) for c in hypergraph.vertices}
    _ = r

    # ---------- helper: reconstruct the committed hyperedges used by compute_chain_weight ----------
    masked = set(mask)
    masked.add(target_cell)

    # best hyperedge (max weight) connecting u and v; returns (frozenset(verts), weight) or None
    def best_edge_between(u: str, v: str):
        best_w = -1.0
        best_verts = None
        for verts, w in hypergraph.edges:
            if u in verts and v in verts:
                fw = float(w)
                if fw > best_w:
                    best_w = fw
                    best_verts = frozenset(verts)
        if best_verts is None:
            return None
        return best_verts, best_w

    def is_blocked(edge_verts) -> bool:
        cnt = 0
        for x in edge_verts:
            if x in masked:
                cnt += 1
                if cnt >= 2:
                    return True
        return False

    def committed_edge_sequence_for_chain(chain: List[str]) -> Optional[List[Tuple[frozenset, float]]]:
        """
        Mirror compute_chain_weight's edge-selection + commit/stop rules,
        but return the committed edge list (each committed once).
        Returns None if chain cannot attach / missing connecting edge.
        """
        if not chain:
            return []  # weight 1.0, no edges

        can_attach_first = best_edge_between(target_cell, chain[0]) is not None
        can_attach_last  = best_edge_between(target_cell, chain[-1]) is not None

        if can_attach_first and not can_attach_last:
            walk = [target_cell] + list(chain)
        elif can_attach_last and not can_attach_first:
            walk = [target_cell] + list(reversed(chain))
        elif can_attach_first and can_attach_last:
            e_first = best_edge_between(target_cell, chain[0])
            e_last  = best_edge_between(target_cell, chain[-1])
            w_first = e_first[1] if e_first else -1.0
            w_last  = e_last[1] if e_last else -1.0
            walk = [target_cell] + (list(chain) if w_first >= w_last else list(reversed(chain)))
        else:
            return None

        committed: List[Tuple[frozenset, float]] = []
        current_edge_verts = None
        current_edge_w = None
        current_edge_blocked = None

        def commit_current_and_should_stop() -> bool:
            nonlocal committed
            if current_edge_verts is None or current_edge_w is None or current_edge_blocked is None:
                return False
            if current_edge_w <= 0.0:
                return True
            committed.append((current_edge_verts, float(current_edge_w)))
            return (not current_edge_blocked)  # stop if active

        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i + 1]
            best = best_edge_between(u, v)
            if best is None:
                return None
            edge_verts, edge_w = best
            edge_blocked = is_blocked(edge_verts)

            if current_edge_verts is None:
                current_edge_verts = edge_verts
                current_edge_w = edge_w
                current_edge_blocked = edge_blocked
                continue

            if edge_verts == current_edge_verts:
                continue  # same hyperedge run, don't commit again

            # edge changed => commit prior one
            should_stop = commit_current_and_should_stop()
            if should_stop:
                return committed

            # start tracking new edge
            current_edge_verts = edge_verts
            current_edge_w = edge_w
            current_edge_blocked = edge_blocked

        # end => commit last tracked edge once
        commit_current_and_should_stop()
        return committed

    # ---------- enumerate chains (your current BFS enumerates all endpoints; keep call same) ----------
    chains = get_inference_chains_bfs(hypergraph, target_cell, mask | {target_cell})
    target_chains = chains.get(target_cell, [])
    if not target_chains:
        return 0.0

    # ---------- compute per-chain weights + edge-sets for IE ----------
    chain_weights: List[float] = []
    chain_edge_sets: List[Set[frozenset]] = []
    chain_edge_weight_maps: List[Dict[frozenset, float]] = []

    w_star_max = 0.0

    for p in target_chains:
        # Your existing chain weight function (fix: keyword-only args)
        w_p = float(compute_chain_weight(
            p,
            hypergraph=hypergraph,
            masked_cells=mask,
            target_cell=target_cell,
        ))
        if w_p <= 0.0:
            continue

        # reconstruct committed edges to enable w(p∩p')
        committed = committed_edge_sequence_for_chain(p)
        if committed is None:
            # if reconstruction fails but compute_chain_weight said >0, fall back: skip from IE but keep for NOR
            chain_weights.append(w_p)
            w_star_max = max(w_star_max, w_p)
            continue

        emap: Dict[frozenset, float] = {}
        eset: Set[frozenset] = set()
        prod = 1.0
        for ev, ew in committed:
            eset.add(ev)
            # if the same hyperedge appears twice (shouldn't), keep the first (commit-on-change means unique runs)
            if ev not in emap:
                emap[ev] = float(ew)
            prod *= float(ew)

        # Prefer the authoritative compute_chain_weight result for w(p)
        chain_weights.append(w_p)
        chain_edge_sets.append(eset)
        chain_edge_weight_maps.append(emap)

        w_star_max = max(w_star_max, w_p)

    if not chain_weights:
        return 0.0

    # rho-safe max (keep your behavior)
    if w_star_max > float(rho):
        return 1.0

    # ---------- L_NOR = 1 - Π(1 - w_p)  (your existing log-space constants) ----------
    log_prod = 0.0
    for w in chain_weights:
        w = float(w)
        if w >= 1.0:
            return 1.0
        if w > 1e-15:
            log_prod += math.log1p(-w) if w < 0.5 else math.log(1.0 - w)

    prod_fail = 0.0 if log_prod < -700 else math.exp(log_prod)
    L_nor = 1.0 - prod_fail

    # ---------- L_IE (3rd-order IE approximation from paper) ----------
    # Work cap to avoid combinatorial explosion (keeps the function usable)
    # We keep all your existing constants; this cap is internal only.
    if len(chain_edge_sets) >= 2:
        # Sort by weight desc and keep top K for IE (still exact L_NOR above)
        K = min(120, len(chain_edge_sets))
        order = sorted(range(len(chain_edge_sets)), key=lambda i: chain_weights[i], reverse=True)[:K]

        w_list = [chain_weights[i] for i in order]
        e_sets = [chain_edge_sets[i] for i in order]
        e_maps = [chain_edge_weight_maps[i] for i in order]

        # helper: intersection weight product over common committed hyperedges
        def w_intersection(i: int, j: int) -> float:
            common = e_sets[i] & e_sets[j]
            if not common:
                return 1.0
            prod = 1.0
            # weights should match across chains for same hyperedge; use i's map
            mi = e_maps[i]
            mj = e_maps[j]
            for ev in common:
                prod *= float(mi.get(ev, mj.get(ev, 1.0)))
            return prod

        def w_intersection3(i: int, j: int, k: int) -> float:
            common = e_sets[i] & e_sets[j] & e_sets[k]
            if not common:
                return 1.0
            prod = 1.0
            mi, mj, mk = e_maps[i], e_maps[j], e_maps[k]
            for ev in common:
                prod *= float(mi.get(ev, mj.get(ev, mk.get(ev, 1.0))))
            return prod

        # s1
        s1 = sum(w_list)

        # s2
        s2 = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                denom = w_intersection(i, j)
                if denom <= 0.0:
                    continue
                s2 += (w_list[i] * w_list[j]) / denom

        # s3
        s3 = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                for k in range(j + 1, K):
                    wab = w_intersection(i, j)
                    wac = w_intersection(i, k)
                    wbc = w_intersection(j, k)
                    if wab <= 0.0 or wac <= 0.0 or wbc <= 0.0:
                        continue
                    wabc = w_intersection3(i, j, k)
                    denom = wab * wac * wbc
                    s3 += (w_list[i] * w_list[j] * w_list[k] * wabc) / denom

        L_ie = s1 - s2 + s3
        # paper clips IE to [0,1]
        L_ie = max(0.0, min(1.0, L_ie))

        # paper returns min(L_IE, L_NOR) (then clip to [0,1])
        L = min(L_nor, L_ie)
    else:
        L = L_nor

    return float(max(1e-12, min(1.0, L)))


# ============================================================
# Utility (your newer form you’ve been using elsewhere)
# ============================================================

def compute_utility_new(*, leakage: float, mask_size: int, lam: float, zone_size: int) -> float:
    """
    u(M) = -λ·L(M) - (1-λ)·|M|/(|I(c*)|-1)
    """
    denom = max(1, int(zone_size) - 1)
    norm = float(mask_size) / float(denom)
    return float(-(lam * float(leakage)) - ((1.0 - lam) * norm))


# ============================================================
# Memory estimator (kept from your combined script)
# ============================================================

def estimate_memory_bytes_standard(
    *,
    num_vertices: int,
    num_edges: int,
    edge_members: int,
    mask_size: int,
    stores_candidate_masks: bool,
    num_candidate_masks: int = 0,
    candidate_mask_members: int = 0,
    includes_inferable_model: bool = False,
    includes_channel_map: bool = False,
    ilp_num_cells: int = 0,
    ilp_num_vars: int = 0,
    ilp_num_constrs: int = 0,
) -> int:
    BYTES_PER_VERTEX = 112
    BYTES_PER_EDGE = 184
    BYTES_PER_EDGE_MEMBER = 72
    BYTES_PER_MASK_MEMBER = 72
    BYTES_PER_MASK_SET = 96

    BYTES_PER_EDGE_STRUCT = 80
    BYTES_PER_FLOAT = 8
    BYTES_PER_INT = 28
    BYTES_PER_CAND_MASK = 96

    BYTES_PER_ILP_CELL = 128
    BYTES_PER_ILP_VAR = 96
    BYTES_PER_ILP_CONSTR = 128

    est = 0
    est += num_vertices * BYTES_PER_VERTEX
    est += num_edges * BYTES_PER_EDGE
    est += edge_members * BYTES_PER_EDGE_MEMBER
    est += BYTES_PER_MASK_SET + mask_size * BYTES_PER_MASK_MEMBER

    if includes_inferable_model:
        est += num_edges * BYTES_PER_EDGE_STRUCT
        est += num_vertices * BYTES_PER_FLOAT
        est += num_edges * BYTES_PER_FLOAT

    if includes_channel_map:
        est += num_edges * (BYTES_PER_INT + BYTES_PER_FLOAT)

    if stores_candidate_masks:
        est += num_candidate_masks * BYTES_PER_CAND_MASK
        est += candidate_mask_members * BYTES_PER_MASK_MEMBER

    if ilp_num_cells or ilp_num_vars or ilp_num_constrs:
        est += ilp_num_cells * BYTES_PER_ILP_CELL
        est += ilp_num_vars * BYTES_PER_ILP_VAR
        est += ilp_num_constrs * BYTES_PER_ILP_CONSTR

    return int(est)


# ============================================================
# Baseline 3 ILP (as in your combined script)
# ============================================================

try:
    from gurobipy import Model, GRB, quicksum
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

try:
    from rtf_core import initialization_phase
except Exception:
    initialization_phase = None


@dataclass(frozen=True)
class CellILP:
    attribute: str
    key: int


def get_insertion_time(cursor, table, key, attr):
    try:
        query = f"SELECT `{attr}` FROM {table}_insertiontime WHERE insertionKey = {key}"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else 0
    except Exception:
        return 0


def instantiate_edges_with_time_filter(cursor, table, key, attr, target_time, hypergraph: Dict[Tuple[str, ...], float]):
    edges = []
    for edge_attrs, weight in hypergraph.items():

        # --- FIX: match schema attr against tuple tokens like "t1.attr" ---
        edge_attr_names = {ea.split(".")[-1] for ea in edge_attrs if isinstance(ea, str)}
        if attr not in edge_attr_names:
            continue

        valid_cells = []
        for edge_attr in edge_attrs:
            attr_name = edge_attr.split('.')[-1]  # already doing this
            it = get_insertion_time(cursor, table, key, attr_name)
            if it >= target_time:
                valid_cells.append(edge_attr)      # keep "t1.attr" tokens for ILP consistency

        if len(valid_cells) > 1:
            edges.append(set(valid_cells))
    return edges



def ilp_approach_matching_java(
    cursor,
    table,
    key,
    target_attr,
    target_time,
    hypergraph: Dict[Tuple[str, ...], float],
):
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi required")

    start_total_ilp = time.time()

    max_id = 0
    edge_counter = -1
    cell_to_id: Dict[CellILP, int] = {}
    cell_to_var = {}
    instantiated_cells = set()
    cells_to_visit = deque()

    cell_to_depth = {}
    max_depth = 0

    edge_vars = []
    existing_rdr_vars: Dict[frozenset, Any] = {}

    model = Model("P2E2_ILP")
    model.setParam('OutputFlag', 0)
    model.setParam('LogToConsole', 0)

    obj = quicksum([])

    deleted_cell = CellILP(f"t1.{target_attr}", key)
    a0 = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name=f"a{max_id}")
    cell_to_var[deleted_cell] = a0
    cell_to_id[deleted_cell] = max_id
    cell_to_depth[deleted_cell] = 0
    max_id += 1

    cells_to_visit.append(deleted_cell)
    instantiated_cells.add(deleted_cell)

    while cells_to_visit:
        curr = cells_to_visit.popleft()
        curr_id = cell_to_id[curr]
        curr_depth = cell_to_depth[curr]
        aj = cell_to_var[curr]
        obj += aj

        curr_attr = curr.attribute.split(".")[-1]
        edges = instantiate_edges_with_time_filter(cursor, table, key, curr_attr, target_time, hypergraph)

        for edge in edges:
            frozenset_edge = frozenset(edge)
            if frozenset_edge in existing_rdr_vars:
                bi = existing_rdr_vars[frozenset_edge]
            else:
                edge_counter += 1
                bi = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"b{edge_counter}")
                edge_vars.append(bi)
                existing_rdr_vars[frozenset_edge] = bi

            hij = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"h{edge_counter}_{curr_id}")
            model.addConstr(aj == hij, name=f"head_hidden_{edge_counter}")
            model.addConstr(bi == hij, name=f"rdr_addr_{edge_counter}")

            tail_tji_vars = []
            for cell_attr in edge:
                cell = CellILP(cell_attr, key)

                if cell not in cell_to_id:
                    t_id = max_id
                    cell_to_id[cell] = max_id
                    max_id += 1
                    a_cell = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"a{t_id}")
                    cell_to_var[cell] = a_cell
                else:
                    t_id = cell_to_id[cell]
                    a_cell = cell_to_var[cell]

                tji = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"t{edge_counter}_{t_id}")
                model.addConstr(tji == a_cell, name=f"tail_sync_{edge_counter}_{t_id}")

                if cell != curr:
                    tail_tji_vars.append(tji)

                if cell not in instantiated_cells:
                    instantiated_cells.add(cell)
                    cells_to_visit.append(cell)
                    cell_to_depth[cell] = curr_depth + 1
                    max_depth = max(max_depth, curr_depth + 1)

            if tail_tji_vars:
                model.addConstr(quicksum(tail_tji_vars) >= bi, name=f"tail_req_{edge_counter}")

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible.ilp")
        raise RuntimeError("Infeasible")

    to_delete = set()
    for cell, cell_id in cell_to_id.items():
        var_name = f"a{cell_id}"
        if model.getVarByName(var_name).X == 1.0:
            to_delete.add(cell)

    activated_dependencies_count = sum(1 for bi_var in edge_vars if bi_var.X == 1.0)

    try:
        ilp_num_vars = int(model.NumVars)
        ilp_num_constrs = int(model.NumConstrs)
    except Exception:
        ilp_num_vars = 0
        ilp_num_constrs = 0

    model.dispose()
    total_ilp_time = time.time() - start_total_ilp

    return to_delete, total_ilp_time, max_depth, len(cell_to_id), activated_dependencies_count, ilp_num_vars, ilp_num_constrs


# ============================================================
# Baseline 3 wrapper + delexp leakage/utility integration
# ============================================================

def baseline_deletion_3(target: str, key: int, dataset: str, threshold: float):
    """
    Returns:
      activated_dependencies_count,
      final_mask,
      memory_bytes,
      max_depth,
      init_time,
      model_time,
      deletion_time,
      leakage,
      utility,
      num_cells_aux
    """
    if not GUROBI_AVAILABLE:
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    if initialization_phase is None or config is None or mysql is None:
        print("[WARN] Missing deps for baseline_deletion_3 (rtf_core/config/mysql).")
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    init_start = time.time()
    init_mgr = initialization_phase.InitializationManager({"key": key, "attribute": target}, dataset, threshold)
    init_mgr.initialize()

    # Build RDRs + weights STRICTLY (same convention as delexp)
    rdrs, rdr_weights = dc_to_rdrs_and_weights_strict(init_mgr)

    # Optional rho auto-adjust (same spirit as delexp main)
    rho = float(RHO)
    if AUTO_ADJUST_RHO and rdr_weights:
        mx = max(rdr_weights)
        if mx > rho:
            rho = min(0.999, mx + 0.01)

    # Hypergraph dict used by ILP instantiation (keys are t1.attr tokens)
    # ILP expects hypergraph over "t1.attr" strings (you used that before),
    # so we convert rdrs attrs -> "t1.<attr>".
    hyperedge_dict: Dict[Tuple[str, ...], float] = {}
    for rdr, w in zip(rdrs, rdr_weights):
        edge = tuple(sorted({f"t1.{a}" for a in rdr}))
        hyperedge_dict[edge] = float(w)

    init_time = time.time() - init_start

    model_time = 0.0
    deletion_time = 0.0
    conn = None
    cursor = None

    activated_dependencies_count = 0
    final_mask: Set[str] = set()
    memory_bytes = 0
    max_depth = 0
    num_cells = 0
    leakage = 0.0
    utility = 0.0

    try:
        deletion_start = time.time()

        db = config.get_database_config(dataset)
        conn = mysql.connector.connect(
            host=db['host'],
            user=db['user'],
            password=db['password'],
            database=db['database'],
            ssl_disabled=db.get('ssl_disabled', True),
        )
        cursor = conn.cursor()
        table = f"{dataset}_data"
        if dataset == "airport":
            table = "airports"

        target_time = get_insertion_time(cursor, table, key, target)

        to_del_cells, total_ilp_time, max_depth, num_cells, activated_dependencies_count, ilp_num_vars, ilp_num_constrs = (
            ilp_approach_matching_java(cursor, table, key, target, target_time, hyperedge_dict)
        )
        model_time = float(total_ilp_time)

        # Apply update-to-null for ILP-selected cells
        for cell in to_del_cells:
            attr = cell.attribute.split('.')[-1]
            cursor.execute(DELETION_QUERY.format(table_name=table, column_name=attr, key=key))
        conn.commit()

        # deletion_time excludes ILP model solve time
        deletion_time = (time.time() - deletion_start) - model_time

        # ILP "mask" as attribute names
        final_mask = {cell.attribute.split('.')[-1] for cell in to_del_cells}

        # === delexp leakage computation on H_actual ===
        # mask M should NOT include the target itself
        mask_for_leakage = set(final_mask)
        mask_for_leakage.discard(target)

        H_max = construct_hypergraph_max(target, rdrs, rdr_weights)

        H_actual = construct_hypergraph_actual(target, rdrs, rdr_weights)

        zone_size = len(H_max.vertices - {target})
        E_star = [(vs, w) for (vs, w) in H_actual.edges if target in vs]
        print("\n[DEBUG leakage]")
        print("target:", target)
        print("target in vertices:", target in H_actual.vertices)
        print("num_edges:", len(H_actual.edges))
        print("num_E_star:", len(E_star))

        if E_star:
            ws = [float(w) for _, w in E_star]
            print("E* weight min/max/mean:", min(ws), max(ws), sum(ws) / len(ws))
        else:
            print("No channels into target => leakage must be 0 for all masks.")

        leakage = compute_leakage_delexp(mask_for_leakage, target, H_actual, rho=rho)
        utility = compute_utility_new(leakage=leakage, mask_size=len(mask_for_leakage), lam=float(LAM), zone_size=zone_size)

        # standardized memory estimate (ILP scale + hypergraph size + mask)
        num_edges = len(H_actual.edges)
        edge_members = sum(len(vs) for vs, _w in H_actual.edges)
        num_vertices = len(H_actual.vertices)

        memory_bytes = estimate_memory_bytes_standard(
            num_vertices=num_vertices,
            num_edges=num_edges,
            edge_members=edge_members,
            mask_size=len(mask_for_leakage),
            stores_candidate_masks=False,
            includes_inferable_model=False,
            includes_channel_map=False,
            ilp_num_cells=num_cells,
            ilp_num_vars=ilp_num_vars,
            ilp_num_constrs=ilp_num_constrs,
        )

    except Exception as e:
        print(f"Error in Baseline 3: {e}")
        import traceback
        traceback.print_exc()
        return 0, set(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    # Your old code did: int(num_cells)-2 (id + target)
    # Keep same convention:
    num_cells_aux = int(num_cells) - 2

    return (
        int(activated_dependencies_count),
        set(final_mask),
        int(memory_bytes),
        int(max_depth),
        float(init_time),
        float(model_time),
        float(deletion_time),
        float(leakage),
        float(utility),
        int(num_cells_aux),
    )


if __name__ == '__main__':
    # Example:
    #print(baseline_deletion_3("OriginCityMarketID", 500, "flight", 0))
    print(baseline_deletion_3("education", 500, "adult", 0))


# justify dc cuttoff points, tau as a function of dc weights, corresponding delmin mask size count, avg leakage quartile distribution (mean mode median)
# paths (inference chains/blocked active) For all the masks once we fix the experiment
# inference zone changes per tau
# depth and width (ablation) -> from the graph
# arboricity
# dcs
# These are the most basic things that we can measure there is nothing more fundamental that we can measure
