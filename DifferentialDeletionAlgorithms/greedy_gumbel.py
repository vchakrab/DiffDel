# greedy_gumbel.py

from __future__ import annotations

import time
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np

import math
import dataclasses

from DifferentialDeletionAlgorithms.leakage import compute_utility_logarithmic, compute_utility, \
    compute_utility_hinge, compute_utility_max
# ✅ import YOUR heavy lifting module
from leakage import (
    get_dataset_weights,   # (if you add the helper)

    construct_local_hypergraph,
    Hypergraph,

    leakage as chain_leakage,
)

import sys
import struct
from typing import Any, Iterable, Optional, Tuple


from dataclasses import dataclass

@dataclass
class _Chain:
    """
    Internal representation of an inference chain.

    edges: list of edge_ids (strings) in forward order, with the last edge enabling the target.
    cells: set of all cells/vertices appearing in any edge along the chain.
    active: whether this chain can still contribute non-zero weight under the evolving mask.
    currWeight: current chain weight w(p, M) under the mask semantics in ComputeChainWeight.
    masked: (cells ∩ M) \\{target}
    """
    edges: List[str]
    cells: Set[str]
    active: bool = True
    currWeight: float = 0.0
    masked: Set[str] = dataclasses.field(default_factory=set)


def _compute_chain_weight(
    chain: _Chain,
    *,
    mask: Set[str],
    target_cell: str,
    edge_verts: Dict[str, Set[str]],
    edge_w: Dict[str, float],
) -> float:
    """
    Implements Algorithm 'ComputeChainWeight' from the paper-style pseudocode.

    Semantics:
      - Walk edges in order.
      - Find earliest edge that is 'active' (some prerequisite is visible OR already reachable).
      - Weight is product of edge weights from that earliest active edge to the end.
      - If no edge becomes active, weight is 0.

    Note: For edges containing the target, prerequisites are verts \\{target}.
          For all other edges, prerequisites are verts.
    """
    reachable: Set[str] = set()
    first_active_idx: Optional[int] = None

    for i, eid in enumerate(chain.edges):
        verts = edge_verts[eid]
        prereqs = set(verts)
        if target_cell in prereqs:
            prereqs.remove(target_cell)

        is_active = False
        for c in prereqs:
            if c not in mask:
                is_active = True
                break
            if c in reachable:
                is_active = True
                break

        if is_active and first_active_idx is None:
            first_active_idx = i

        reachable |= verts

    if first_active_idx is None:
        return 0.0

    w = 1.0
    for eid in chain.edges[first_active_idx:]:
        w *= float(edge_w[eid])
    return float(w)


def _mask_disjoint_select(
    chains: List[_Chain],
    *,
    mask: Set[str],
    target_cell: str,
    edge_verts: Dict[str, Set[str]],
    edge_w: Dict[str, float],
) -> Tuple[List[_Chain], float]:
    """
    Algorithm 'MaskDisjointSelect': select a greedy mask-disjoint subset D*
    and compute P_all = Π_{p in D*} (1 - w(p, M)).
    """
    active_chains: List[_Chain] = []
    for p in chains:
        if not p.active:
            continue
        p.masked = (p.cells & mask) - {target_cell}
        p.currWeight = _compute_chain_weight(
            p, mask=mask, target_cell=target_cell, edge_verts=edge_verts, edge_w=edge_w
        )
        if p.currWeight > 0.0:
            active_chains.append(p)

    active_chains.sort(key=lambda p: float(p.currWeight), reverse=True)

    D_star: List[_Chain] = []
    used_cells: Set[str] = set()
    P_all = 1.0

    for p in active_chains:
        if p.masked.isdisjoint(used_cells):
            D_star.append(p)
            used_cells |= p.masked
            P_all *= (1.0 - float(p.currWeight))

    return D_star, float(P_all)


def _compute_premoved(
    D_star: List[_Chain],
    candidates: Set[str],
) -> Dict[str, float]:
    """
    Algorithm 'ComputePRemoved'.
    Returns P_removed[c] = Π_{p in D*, c in p.cells} (1 - p.currWeight).
    """
    P_removed: Dict[str, float] = {c: 1.0 for c in candidates}
    if not candidates:
        return P_removed

    for p in D_star:
        one_minus = 1.0 - float(p.currWeight)
        for c in p.cells:
            if c in P_removed:
                P_removed[c] *= one_minus

    return P_removed


def _build_chains_and_index(
    *,
    hypergraph: Hypergraph,
    target_cell: str,
    tau: Optional[float] = None,
) -> Tuple[List[_Chain], Dict[str, List[_Chain]], Dict[str, Set[str]], Dict[str, float]]:
    """
    Phase 1 preprocessing:
      - Convert hypergraph to edge_dict (edge_id -> (verts, weight))
      - Enumerate chains once
      - Build CellToChains inverted index

    Returns:
      chains, cell_to_chains, edge_verts, edge_w
    """
    # leakage.py helper: edge_dict is {edge_id: (set(verts), weight)}
    from leakage import hypergraph_to_edge_dict, iter_chains_with_masked

    edge_dict = hypergraph_to_edge_dict(hypergraph, tau=tau)
    edge_verts = {eid: set(vs) for eid, (vs, _w) in edge_dict.items()}
    edge_w = {eid: float(_w) for eid, (_vs, _w) in edge_dict.items()}

    # Enumerate chains in the hypergraph. We enumerate under an empty mask
    # to get the *structural* chain edge sequences; weights are recomputed per-iteration.
    chains: List[_Chain] = []
    cell_to_chains: Dict[str, List[_Chain]] = {}

    empty_mask: Set[str] = set()
    for (edge_ids, _masked_cells) in iter_chains_with_masked(empty_mask, target_cell, edge_dict):
        # edge_ids is a list[str] of edge identifiers
        cells: Set[str] = set()
        for eid in edge_ids:
            cells |= edge_verts.get(eid, set())
        p = _Chain(edges=list(edge_ids), cells=cells, active=True, currWeight=0.0)
        chains.append(p)
        for c in cells:
            cell_to_chains.setdefault(c, []).append(p)

    return chains, cell_to_chains, edge_verts, edge_w


def _p2e2_pointer_size() -> int:
    """Approx bytes for a reference/pointer on this Python build (usually 8 on 64-bit)."""
    return struct.calcsize("P")

def _p2e2_int_size() -> int:
    """Approx size of a Python int object (implementation-dependent but consistent per runtime)."""
    return sys.getsizeof(0)

def _p2e2_bool_size() -> int:
    """Approx size of a Python bool object."""
    return sys.getsizeof(False)

def _p2e2_container_overhead(obj: Any) -> int:
    """
    Container overhead only (not deep). Used to add overhead for sets/dicts/lists used in the model.
    """
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0

def _p2e2_sample_avg_size(items: Iterable[Any], max_samples: int = 64) -> int:
    """
    Average sys.getsizeof() of a sample of identifiers (e.g., vertex keys).
    This makes the estimate reflect whether vertices are strings, tuples, etc.
    """
    total = 0
    n = 0
    for x in items:
        total += sys.getsizeof(x)
        n += 1
        if n >= max_samples:
            break
    return int(total / n) if n > 0 else 0

def _p2e2_hg_vertices(hg: Any) -> Tuple[int, Optional[Iterable[Any]]]:
    """
    Best-effort vertex extraction. Returns (count, iterable_of_vertex_ids or None).
    Works with common layouts: hg.vertices, hg.V, hg.nodes, hg.adj keys.
    """
    if hg is None:
        return 0, None

    for attr in ("vertices", "V", "nodes"):
        if hasattr(hg, attr):
            v = getattr(hg, attr)
            try:
                return len(v), v
            except Exception:
                # if it's a generator or custom object
                try:
                    vv = list(v)
                    return len(vv), vv
                except Exception:
                    pass

    # adjacency dict keyed by vertex
    for attr in ("adj", "out_edges", "in_edges"):
        if hasattr(hg, attr):
            v = getattr(hg, attr)
            try:
                return len(v), v.keys() if hasattr(v, "keys") else v
            except Exception:
                pass

    return 0, None

def _p2e2_hg_edges_and_total_tail_elems(hg: Any) -> Tuple[int, int]:
    """
    Best-effort edge extraction.
    Returns:
      E = number of hyperedges
      T = total tail elements across all hyperedges (sum |edge.tail| or |edge|)
    This is the key P2E2 term: edge storage scales with sum of arities.
    """
    if hg is None:
        return 0, 0

    # If your Hypergraph stores explicit hyperedges in a list
    for attr in ("hyperedges", "edges", "E"):
        if hasattr(hg, attr):
            edges = getattr(hg, attr)
            try:
                # edges could be list[tuple[head, tail]] or list[edge_obj]
                E = len(edges)
                total_tail = 0
                for e in edges:
                    # cases:
                    #   - (head, tail_iterable)
                    #   - edge object with .tail
                    #   - edge iterable of tail nodes (rare)
                    if isinstance(e, tuple) and len(e) == 2:
                        tail = e[1]
                        try:
                            total_tail += len(tail)
                        except Exception:
                            total_tail += len(list(tail))
                    elif hasattr(e, "tail"):
                        tail = getattr(e, "tail")
                        try:
                            total_tail += len(tail)
                        except Exception:
                            total_tail += len(list(tail))
                    else:
                        # fallback: treat e itself as tail container
                        try:
                            total_tail += len(e)
                        except Exception:
                            total_tail += len(list(e))
                return E, total_tail
            except Exception:
                pass

    # If your Hypergraph stores head->list_of_tails adjacency
    # e.g., hg.out[head] = [tail1, tail2, ...] where tail is iterable
    for attr in ("out", "outgoing", "head_to_tails", "cell2Edge"):
        if hasattr(hg, attr):
            out = getattr(hg, attr)
            try:
                E = 0
                total_tail = 0
                items = out.items() if hasattr(out, "items") else out
                for _, tails in items:
                    # tails could be list of hyperedges, each hyperedge is iterable of tail vertices
                    for t in tails:
                        E += 1
                        try:
                            total_tail += len(t)
                        except Exception:
                            total_tail += len(list(t))
                return E, total_tail
            except Exception:
                pass

    return 0, 0

def p2e2_estimate_model_memory_bytes(hg: Any, mask: Optional[Iterable[Any]] = None) -> int:
    """
    P2E2-style standardized memory estimate, adapted to Python.

    Mirrors the paper's idea:
      memory ≈ (per-vertex storage)*|V| + (per-edge storage)*|E| + (per-tail-pointer)*sum_arities
    but uses Python runtime object sizes for pointer/int/bool and samples vertex-id sizes.

    NOTE: this is an estimate meant for consistent comparison within your experiments.
    """
    ptr = _p2e2_pointer_size()
    i_sz = _p2e2_int_size()
    b_sz = _p2e2_bool_size()

    V, vertex_ids = _p2e2_hg_vertices(hg)
    E, total_tail = _p2e2_hg_edges_and_total_tail_elems(hg)

    avg_vid = _p2e2_sample_avg_size(vertex_ids if vertex_ids is not None else [], max_samples=64)

    # ---- Per-vertex fields (P2E2 Java had: table idx, row idx, insertionTime, state, cost)
    # Python analogue:
    #   - an identifier (string/tuple) => avg_vid
    #   - a few scalar fields that exist per node in your model bookkeeping
    #   - references in adjacency structures (pointers)
    #
    # We model "a node record" as:
    #   id + 3 ints + 1 bool + 2 pointers  (reasonable analogue to P2E2)
    per_vertex = avg_vid + (3 * i_sz) + b_sz + (2 * ptr)

    # ---- Per-edge fields (P2E2 Java had: pointer head->edge + minCell + per-element pointer)
    # Python analogue:
    #   - overhead for edge container object + a couple pointers (head reference, cached min, etc.)
    #   - per-tail element pointer/reference
    #
    # We treat each hyperedge as:
    #   fixed: 3 pointers + 1 int, and tail pointers: total_tail * ptr
    per_edge_fixed = (3 * ptr) + i_sz
    per_tail_ptr = ptr

    base = (V * per_vertex) + (E * per_edge_fixed) + (total_tail * per_tail_ptr)

    # ---- Add shallow container overheads that are definitely present
    base += _p2e2_container_overhead(mask) if mask is not None else 0

    # If your hypergraph has big containers, you can optionally add their shallow overhead
    # (not deep) so datasets with huge dicts/lists show that cost.
    for attr in ("vertices", "V", "nodes", "adj", "out", "hyperedges", "edges"):
        if hasattr(hg, attr):
            base += _p2e2_container_overhead(getattr(hg, attr))

    return int(base)
# -----------------------------------------
# everything below is just Gumbel greedy
# -----------------------------------------
def dcs_to_hyperedges_and_weights(
    dataset: str,
    denial_constraints: List[List[Tuple[str, str, str]]],
) -> Tuple[List[Tuple[str, ...]], List[float]]:
    """
    Build schema-level hyperedges (sorted attribute tuples) + aligned weights
    using delexp weights convention (index matches denial_constraints order).
    """
    weights_obj = get_dataset_weights(dataset)

    rdrs: List[Tuple[str, ...]] = []
    rdr_weights: List[float] = []

    for i, dc in enumerate(denial_constraints or []):
        attrs: Set[str] = set()
        for pred in dc:
            if not isinstance(pred, (list, tuple)) or len(pred) < 3:
                continue
            for tok in (pred[0], pred[2]):
                if isinstance(tok, str) and "." in tok:
                    attrs.add(tok.split(".")[-1])

        if len(attrs) >= 2:
            rdrs.append(tuple(sorted(attrs)))
            try:
                rdr_weights.append(float(weights_obj[i]))
            except Exception:
                rdr_weights.append(1.0)

    return rdrs, rdr_weights
import os
import pickle
import re

# --- add these helper functions anywhere above gumbel_deletion_main ---

def _safe_filename(s: str) -> str:
    """Make a string safe for filenames."""
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_") or "obj"

def save_hypergraph_pkl(
    hg: Any,
    *,
    dataset: str,
    target_cell: str,
    mode: str = "MAX",
    out_dir: str = "hypergraphs_pkl",
) -> Tuple[str, int, bool]:
    """
    Save hypergraph to a pickle file and return (path, file_size_bytes, ok).

    If pickling the object fails (e.g., contains non-pickleable members),
    we fall back to pickling a lightweight dict snapshot (hg.__dict__) so
    you still get a stable file size metric.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"hg_{_safe_filename(dataset)}_{_safe_filename(target_cell)}_{_safe_filename(mode)}.pkl"
    path = os.path.join(out_dir, fname)

    ok = True
    try:
        with open(path, "wb") as f:
            pickle.dump(hg, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        ok = False
        # Fallback: serialize a snapshot instead of the object itself
        snapshot = {
            "__class__": type(hg).__name__,
            "__module__": type(hg).__module__,
            "__dict__": getattr(hg, "__dict__", None),
            # optional: try to include common fields if present
            "vertices": getattr(hg, "vertices", None) if hasattr(hg, "vertices") else None,
            "edges": getattr(hg, "edges", None) if hasattr(hg, "edges") else None,
            "hyperedges": getattr(hg, "hyperedges", None) if hasattr(hg, "hyperedges") else None,
            "out": getattr(hg, "out", None) if hasattr(hg, "out") else None,
            "adj": getattr(hg, "adj", None) if hasattr(hg, "adj") else None,
        }
        with open(path, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_bytes = os.path.getsize(path)
    return path, int(size_bytes), ok
def gumbel_noise(scale: float) -> float:
    u = random.random()
    u = max(1e-10, min(1.0 - 1e-10, u))
    return float(-scale * np.log(-np.log(u)))
def gumbel_noise_l(epsilon_prime: float, lam: float) -> float:
    """
    Gumbel noise for exponential mechanism with sensitivity Δu = lam.
    Scale b = 2Δu / ε' = 2λ / ε'
    """
    if epsilon_prime <= 0.0:
        raise ValueError("epsilon_prime must be > 0")

    b = (2.0) / epsilon_prime #2*Lam
    u = random.random()
    u = max(1e-10, min(1.0 - 1e-10, u))
    return -b * np.log(-np.log(u))

def inference_zone_union(target: str, hypergraph):
    union = set()
    for vertex in hypergraph.vertices:
        if vertex != target:
            union.add(vertex)
    return union
def marginal_gain(
    *,
    c: str,
    M_curr: Set[str],
    hypergraph: Hypergraph,
    target_cell: str,
    lam: float,
    denom_I_minus_1: int,
    leakage_method: str,
    L0 = 0.25,
    edge_tau: Optional[float],
) -> Tuple[float, float, float]:
    L_curr = chain_leakage(
        M_curr, target_cell, hypergraph,

        leakage_method=leakage_method,
        return_counts=False,
    )
    L_new = chain_leakage(
        M_curr | {c}, target_cell, hypergraph,
        leakage_method=leakage_method,
        return_counts=False,
    )

    U_curr = compute_utility_max(
        leakage = L_curr,
        mask_size = len(M_curr),
        lam = lam,
        zone_size = denom_I_minus_1 + 1,
        L0 = L0
    )

    U_new = compute_utility_max(
        leakage = L_new,
        mask_size = len(M_curr) + 1,
        lam = lam,
        zone_size = denom_I_minus_1 + 1,
        L0 = L0
    )

    delta_u = U_new - U_curr
    return float(delta_u), float(L_curr), float(L_new)

def marginal_gain_given_curr(
    *,
    c: str,
    M_curr: Set[str],
    hypergraph: Hypergraph,
    target_cell: str,
    lam: float,
    L0: float,
    zone_size: int,
    leakage_method: str,
    L_curr: float,
    U_curr: float,
) -> Tuple[float, float, float, float]:
    """
    Efficient marginal gain helper for Version B.

    Inputs:
      - L_curr, U_curr are the CURRENT leakage/utility under M_curr
        (precomputed once per iteration in the caller).

    Returns:
      (delta_u, L_new, U_new, L_curr)  # includes curr for convenience/debug
    """
    L_new = chain_leakage(
        M_curr | {c},
        target_cell,
        hypergraph,
        leakage_method=leakage_method,
        return_counts=False,
    )

    U_new = compute_utility_max(
        leakage=float(L_new),
        mask_size=len(M_curr) + 1,
        lam=float(lam),
        zone_size=int(zone_size),
        L0=float(L0),
    )

    delta_u = float(U_new) - float(U_curr)
    return float(delta_u), float(L_new), float(U_new), float(L_curr)


# def greedy_gumbel_max_deletion(
#     *,
#     hypergraph: Hypergraph,
#     hyperedges: List[Tuple[str, ...]],  # kept for backward-compat; unused (hypergraph already contains edges)
#     target_cell: str,
#     lam: float,
#     epsilon: float,
#     L0: float,
#     K: int,
#     leakage_method: str = "greedy_disjoint",  # kept for signature compatibility
#     edge_tau: Optional[float] = None,
# ) -> Tuple[Set[str], float]:
#     """
#     DelGum: Fast Greedy Gumbel-Max Deletion (paper-style implementation).
#
#     Key differences vs the older version in this repo:
#       - Preprocesses *all chains once* using leakage.py iter_chains_with_masked + hypergraph_to_edge_dict.
#       - Per-iteration updates only affected chains via an inverted index (cell -> chains).
#       - Uses the closed-form lower bound ΔL_lb(c) and per-iteration Gumbel-Max selection.
#       - Includes DP-compatible early stopping as a 'stop' action with Gumbel noise.
#
#     Returns:
#       (mask M, wall_time_seconds)
#     """
#     t0 = time.time()
#
#     if K <= 0 or epsilon <= 0.0:
#         return set(), float(time.time() - t0)
#     # ----------------------------
#     # Phase 1: preprocessing
#     # ----------------------------
#     chains, cell_to_chains, edge_verts, edge_w = _build_chains_and_index(
#         hypergraph=hypergraph,
#         target_cell=target_cell,
#         tau=edge_tau,
#     )
#
#     # Inference zone I(c*): union of vertices reachable in local hypergraph.
#     # We reuse your existing helper (union over edges).
#     I = inference_zone_union(target_cell, hypergraph)
#     I.discard(target_cell)
#
#     M: Set[str] = set()
#     epsilon_prime = float(epsilon) / float(K)
#
#     # ----------------------------
#     # Phase 2: iterative selection
#     # ----------------------------
#     for _k in range(1, K + 1):
#         # Step 2a: current leakage state under mask-disjoint approx
#         D_star, P_all = _mask_disjoint_select(
#             chains,
#             mask=M,
#             target_cell=target_cell,
#             edge_verts=edge_verts,
#             edge_w=edge_w,
#         )
#
#         # If leakage already ~0, early stop quickly (still DP-safe due to stop action below).
#         candidates = set(I) - set(M)
#         if not candidates:
#             break
#
#         # Step 2b: compute P_removed(c) for all candidates
#         P_removed = _compute_premoved(D_star, candidates)
#
#         # Step 2c: score candidates with ΔL_lb and add Gumbel noise
#         best_c: Optional[str] = None
#         best_score = -float("inf")
#
#         # Constant deletion cost term as written in your pseudocode
#         size_penalty = (1.0 - float(lam)) * (1.0 / float(max(1, len(I))))
#
#         for c in candidates:
#             pr = float(P_removed.get(c, 1.0))
#
#             # Guard against divide-by-zero; if pr==0 then keeping prob is undefined but
#             # it means this candidate would remove essentially all surviving probability.
#             if pr <= 0.0:
#                 P_keep = 0.0
#                 delta_L_lb = float(P_all)  # maximal
#             else:
#                 P_keep = float(P_all) / pr
#                 delta_L_lb = P_keep * (1.0 - pr)
#
#             delta_u = (float(lam) * float(delta_L_lb)) - float(size_penalty)
#
#             # Gumbel noise with scale = 2*lam/epsilon' (matches your algorithm listing)
#             u = random.random()
#             g = -(2.0 * float(lam) / float(epsilon_prime)) * math.log(-math.log(max(1e-12, u)))
#             score = float(delta_u) + float(g)
#
#             if score > best_score:
#                 best_score = score
#                 best_c = c
#
#         # Step 2d: early stopping action (zero utility + Gumbel noise)
#         u_stop = random.random()
#         g_stop = -(2.0 * float(lam) / float(epsilon_prime)) * math.log(-math.log(max(1e-12, u_stop)))
#         s_stop = float(g_stop)
#
#         if best_c is None or s_stop > best_score:
#             break
#
#         # Step 2e: commit choice and update only affected chains
#         M.add(best_c)
#
#         affected = cell_to_chains.get(best_c, [])
#         for p in affected:
#             if not p.active:
#                 continue
#             w_new = _compute_chain_weight(
#                 p, mask=M, target_cell=target_cell, edge_verts=edge_verts, edge_w=edge_w
#             )
#             p.currWeight = float(w_new)
#             if w_new <= 0.0:
#                 p.active = False
#
#     return M, float(time.time() - t0)
#
#
#     denom_I_minus_1 = max(1, len(I) - 1)
#
#     epsilon_prime = float(epsilon) / float(K)
#     g_scale = (2.0 * float(lam)) / max(1e-12, epsilon_prime)
#
#     for _k in range(1, K + 1):
#         candidates = sorted(I - M)
#         if not candidates:
#             break
#
#         best_c = None
#         best_score = -1e300
#
#         for c in candidates:
#             delta_u, _Lc, _Ln = marginal_gain(
#                 c=c,
#                 M_curr=M,
#                 hypergraph=hypergraph,
#                 target_cell=target_cell,
#                 lam=lam,
#                 L0 = L0,
#                 denom_I_minus_1=denom_I_minus_1,
#                 leakage_method=leakage_method,
#                 edge_tau=edge_tau,
#             )
#             score = float(delta_u) + gumbel_noise_l(epsilon_prime, lam)
#             if score > best_score:
#                 best_score = score
#                 best_c = c
#
#         s_stop = gumbel_noise_l(epsilon_prime, lam)
#         if best_c is None:
#             break
#         if s_stop > best_score:
#             break
#
#         M.add(best_c)
#
#     return M, float(time.time() - t0)
from typing import Optional, Set, Tuple, List, Dict
import math
import random
import time
def gumbel_noise_scale(scale: float) -> float:
    # g = -scale * ln(-ln(u)),  u ~ Uniform(0,1)
    u = random.random()
    u = max(1e-12, min(1.0 - 1e-12, u))
    return float(-scale * math.log(-math.log(u)))

def greedy_gumbel_max_deletion(
    *,
    hypergraph: Hypergraph,
    hyperedges: List[Tuple[str, ...]],  # kept for backward-compat; unused
    target_cell: str,
    lam: float,       # kept for signature compatibility; NOT USED in LaTeX scoring
    epsilon: float,
    L0: float,
    K: int,
    leakage_method: str = "greedy_disjoint",  # kept for signature compatibility
    edge_tau: Optional[float] = None,
) -> Tuple[Set[str], float]:
    """
    MATCHES the LaTeX Algorithm DelGum exactly (Phase 2 scoring):

      - Uses MaskDisjointSelect -> (D*, P_all)
      - L_curr = 1 - P_all
      - Reward R = max(0, L0 - L)
      - Candidate L_upper = 1 - P_all / P_removed[c]
      - Δu_lb(c) = R_new - R_curr - 1/|Z|
      - Gumbel noise scale = 2/ε' (NO λ factor)
      - Stop action uses score = 0 + Gumbel(2/ε')

    NOTE: `lam` is not part of the LaTeX algorithm’s scoring; it is unused here.
    """

    t0 = time.time()

    if K <= 0 or epsilon <= 0.0:
        return set(), float(time.time() - t0)

    # Phase 1: preprocessing
    chains, cell_to_chains, edge_verts, edge_w = _build_chains_and_index(
        hypergraph=hypergraph,
        target_cell=target_cell,
        tau=edge_tau,
    )

    # Z = I(c*)
    Z = inference_zone_union(target_cell, hypergraph)
    Z.discard(target_cell)

    M: Set[str] = set()
    epsilon_prime = float(epsilon) / float(K)

    # Gumbel scale in LaTeX: 2/ε'
    g_scale = 2.0 / max(1e-12, float(epsilon_prime))

    for _k in range(1, K + 1):
        candidates = set(Z) - set(M)
        if not candidates:
            break

        # (D*, P_all) = MaskDisjointSelect(P, M)
        D_star, P_all = _mask_disjoint_select(
            chains,
            mask=M,
            target_cell=target_cell,
            edge_verts=edge_verts,
            edge_w=edge_w,
        )

        # L_curr = 1 - P_all
        L_curr = 1.0 - float(P_all)

        # R_curr = max(0, L0 - L_curr)
        R_curr = max(0.0, float(L0) - float(L_curr))

        # Precompute P_removed[c] for all candidates
        P_removed = _compute_premoved(D_star, candidates)

        # Cost term in LaTeX: 1/|Z|
        cost = 1.0 / float(max(1, len(Z)))

        best_c: Optional[str] = None
        best_score = -float("inf")

        # Score each candidate
        for c in candidates:
            pr = float(P_removed.get(c, 1.0))

            # L_upper = 1 - P_all / P_removed[c]
            # Guard pr<=0
            if pr <= 0.0:
                # If pr is ~0, then P_all/pr is huge; L_upper -> -inf, reward saturates at L0
                L_upper = -float("inf")
            else:
                L_upper = 1.0 - (float(P_all) / pr)

            # R_new = max(0, L0 - L_upper)
            R_new = max(0.0, float(L0) - float(L_upper))

            # Δu_lb = R_new - R_curr - 1/|Z|
            delta_u_lb = float(R_new) - float(R_curr) - float(cost)

            # g_c = Gumbel(scale = 2/ε')
            g_c = gumbel_noise_scale(g_scale)

            score = float(delta_u_lb) + float(g_c)

            if score > best_score:
                best_score = score
                best_c = c

        # Early stop option: s_stop = 0 + Gumbel(scale = 2/ε')
        s_stop = gumbel_noise_scale(g_scale)
        if best_c is None or s_stop > best_score:
            break

        # Select and update
        M.add(best_c)

        # UpdateChainWeights(P, CellToChains[c*], M)
        affected = cell_to_chains.get(best_c, [])
        for p in affected:
            if not p.active:
                continue
            w_new = _compute_chain_weight(
                p,
                mask=M,
                target_cell=target_cell,
                edge_verts=edge_verts,
                edge_w=edge_w,
            )
            p.currWeight = float(w_new)
            if w_new <= 0.0:
                p.active = False

    return M, float(time.time() - t0)


def gumbel_deletion_main(
    dataset: str,
    target_cell: str,
    *,
    epsilon: float = 1.0,
    lam: float = 0.5,
    K: int = 40,
    L0: float = 0.25,
    leakage_method: str = "greedy_disjoint",
    edge_tau: Optional[float] = None,
) -> Dict[str, Any]:

    init_start = time.time()

    # load denial_constraints from your existing dc_configs module
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

    # ✅ build hyperedges + weights using the IMPORTED heavy-lifting function
    hyperedges, weights = dcs_to_hyperedges_and_weights(dataset, raw_dcs)

    # ✅ build local hypergraph once (also imported)
    H_local = construct_local_hypergraph(target_cell, hyperedges, weights, mode="MAX")
    hg_pkl_path, hg_pkl_size, hg_pickled_ok = save_hypergraph_pkl(
        H_local,
        dataset = dataset,
        target_cell = target_cell,
        mode = "MAX",
        out_dir = "hypergraphs_pkl",
    )
    float_sz = sys.getsizeof(0.0)
    memory_bytes = int(hg_pkl_size + float_sz)
    init_time = float(time.time() - init_start)

    model_start = time.time()

    final_mask, greedy_time = greedy_gumbel_max_deletion(
        hypergraph=H_local,
        hyperedges=hyperedges,
        target_cell=target_cell,
        lam=lam,
        epsilon=epsilon,
        K=K,
        L0=L0,
        leakage_method=leakage_method,
        edge_tau=edge_tau,
    )

    # leakage + chain counts (uses imported leakage(..., return_counts=True))
    L, num_chains, active_chains, blocked_chains = chain_leakage(
        final_mask,
        target_cell,
        H_local,
        leakage_method=leakage_method,
        return_counts=True,
    )
    base_leak = chain_leakage(set(), target_cell, H_local, leakage_method=leakage_method)

    inference_zone = inference_zone_union(target_cell, H_local)
    denom = max(1, len(inference_zone))
    utility = compute_utility_max(leakage = L, mask_size = len(final_mask),lam =lam, zone_size = len(inference_zone), L0 = L0)

    model_time = float(time.time() - model_start)

    return {
        "init_time": init_time,
        "model_time": model_time,
        "del_time": 0.0,

        "leakage": float(L),
        "utility": float(utility),
        "mask": final_mask,
        "mask_size": int(len(final_mask)),

        # ✅ now these are real chain counts, non-negative
        "num_paths": int(num_chains),
        "baseline_leakage": float(base_leak),
        "memory_overhead_bytes": memory_bytes,
        "num_instantiated_cells": int(len(set(inference_zone))),
    }


if __name__ == "__main__":
    print(gumbel_deletion_main("flight", "FlightNum"))
    print(gumbel_deletion_main("adult", "education"))
    print(gumbel_deletion_main("airport", "scheduled_service"))
    print(gumbel_deletion_main("hospital", "ProviderNumber"))
    print(gumbel_deletion_main("tax", "marital_status"))
# init, model, del, total, leakage, num paths, memory, mask size, instantiated