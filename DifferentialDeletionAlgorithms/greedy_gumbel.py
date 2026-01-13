# greedy_gumbel.py

from __future__ import annotations

import time
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np

from DifferentialDeletionAlgorithms.leakage import compute_utility_logarithmic, compute_utility, compute_utility_hinge
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
    if lam <= 0.0:
        raise ValueError("lam must be > 0")

    b = (2.0 * lam) / epsilon_prime
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

    U_curr = compute_utility(
        leakage = L_curr,
        mask_size = len(M_curr),
        lam = lam,
        zone_size = denom_I_minus_1 + 1,
    )

    U_new = compute_utility(
        leakage = L_new,
        mask_size = len(M_curr) + 1,
        lam = lam,
        zone_size = denom_I_minus_1 + 1,
    )

    delta_u = U_new - U_curr
    return float(delta_u), float(L_curr), float(L_new)


def greedy_gumbel_max_deletion(
    *,
    hypergraph: Hypergraph,
    hyperedges: List[Tuple[str, ...]],
    target_cell: str,
    lam: float,
    epsilon: float,
    K: int,
    leakage_method: str = "greedy_disjoint",     # or "greedy_disjoint"
    edge_tau: Optional[float] = None,
) -> Tuple[Set[str], float]:
    t0 = time.time()
    I = inference_zone_union(target_cell, hypergraph)
    M: Set[str] = set()

    if K <= 0 or epsilon <= 0:
        return M, float(time.time() - t0)

    denom_I_minus_1 = max(1, len(I) - 1)

    epsilon_prime = float(epsilon) / float(K)
    g_scale = (2.0 * float(lam)) / max(1e-12, epsilon_prime)

    for _k in range(1, K + 1):
        candidates = sorted(I - M)
        if not candidates:
            break

        best_c = None
        best_score = -1e300

        for c in candidates:
            delta_u, _Lc, _Ln = marginal_gain(
                c=c,
                M_curr=M,
                hypergraph=hypergraph,
                target_cell=target_cell,
                lam=lam,
                denom_I_minus_1=denom_I_minus_1,
                leakage_method=leakage_method,
                edge_tau=edge_tau,
            )
            score = float(delta_u) + gumbel_noise_l(epsilon_prime, lam)
            if score > best_score:
                best_score = score
                best_c = c

        s_stop = gumbel_noise_l(epsilon_prime, lam)
        if best_c is None:
            break
        if s_stop > best_score:
            break

        M.add(best_c)

    return M, float(time.time() - t0)


def gumbel_deletion_main(
    dataset: str,
    target_cell: str,
    *,
    epsilon: float = 1.0,
    lam: float = 0.5,
    K: int = 40,
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
    utility = compute_utility(leakage = L, mask_size = len(final_mask),lam =lam, zone_size = len(inference_zone))

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
    print(gumbel_deletion_main("flight", "FlightNum", K=13))
    print(gumbel_deletion_main("adult", "education"))
    print(gumbel_deletion_main("airport", "scheduled_service"))
    print(gumbel_deletion_main("hospital", "ProviderNumber"))
    print(gumbel_deletion_main("tax", "marital_status"))
# init, model, del, total, leakage, num paths, memory, mask size, instantiated