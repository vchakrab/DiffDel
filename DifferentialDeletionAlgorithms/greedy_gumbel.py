# greedy_gumbel.py

from __future__ import annotations

import time
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np

# ✅ import YOUR heavy lifting module
from leakage import (
    get_dataset_weights,   # (if you add the helper)

    construct_local_hypergraph,
    Hypergraph,

    leakage as chain_leakage,
)

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

def gumbel_noise(scale: float) -> float:
    u = random.random()
    u = max(1e-10, min(1.0 - 1e-10, u))
    return float(-scale * np.log(-np.log(u)))


def inference_zone_union(target: str, hyperedges: Iterable[Iterable[str]]) -> List[str]:
    z: Set[str] = set()
    for e in hyperedges:
        z |= set(e)
    z.discard(target)
    return sorted(z)


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
        edge_tau=edge_tau,
        leakage_method=leakage_method,
        return_counts=False,
    )
    L_new = chain_leakage(
        M_curr | {c}, target_cell, hypergraph,
        edge_tau=edge_tau,
        leakage_method=leakage_method,
        return_counts=False,
    )

    penalty = (1.0 - float(lam)) / float(denom_I_minus_1)
    delta_u = float(lam) * (float(L_curr) - float(L_new)) - penalty
    return float(delta_u), float(L_curr), float(L_new)


def greedy_gumbel_max_deletion(
    *,
    hypergraph: Hypergraph,
    hyperedges: List[Tuple[str, ...]],
    target_cell: str,
    lam: float,
    epsilon: float,
    K: int,
    leakage_method: str = "noisy_or",     # or "greedy_disjoint"
    edge_tau: Optional[float] = None,
) -> Tuple[Set[str], float]:
    t0 = time.time()
    I = set(inference_zone_union(target_cell, hyperedges))
    M: Set[str] = set()

    if K <= 0 or epsilon <= 0:
        return M, float(time.time() - t0)

    denom_I_minus_1 = max(1, len(I) - 1)

    epsilon_prime = float(epsilon) / float(K)
    g_scale = (2.0 * float(lam)) / max(1e-12, epsilon_prime)

    for _k in range(1, K + 1):
        candidates = list(I - M)
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
            score = float(delta_u) + gumbel_noise(g_scale)
            if score > best_score:
                best_score = score
                best_c = c

        s_stop = gumbel_noise(g_scale)
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
    leakage_method: str = "noisy_or",
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
        edge_tau=edge_tau,
        leakage_method=leakage_method,
        return_counts=True,
    )

    inference_zone = inference_zone_union(target_cell, hyperedges)
    denom = max(1, len(inference_zone))
    utility = float(-1 * lam * float(L) - ((1 - lam) * len(final_mask)) / denom)

    model_time = float(time.time() - model_start)

    return {
        "init_time": init_time,
        "model_time": model_time,
        "del_time": 0.0,

        "leakage": float(L),
        "utility": float(utility),
        "mask_size": int(len(final_mask)),
        "mask": set(final_mask),

        # ✅ now these are real chain counts, non-negative
        "num_paths": int(num_chains),
        "paths_blocked": int(blocked_chains),

        "lambda": float(lam),
        "inference_zone_size": int(len(set(inference_zone))),
        "denom_I_minus_1": int(max(1, len(set(inference_zone)) - 1)),
        "greedy_time": float(greedy_time),
    }
