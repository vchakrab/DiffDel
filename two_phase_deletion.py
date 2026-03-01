#!/usr/bin/env python3
"""
del2ph.py (two-phase deletion)
"""

from __future__ import annotations

import os
import time
import pickle
import sys
import numpy as np
from leakage import (
    dc_to_rdrs_and_weights,
    construct_hypergraph_max,
    construct_hypergraph_actual,
    compute_utility_em,
    leakage as leakage_model,
)
from typing import Any, Dict, Iterable, Optional, Tuple
from itertools import chain, combinations

def template_pkl_path(
    dataset: str,
    target_attribute: str,
    epsilon: float,
    L0: float,
    lambda_penalty: float,
    hypergraph_mode: str,
    save_dir: str,
) -> str:
    fname = (
        f"{dataset}_"
        f"{target_attribute}_"
        f"eps{epsilon}_"
        f"L0{L0}_"
        f"lam{lambda_penalty}_"
        f"{hypergraph_mode}.pkl"
    )
    return os.path.join(save_dir, fname)


def get_template_pkl_size_bytes(
    dataset: str,
    target_attribute: str,
    epsilon: float,
    L0: float,
    lambda_penalty: float,
    hypergraph_mode: str,
    save_dir: str,
) -> int:
    p = template_pkl_path(
        dataset,
        target_attribute,
        epsilon,
        L0,
        lambda_penalty,
        hypergraph_mode,
        save_dir,
    )
    try:
        return int(os.path.getsize(p))
    except Exception:
        return 0

def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.array([], dtype=float)
    m = float(np.max(scores))
    ex = np.exp(scores - m)
    z = float(np.sum(ex))
    if z <= 0.0 or not np.isfinite(z):
        return np.ones_like(scores, dtype=float) / max(1, scores.size)
    return ex / z


def exp_mech_probs(utilities: np.ndarray, epsilon: float, zone_size: int) -> np.ndarray:
    if utilities.size == 0:
        return np.array([], dtype=float)

    z = max(1, int(zone_size))
    sens = 1.0 / float(z)
    scores = (float(epsilon) * utilities.astype(float)) / (2.0 * sens)
    return stable_softmax(scores)

def load_parsed_dcs_for_dataset(dataset: str):
    ds = str(dataset).lower()
    mod_name = "NCVoter" if ds == "ncvoter" else ds.capitalize()
    dc_module_path = f"DCandDelset.dc_configs.top{mod_name}DCs_parsed"
    try:
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        return getattr(dc_module, "denial_constraints", []) or []
    except Exception:
        return []

def build_template_two_phase(
    dataset: str,
    target_attribute: str,
    *,
    save_dir: str,
    epsilon: float,
    lam: float,
    L0: float,
    lambda_penalty: float,
    tau: Optional[float],
    leakage_method: str,
    hypergraph_mode: str,
    mask_space,
) -> Dict[str, Any]:

    raw_dcs = load_parsed_dcs_for_dataset(dataset)

    class _Init:
        def __init__(self, dataset, denial_constraints):
            self.dataset = dataset
            self.denial_constraints = denial_constraints

    init_manager = _Init(dataset, raw_dcs)
    rdrs, rdr_weights = dc_to_rdrs_and_weights(init_manager)

    if str(hypergraph_mode).upper() == "ACTUAL":
        H = construct_hypergraph_actual(target_attribute, rdrs, rdr_weights)
    else:
        H = construct_hypergraph_max(target_attribute, rdrs, rdr_weights)

    zone_set = set()
    for (edge_verts, _w) in getattr(H, "edges", []):
        for v in edge_verts:
            if v != target_attribute:
                zone_set.add(v)

    zone = sorted(zone_set)
    zone_size = len(zone)

    candidate_masks = [frozenset(m) for m in powerset(zone)] or [frozenset()]

    Leakage = {}
    Utility = {}
    Probability = {}

    utilities_arr = np.empty(len(candidate_masks), dtype=float)

    for i, m in enumerate(candidate_masks):
        L, num_chains, _, _ = leakage_model(
            mask=set(m),
            target_cell=target_attribute,
            hypergraph=H,
            tau=tau,
            return_counts=True,
        )

        U = compute_utility_em(
            leakage=float(L),
            mask_size=len(m),
            zone_size=int(zone_size),
            L0=float(L0),
            lambda_penalty=float(lambda_penalty),
        )

        Leakage[m] = float(L)
        Utility[m] = float(U)
        utilities_arr[i] = float(U)

    probs_arr = exp_mech_probs(utilities_arr, epsilon=float(epsilon), zone_size=zone_size)
    for m, p in zip(candidate_masks, probs_arr):
        Probability[m] = float(p)

    L_empty = leakage_model(mask=set(), target_cell=target_attribute, hypergraph=H, tau=tau)

    T = {
        "dataset": dataset,
        "target": target_attribute,
        "epsilon": epsilon,
        "lam": lam,
        "L0": L0,
        "hypergraph_mode": hypergraph_mode,
        "zone": zone,
        "R_intra": candidate_masks,
        "Leakage": Leakage,
        "Utility": Utility,
        "Probability": Probability,
        "baseline_leakage_empty_mask": float(L_empty),
        "num_instantiated_cells": int(len(zone)),
    }

    os.makedirs(save_dir, exist_ok=True)

    pkl_path = template_pkl_path(
        dataset,
        target_attribute,
        epsilon,
        L0,
        lambda_penalty,
        hypergraph_mode,
        save_dir,
    )

    with open(pkl_path, "wb") as f:
        pickle.dump(T, f, protocol=pickle.HIGHEST_PROTOCOL)

    return T

def two_phase_deletion_main(
    dataset: str,
    key: int,
    target_cell: str,
    *,
    epsilon: float = 50.0,
    lam: float = 0.5,
    L0: float = 0.25,
    lambda_penalty: float = 100.0,
    tau: Optional[float] = None,
    leakage_method: str = "greedy_disjoint",
    hypergraph_mode: str = "MAX",
    template_dir: str = "templates",
    mask_method: str = "None",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:

    if rng is None:
        rng = np.random.default_rng()

    template_path = template_pkl_path(
        dataset,
        target_cell,
        epsilon,
        L0,
        lambda_penalty,
        hypergraph_mode,
        template_dir,
    )

    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            T = pickle.load(f)
    else:
        if os.path.exists(template_path):
            with open(template_path, "rb") as f:
                T = pickle.load(f)
        else:
            build_start = time.time()

            T = build_template_two_phase(
                dataset,
                target_cell,
                save_dir = template_dir,
                epsilon = epsilon,
                lam = lam,
                L0 = L0,
                lambda_penalty = lambda_penalty,
                tau = tau,
                leakage_method = leakage_method,
                hypergraph_mode = hypergraph_mode,
                mask_space = mask_method,
            )

            build_time = float(time.time() - build_start)

            os.makedirs(template_dir, exist_ok = True)
            csv_path = os.path.join(template_dir, f"{dataset}_template_build_times.csv")

            file_exists = os.path.exists(csv_path)

            with open(csv_path, "a") as f:
                # write header once
                if not file_exists:
                    f.write(
                        "dataset,target,epsilon,L0,lambda_penalty,hypergraph_mode,build_time_seconds\n"
                    )

                f.write(
                    f"{dataset},"
                    f"{target_cell},"
                    f"{epsilon},"
                    f"{L0},"
                    f"{lambda_penalty},"
                    f"{hypergraph_mode},"
                    f"{build_time:.6f}\n"
                )

    template_bytes = get_template_pkl_size_bytes(
        dataset,
        target_cell,
        epsilon,
        L0,
        lambda_penalty,
        hypergraph_mode,
        template_dir,
    )

    py_float_bytes = int(sys.getsizeof(0.0))

    model_start = time.time()

    masks = T["R_intra"]
    probs_dict = T["Probability"]
    probs = np.array([probs_dict[m] for m in masks], dtype=float)

    s = float(probs.sum())
    if s <= 0.0 or not np.isfinite(s):
        probs = np.ones_like(probs) / max(1, probs.size)
    else:
        probs /= s

    idx = int(rng.choice(len(masks), p=probs))
    chosen = masks[idx]

    model_time = float(time.time() - model_start)

    leakage_val = float(T["Leakage"][chosen])
    utility_val = float(T["Utility"][chosen])
    leakage_empty = float(T["baseline_leakage_empty_mask"])

    memory_overhead_bytes = int(template_bytes + py_float_bytes)

    return {
        "init_time": 0.0,
        "model_time": float(model_time),
        "del_time": 0.0, # deletion happens in experiment_1.py
        "leakage": leakage_val,
        "utility": utility_val,
        "mask_size": len(chosen),
        "mask": set(chosen),
        "baseline_leakage": leakage_empty,
        "num_paths": num_chains,
        "memory_overhead_bytes": memory_overhead_bytes,
        "num_instantiated_cells": int(T["num_instantiated_cells"]),
    }

if __name__ == "__main__":
    print(two_phase_deletion_main("tax", key=1, target_cell="marital_status"))