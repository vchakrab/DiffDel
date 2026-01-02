#!/usr/bin/env python3
"""
baseline_3_ablation.py

Ablation runner for Baseline-3 (ILP) + leakage model + tau filtering.

Fixes Python 3.13 dataclass import crash by ensuring the dynamically-loaded
module is registered in sys.modules BEFORE exec_module().

Also adds strong debug output for why H_actual becomes 0.

Edit the USER SETTINGS section.
"""

from __future__ import annotations

import os
import sys
import csv
import time
import math
import types
import traceback
import importlib.util
from typing import Any, Dict, List, Tuple, Set, Optional

# =========================
# USER SETTINGS
# =========================

# Path to your Baseline-3 integrated module (the file that defines the functions)
BASELINE3_PY = "baseline_deletion_3.py"

# Python module name to register under (avoid collisions)
BASELINE3_MODNAME = "baseline_deletion_3"

# Output CSV
OUT_CSV = "ablation_tau_baseline3.csv"

# Tau values to sweep
TAU_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Iterations per dataset (how many random keys / trials)
ITERS = 50

# Toggle extra debug
VERBOSE_DEBUG = True

# =========================
# Helper: robust float
# =========================
def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


# =========================
# FIX: dynamic module loader
# =========================
def load_module_from_path(path: str, modname: str) -> types.ModuleType:
    """
    Dynamically loads a module from a .py path.

    CRITICAL FIX (Python 3.13 + dataclasses):
      Insert the module into sys.modules BEFORE exec_module.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline module not found: {path}")

    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create spec for module at: {path}")

    mod = importlib.util.module_from_spec(spec)

    # >>> The important line that avoids your crash <<<
    sys.modules[modname] = mod

    spec.loader.exec_module(mod)  # type: ignore
    return mod


# =========================
# Tau filter
# =========================
def apply_tau_filter(
    rdrs: List[Any],
    weights: Dict[Any, Any],
    tau: float,
) -> Tuple[List[Any], Dict[Any, float]]:
    """
    Keep only hyperedges with weight <= tau.
    Returns filtered rdrs list and filtered numeric weights dict.
    """
    w_num: Dict[Any, float] = {k: safe_float(v) for k, v in (weights or {}).items()}

    keep_rdrs: List[Any] = []
    keep_w: Dict[Any, float] = {}

    for e in rdrs or []:
        w = w_num.get(e, float("nan"))
        if not math.isfinite(w):
            # weight missing/NaN => drop (and report later via debug)
            continue
        if w <= tau:
            keep_rdrs.append(e)
            keep_w[e] = w

    return keep_rdrs, keep_w


def weight_stats(weights: Dict[Any, float]) -> Tuple[int, float, float]:
    if not weights:
        return (0, float("nan"), float("nan"))
    vals = [v for v in weights.values() if math.isfinite(v)]
    if not vals:
        return (len(weights), float("nan"), float("nan"))
    return (len(vals), min(vals), max(vals))


# =========================
# CSV
# =========================
CSV_HEADER = [
    "method",
    "dataset",
    "tau",
    "iter",
    "key",
    "H_max",
    "H_actual",
    "zone_size",
    "leakage",
    "utility",
    "mask_size",
    "total_time_s",
    "ilp_time_s",
    "max_depth",
    "ilp_num_cells",
    "ilp_num_vars",
    "ilp_num_constrs",
    "activated_dependencies_count",
    "debug_note",
]

def init_csv(path: str) -> None:
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    if need_header:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)


def append_csv(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(c, "") for c in CSV_HEADER])


# =========================
# Debug helpers for H_actual == 0
# =========================
def try_edge_contains_target(edge: Any, target: str) -> bool:
    """
    Best-effort: check whether an edge structure contains target.
    Your edge may be a tuple/list/dict/custom object.
    """
    if edge is None:
        return False
    try:
        # common: edge is tuple/list of strings or has "head"/"body"
        if isinstance(edge, (tuple, list, set)):
            return target in edge or any(str(x) == target for x in edge)
        if isinstance(edge, dict):
            return target in edge.values() or target in edge.keys()
        # if object: check attrs
        for attr in ("head", "consequent", "to", "rhs", "y"):
            if hasattr(edge, attr) and str(getattr(edge, attr)) == target:
                return True
    except Exception:
        return False
    return False


def summarize_h_actual_issue(
    target_cell: str,
    rdrs: List[Any],
    weights: Dict[Any, float],
    H_max: float,
    H_actual: float,
    zone_size: int,
) -> str:
    if H_actual != 0:
        return ""

    n_edges = len(rdrs or [])
    n_w, wmin, wmax = weight_stats(weights)
    any_edge_mentions_target = any(try_edge_contains_target(e, target_cell) for e in (rdrs or []))

    reasons: List[str] = []
    reasons.append(f"H_actual==0 with n_edges={n_edges}, n_weights={n_w}, wmin={wmin}, wmax={wmax}, zone_size={zone_size}")
    if n_edges == 0:
        reasons.append("No edges after filtering => actual hypergraph empty (common if tau too small or weights missing).")
    if not any_edge_mentions_target:
        reasons.append("No edge appears to mention the target cell => target may be disconnected/unreachable.")
    if not math.isfinite(H_max) or H_max == 0:
        reasons.append("H_max is 0/NaN too => inference zone might be empty, or constructor returned degenerate value.")
    reasons.append("If construct_hypergraph_actual depends on instance-specific activation (DB/key) and you're not passing it, it may return 0 by design.")
    return " | ".join(reasons)


# =========================
# Main
# =========================
def main() -> None:
    init_csv(OUT_CSV)

    # Load your baseline module robustly (fixes the dataclass crash)
    mod = load_module_from_path(BASELINE3_PY, BASELINE3_MODNAME)
    print(f"[OK] Loaded Baseline-3 module from: {os.path.abspath(BASELINE3_PY)} as {BASELINE3_MODNAME}")

    # Required functions (as per your integrated baseline 3 setup)
    # NOTE: these names must exist in baseline_deletion_3.py
    dc_to_rdrs_and_weights_strict = getattr(mod, "dc_to_rdrs_and_weights_strict")
    construct_hypergraph_max = getattr(mod, "construct_hypergraph_max")
    construct_hypergraph_actual = getattr(mod, "construct_hypergraph_actual")
    compute_leakage_delexp = getattr(mod, "compute_leakage_delexp")
    compute_utility_new = getattr(mod, "compute_utility_new")
    estimate_memory_bytes_standard = getattr(mod, "estimate_memory_bytes_standard", None)
    ilp_approach_matching_java = getattr(mod, "ilp_approach_matching_java")

    # Config / dataset utilities must exist in your module too
    DATASETS = getattr(mod, "DATASETS")
    TARGET_ATTR = getattr(mod, "TARGET_ATTR")
    normalize_dataset_name = getattr(mod, "normalize_dataset_name")
    get_random_key = getattr(mod, "get_random_key")

    # Some pipelines expose initialization_phase; optional
    initialization_phase = getattr(mod, "initialization_phase", None)

    for tau in TAU_VALUES:
        print(f"\n==================== tau={tau} ====================")

        for ds_raw in DATASETS:
            ds = normalize_dataset_name(ds_raw)
            target_attr = TARGET_ATTR[ds]
            print(f"[baseline3] Dataset={ds} target={target_attr}")

            # Offline init if your module needs it (safe no-op if None)
            if initialization_phase is not None:
                try:
                    initialization_phase(ds)
                except Exception:
                    print("[WARN] initialization_phase failed; continuing.\n" + traceback.format_exc())

            for it in range(ITERS):
                key = get_random_key(ds)
                if key is None:
                    continue

                t0 = time.time()
                debug_note = ""

                try:
                    # Build (rdrs, weights) from denial constraints / template logic
                    rdrs, weights = dc_to_rdrs_and_weights_strict(ds, key)

                    # Tau filter
                    rdrs_f, weights_f = apply_tau_filter(rdrs, weights, tau)

                    # Compute hypergraph metrics used for leakage eval
                    target_cell = target_attr
                    H_max = construct_hypergraph_max(target_cell, rdrs_f, weights_f)
                    H_actual = construct_hypergraph_actual(target_cell, rdrs_f, weights_f)

                    # Zone size best-effort:
                    # If H_max is an object w/ vertices, use it. If it returns a float, we fallback.
                    zone_size = -1
                    try:
                        # Some implementations return an object, not a scalar:
                        if hasattr(H_max, "vertices"):
                            zone_size = len(set(H_max.vertices) - {target_cell})  # type: ignore
                        elif hasattr(mod, "last_vertices"):
                            # optional debugging hook if you store vertices somewhere
                            zone_size = len(set(getattr(mod, "last_vertices")) - {target_cell})
                        else:
                            zone_size = -1
                    except Exception:
                        zone_size = -1

                    # If your construct_* functions return floats (entropy), zone size can't be inferred here.
                    # We'll still compute leakage and report.
                    # Leakage function (your delexp-style)
                    leakage = compute_leakage_delexp(
                        target_cell=target_cell,
                        rdrs=rdrs_f,
                        weights=weights_f,
                        # If your signature differs, adjust here to match your baseline_deletion_3.py
                    )

                    # Run ILP (schema-only; you can change if you later want DB-backed)
                    ilp_start = time.time()
                    (
                        to_del_cells,
                        total_ilp_time,
                        max_depth,
                        ilp_num_cells,
                        ilp_num_vars,
                        ilp_num_constrs,
                        activated_dependencies_count,
                    ) = ilp_approach_matching_java(
                        dataset=ds,
                        key=key,
                        target_attr=target_attr,
                        cursor=None,
                        target_time=0,
                        # If your ILP function has extra args, keep them consistent with your module.
                    )
                    ilp_time = time.time() - ilp_start

                    mask_size = len(to_del_cells) if to_del_cells is not None else 0

                    utility = compute_utility_new(
                        to_del=to_del_cells,
                        dataset=ds,
                        target_attr=target_attr,
                        # If your signature differs, adjust.
                    )

                    if VERBOSE_DEBUG:
                        # Add a tight but very informative note if H_actual is 0
                        debug_note = summarize_h_actual_issue(
                            target_cell=target_cell,
                            rdrs=rdrs_f,
                            weights=weights_f,
                            H_max=safe_float(H_max, float("nan")),
                            H_actual=safe_float(H_actual, float("nan")),
                            zone_size=zone_size,
                        )

                        # And print quickly on console
                        if safe_float(H_actual, float("nan")) == 0.0:
                            n_w, wmin, wmax = weight_stats(weights_f)
                            print(
                                f"  [DBG] it={it} key={key} H_max={H_max} H_actual={H_actual} "
                                f"edges={len(rdrs_f)} weights={n_w} wmin={wmin} wmax={wmax} note={debug_note}"
                            )

                    total_time = time.time() - t0

                    append_csv(OUT_CSV, {
                        "method": "baseline3_ilp",
                        "dataset": ds,
                        "tau": tau,
                        "iter": it,
                        "key": key,
                        "H_max": H_max,
                        "H_actual": H_actual,
                        "zone_size": zone_size,
                        "leakage": leakage,
                        "utility": utility,
                        "mask_size": mask_size,
                        "total_time_s": total_time,
                        "ilp_time_s": ilp_time,
                        "max_depth": max_depth,
                        "ilp_num_cells": ilp_num_cells,
                        "ilp_num_vars": ilp_num_vars,
                        "ilp_num_constrs": ilp_num_constrs,
                        "activated_dependencies_count": activated_dependencies_count,
                        "debug_note": debug_note,
                    })

                except Exception:
                    err = traceback.format_exc()
                    print(f"[ERR] ds={ds} tau={tau} it={it} key={key}\n{err}")
                    append_csv(OUT_CSV, {
                        "method": "baseline3_ilp",
                        "dataset": ds,
                        "tau": tau,
                        "iter": it,
                        "key": key,
                        "debug_note": "EXCEPTION: " + err.replace("\n", " | "),
                    })

    print(f"\n[OK] Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
