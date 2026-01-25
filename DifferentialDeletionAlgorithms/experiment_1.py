#!/usr/bin/env python3
"""
run_standardized_experiments.py

Runs:
- delmin  (baseline_deletion_3.baseline_deletion_3)
- delexp  (exponential_deletion.exponential_deletion_main)
- delgum  (greedy_gumbel.gumbel_deletion_main)

Fixes / guarantees:
- paths_blocked is an ESTIMATE (NO path construction).
- paths_blocked/memory computations are NOT counted in init/model/del times.
- update-to-NULL is measured separately as update_time + num_cells_updated.
- standardized CSV schema across all methods.
- skips flights and tax datasets (per your request).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple, Union, List

import mysql.connector

import config

try:
    # preferred (repo layout)
    from DifferentialDeletionAlgorithms import (
        baseline_deletion_3,
        greedy_gumbel,
        exponential_deletion,
        two_phase_deletion,
    )
except Exception:
    # fallback (local files / flat layout)
    import baseline_deletion_3  # type: ignore
    import greedy_gumbel  # type: ignore
    import exponential_deletion  # type: ignore
    import two_phase_deletion  # type: ignore


# ----------------------------
# Config (NO flights / tax)
# ----------------------------

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]

ORIGINAL_TABLE_NAMES = {
    "airport": "airports",
    "hospital": "hospital_data",
    "adult": "adult_data",
    "flight" : "flight_data",
    "tax" : "tax_data",
}
K_SIZE = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}
TARGET_ATTR = {
    "airport": "home_link",
    "hospital": "ProviderNumber",
    "tax": "city",
    "adult": "education",
    "flight": "FlightDate"
}

ITERS = 100


# ----------------------------
# Small utilities
# ----------------------------

def normalize_dataset_name(ds: str) -> str:
    if ds.lower() in ("onlineretail", "online_retail", "online-retail"):
        return "Onlineretail"
    return ds


def get_db_config_robust(dataset: str) -> Dict[str, Any]:
    """
    config.get_database_config sometimes uses 'database' vs 'database_name'.
    Normalize it so mysql.connector.connect works.
    """
    cfg = config.get_database_config(dataset)
    if "database" not in cfg and "database_name" in cfg:
        cfg["database"] = cfg["database_name"]
    return cfg


def get_random_key(dataset: str) -> Optional[int]:
    dataset = normalize_dataset_name(dataset)
    db_details = get_db_config_robust(dataset)

    conn = mysql.connector.connect(
        host=db_details.get("host", "localhost"),
        user=db_details.get("user", "root"),
        password=db_details.get("password", ""),
        database=db_details.get("database"),
        ssl_disabled=db_details.get("ssl_disabled", True),
    )
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"SELECT ID FROM {dataset}_copy_data ORDER BY RAND() LIMIT 1;"
        )
        row = cursor.fetchone()
        return int(row[0]) if row else None
    finally:
        cursor.close()
        conn.close()


def setup_database_copies():
    """
    Create:
      - <dataset>_copy_data LIKE <original_table>
      - <dataset>_copy_data_insertiontime LIKE <original_table>_insertiontime (if exists)
    """
    print("Setting up database copies...")
    for dataset in DATASETS:
        dataset = normalize_dataset_name(dataset)
        try:
            db_config = get_db_config_robust(dataset)
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            original_table = ORIGINAL_TABLE_NAMES[dataset]
            copied_table = f"{dataset}_copy_data"

            print(f"  - Creating copy for '{dataset}': {copied_table}")
            cursor.execute(f"DROP TABLE IF EXISTS {copied_table};")
            cursor.execute(f"CREATE TABLE {copied_table} LIKE {original_table};")
            cursor.execute(f"INSERT INTO {copied_table} SELECT * FROM {original_table};")

            # Copy insertiontime table only if it exists
            original_time_table = f"{original_table}_insertiontime"
            copied_time_table = f"{copied_table}_insertiontime"

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
                """,
                (original_time_table,),
            )
            exists = int(cursor.fetchone()[0]) == 1
            if exists:
                cursor.execute(f"DROP TABLE IF EXISTS {copied_time_table};")
                cursor.execute(f"CREATE TABLE {copied_time_table} LIKE {original_time_table};")
                cursor.execute(f"INSERT INTO {copied_time_table} SELECT * FROM {original_time_table};")

            conn.commit()
            cursor.close()
            conn.close()

        except mysql.connector.Error as err:
            print(f"    ERROR for dataset {dataset}: {err}")
    print("Setup complete.\n")


def cleanup_database_copies():
    print("Cleaning up database copies...")
    for dataset in DATASETS:
        dataset = normalize_dataset_name(dataset)
        try:
            db_config = get_db_config_robust(dataset)
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            copied_table = f"{dataset}_copy_data"
            copied_time_table = f"{copied_table}_insertiontime"

            print(f"  - Dropping copy for '{dataset}': {copied_table}")
            cursor.execute(f"DROP TABLE IF EXISTS {copied_table};")
            cursor.execute(f"DROP TABLE IF EXISTS {copied_time_table};")

            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"    ERROR for dataset {dataset}: {err}")
    print("Cleanup complete.\n")


# ----------------------------
# Update-to-NULL (measured consistently)
# ----------------------------

MaskItem = Union[
    Tuple[int, str],        # (row_id, col_name)
    Tuple[str, int],        # (col_name, row_id)
    Dict[str, Any],         # {"id": <int>, "attr": <str>} etc.
    str                     # parseable strings; best-effort
]


def _parse_mask_item(item: MaskItem) -> Optional[Tuple[int, str]]:
    try:
        if isinstance(item, tuple) and len(item) == 2:
            a, b = item
            if isinstance(a, int) and isinstance(b, str):
                return (a, b)
            if isinstance(a, str) and isinstance(b, int):
                return (b, a)

        if isinstance(item, dict):
            rid = item.get("id", item.get("row", item.get("row_id")))
            col = item.get("attr", item.get("col", item.get("column")))
            if isinstance(rid, int) and isinstance(col, str):
                return (rid, col)

        if isinstance(item, str):
            s = item.strip()
            # "123:Attr"
            if ":" in s:
                left, right = s.split(":", 1)
                left, right = left.strip(), right.strip()
                if left.isdigit() and right:
                    return (int(left), right)
            # "Attr@123"
            if "@" in s:
                left, right = s.split("@", 1)
                left, right = left.strip(), right.strip()
                if right.isdigit() and left:
                    return (int(right), left)
    except Exception:
        return None

    return None


def update_mask_to_null(dataset: str, key: int, mask: Any) -> Tuple[float, int]:
    """
    Interprets mask as:
      - set of column names (strings)  -> updates those columns to NULL on row ID=key
      - OR iterable of (row_id, col) items -> updates per item
    Returns: (update_time_seconds, num_cells_updated)
    """
    dataset = normalize_dataset_name(dataset)
    t0 = time.time()
    updated = 0


    db_config = get_db_config_robust(dataset)
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        # Case A: mask is a set/list of column names
        if isinstance(mask, (set, list, tuple)) and all(isinstance(x, str) for x in mask):
            cols: List[str] = sorted(set(mask))
            cols = [c for c in cols if c.replace("_", "").isalnum()]
            if cols:
                set_clause = ", ".join([f"`{c}` = NULL" for c in cols])
                sql = f"UPDATE `{dataset}_copy_data` SET {set_clause} WHERE `ID` = %s"
                cursor.execute(sql, (key,))
                updated += int(cursor.rowcount or 0)

        else:
            # Case B: iterable of items possibly including row_id
            try:
                iterator = iter(mask)
            except TypeError:
                iterator = iter([])

            for item in iterator:
                parsed = _parse_mask_item(item)
                if not parsed:
                    continue
                rid, col = parsed
                if not col.replace("_", "").isalnum():
                    continue
                cursor.execute(
                    f"UPDATE `{dataset}_copy_data` SET `{col}` = NULL WHERE `ID` = %s",
                    (rid,),
                )
                updated += int(cursor.rowcount or 0)

        conn.commit()

    except Exception:
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

    return (float(time.time() - t0), int(updated))


# ----------------------------
# Metrics: paths_blocked estimate (NO enumeration)
# ----------------------------

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def estimate_paths_blocked(num_paths: int, leakage: Optional[float]) -> int:
    """
    No path construction.
    leakage in [0,1]:
      active ~= round(leakage * num_paths)
      blocked ~= num_paths - active
    """
    if leakage is None:
        return -1
    if num_paths is None or num_paths < 0:
        return -1
    L = max(0.0, min(1.0, float(leakage)))
    active = int(round(L * num_paths))
    active = clamp_int(active, 0, num_paths)
    return int(num_paths - active)


# ----------------------------
# CSV
# ----------------------------

def write_csv_header(f):
    f.write(
        "method,dataset,target_attribute,total_time,init_time,model_time,del_time,"
        "leakage,baseline_leakage_empty_mask,utility,"
        "total_paths,mask_size,"
        "model_size,num_instantiated_cells"
    )


def write_csv_row(f, row: Dict[str, Any]):
    def fmt(x):
        if x is None:
            return ""
        return str(x)

    f.write(
        ",".join([
            fmt(row.get("method")),
            fmt(row.get("dataset")),
            fmt(row.get("target_attribute")),
            fmt(row.get("total_time")),
            fmt(row.get("init_time")),
            fmt(row.get("model_time")),
            fmt(row.get("del_time")),
            fmt(row.get("leakage")),
            fmt(row.get("baseline_leakage_empty_mask")),
            fmt(row.get("utility")),
            fmt(row.get("total_paths")),
            fmt(row.get("mask_size")),
            fmt(row.get("memory_overhead_bytes")),
            fmt(row.get("num_instantiated_cells")),
        ]) + "\n"
    )


def standardize_row(
    *,
    method: str,
    dataset: str,
    attr: str,
    raw: Dict[str, Any],
    update_time: float,
) -> Dict[str, Any]:
    init_time = raw.get("init_time")
    model_time = float(raw.get("model_time"))
    del_time = float(raw.get("del_time"))
    delete_db_time = float(update_time)
    total_time = init_time + model_time + del_time
    total_time_minus_init = total_time - init_time #useful only for 2ph

    return {
        "method": method,
        "dataset": dataset,
        "target_attribute": attr,
        "total_time": total_time,
        "init_time": init_time,
        "model_time": model_time,
        "del_time": del_time,
        "leakage": raw.get("leakage"),
        "baseline_leakage_empty_mask": raw.get("baseline_leakage"),
        "utility": raw.get("utility"),
        "total_paths": int(raw.get("num_paths", -1) or -1),
        "mask_size": int(raw.get("mask_size", 0) or 0),
        "memory_overhead_bytes": raw.get("memory_overhead_bytes"),
        "num_instantiated_cells": raw.get("num_instantiated_cells", None),
        # passthrough method params / diagnostics when present
    }


# ----------------------------
# Collectors
# ----------------------------

def run_delmin(out_csv: str):
    with open(out_csv, "w", newline="") as f:
        write_csv_header(f)

        for ds in DATASETS:
            ds = normalize_dataset_name(ds)
            attr = TARGET_ATTR[ds]
            print(f"[delmin] Dataset={ds}, attr={attr}")

            for i in range(ITERS):
                key = get_random_key(ds)
                if key is None:
                    continue
                try:
                    raw = baseline_deletion_3.baseline_deletion_3(
                        target=attr,
                        key=key,
                        dataset=ds,
                        threshold=0.0,
                    )

                    # update-to-null measured separately
                    upd_t, upd_cnt = update_mask_to_null(ds, key, raw["mask"])
                    raw["del_time"] = upd_t
                    # since leakage=0 => all blocked

                    # use baseline-provided memory (MEDIUM)
                    memory_overhead = int(raw["memory_overhead_bytes"])

                    row = standardize_row(
                        method="delmin",
                        dataset=ds,
                        attr=attr,
                        raw=raw,
                        update_time=upd_t,
                    )
                    write_csv_row(f, row)

                except Exception as e:
                    print(f"  [delmin] iter {i+1} error: {e}")
                    continue


def run_delexp(
    out_csv: str,
    verbose: bool,
    lam: float = 0.1,
    epsilon: float = 0.1,
    which_ablation: str = None,
    *,
    mask_method: Optional[str] = None,
    leakage_method: str = "greedy_disjoint",
    method_name: str = "delexp",
):
    """
    which_ablation:
      None  -> full standardized CSV
      "l"   -> lambda ablation
      "e"   -> epsilon ablation
    """
    to_write = None
    if which_ablation is None:
        pass
    elif which_ablation == "l":
        to_write = "lambda"
    elif which_ablation == "e":
        to_write = "epsilon"
    else:
        raise ValueError("which_ablation must be None, 'l', or 'e'")

    with open(out_csv, "a", newline="") as f:
        if verbose:
            write_csv_header(f)
        else:
            f.write(f"{to_write},leakage,utility,mask_size\n")

        for ds in DATASETS:
            ds = normalize_dataset_name(ds)
            attr = TARGET_ATTR[ds]
            print(f"[delexp] Dataset={ds}, attr={attr}")

            for i in range(ITERS):
                key = get_random_key(ds)
                if key is None:
                    continue

                try:
                    raw = exponential_deletion.exponential_deletion_main(
                        dataset=ds,
                        key=key,
                        target_cell=attr,
                        epsilon=epsilon,
                        lam=lam,
                        leakage_method=str(leakage_method),
                        mask_method=mask_method,
                    )
                    print(raw)
                    print(raw.get("leakage"))
                    # ensure params are present for CSV even if callee doesn't include them
                    raw.setdefault("epsilon", float(epsilon))
                    raw.setdefault("lam", float(lam))
                    raw.setdefault("rho", float(raw.get("rho", 0.9) or 0.9))
                    raw.setdefault("leakage_method", str(leakage_method))

                    mask_obj = raw.get("mask", set())

                    # update-to-null measured separately
                    upd_t, upd_cnt = update_mask_to_null(ds, key, (mask_obj.update({attr})))

                    # Prefer method-provided counts (these are actual inference chains)
                    num_paths = int(raw.get("num_paths", -1) or -1)
                    if "paths_blocked" in raw and raw.get("paths_blocked") is not None:
                        paths_blocked = int(raw.get("paths_blocked") or 0)
                    else:
                        leakage = raw.get("leakage", None)
                        paths_blocked = estimate_paths_blocked(num_paths, leakage)

                    # IMPORTANT: delexp supplies its own (largest) memory estimate
                    memory_overhead = int(raw.get("memory_overhead_bytes", 0) or 0)

                    row = standardize_row(
                        method=method_name,
                        dataset=ds,
                        attr=attr,
                        raw=raw,
                        update_time=upd_t,
                        num_cells_updated=upd_cnt,
                        paths_blocked=paths_blocked,
                        memory_overhead_bytes=memory_overhead,
                    )

                    if verbose:
                        write_csv_row(f, row)
                    else:
                        if to_write == "lambda":
                            f.write(
                                f"{lam},{row['leakage']},{row['utility']},{row['mask_size']}\n"
                            )
                        elif to_write == "epsilon":
                            f.write(
                                f"{epsilon},{row['leakage']},{row['utility']},{row['mask_size']}\n"
                            )

                except Exception as e:
                    print(f"  [delexp] iter {i+1} error: {e}")
                    continue


def run_delexp_canonical(
    out_csv: str,
    *,
    verbose: bool = True,
    lam: float = 0.75,
    epsilon: float = 25,
    leakage_method: str = "greedy_disjoint",
):
    """Convenience wrapper: DelExp with canonical mask space (mask_method='canonical')."""
    return run_delexp(
        out_csv,
        verbose=verbose,
        lam=lam,
        epsilon=epsilon,
        which_ablation=None,
        mask_method="canonical",
        leakage_method=leakage_method,
        method_name="delexp_canonical",
    )



def run_delgum(
    out_csv: str,
    verbose: bool = False,
    lam: float = .75,
    epsilon: float = 0.1,
    L0: float = 0.25,
    which_ablation=None,
    *,
    leakage_method: str = "greedy_disjoint",
    method_name: str = "delgum",
):
    """
    Ablation now supports ONLY:
      - which_ablation == "l"  -> ablate lambda
      - which_ablation == "e"  -> ablate epsilon
    """
    to_write = None
    if which_ablation is None:
        pass
    elif which_ablation == "l":
        to_write = "lambda"
    elif which_ablation == "e":
        to_write = "epsilon"
    elif which_ablation == "eo":
        to_write = "epslo"
    else:
        raise ValueError("which_ablation must be one of: None, 'l' (lambda), 'e' (epsilon)")

    with open(out_csv, "a", newline="") as f:
        if verbose:
            write_csv_header(f)
        else:
            if to_write != "epslo":
                f.write(f"dataset,{to_write},leakage,utility,mask_size\n")
            else:
                f.write(f"dataset,epsilon,L0,leakage,utility,mask_size\n")


        for ds in DATASETS:
            ds = normalize_dataset_name(ds)
            attr = TARGET_ATTR[ds]
            print(f"[delgum] Dataset={ds}, attr={attr}")

            for i in range(ITERS):
                key = get_random_key(ds)
                if key is None:
                    continue

                try:
                    # NOTE: greedy_gumbel.gumbel_deletion_main does NOT take `key`.
                    raw = greedy_gumbel.gumbel_deletion_main(
                        dataset=ds,
                        target_cell=attr,
                        epsilon=float(epsilon),
                        L0=L0,
                        lam=float(lam),
                        K=int(K_SIZE[ds]),
                        leakage_method=str(leakage_method),
                    )
                    if verbose:

                        mask_obj = raw.get("mask", set())
                        upd_t, upd_cnt = update_mask_to_null(ds, key, (mask_obj.update({attr})))
                        raw["del_time"] = upd_t
                        row = standardize_row(
                        method=method_name,
                        dataset=ds,
                        attr=attr,
                        raw=raw,
                        update_time=upd_t,

                        )
                        write_csv_row(f, row)
                    else:
                        if to_write == "lambda":
                            f.write(f"{ds},{lam},{raw['leakage']},{raw['utility']},{raw['mask_size']}\n")
                        elif to_write == "epsilon":
                            f.write(f"{ds},{epsilon},{raw['leakage']},{raw['utility']},{raw['mask_size']}\n")
                        else:
                            f.write(
                                f"{ds},{epsilon},{L0},{raw['leakage']},{raw['utility']},{raw['mask_size']}\n")

                except Exception as e:
                    print(f"  [delgum] iter {i+1} error: {e}")
                    continue

import os

def delete_2ph_template(dataset: str, attr: str, template_dir: str) -> None:
    pkl_path = os.path.join(template_dir, f"{dataset}_{attr}.pkl")
    try:
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            print(f"[del2ph] deleted template: {pkl_path}")
    except Exception as e:
        print(f"[del2ph] warning: could not delete template {pkl_path}: {e}")

def run_del2ph(
    out_csv: str,
    *,
    epsilon: float = .1,
    lambda_penalty: float = .75,
    L0: float = 0.25,
    mask_method: Optional[str] = None,
    leakage_method: str = "greedy_disjoint",
    template_dir: str = "templates",
    method_name: str = "del2ph",
    verbose: bool = False,
    which_ablation: Optional[str] = None,
):
    """
    2-phase exponential-style (NEW λ/ε):
      - offline template cached in templates_2ph/  (not timed per-iteration)
      - online sampling returns mask/leakage/utility fast
      - update-to-null measured here for fairness (same as others)

    Uses:
      epsilon = 50
      lam = 0.5
    """
    TEMPLATE_DIR = str(template_dir)
    to_write = None
    if which_ablation is None:
        pass
    elif which_ablation == "l":
        to_write = "lambda"
    elif which_ablation == "e":
        to_write = "epsilon"
    elif which_ablation == "eo":
        to_write = "epslo"
    else:
        raise ValueError("which_ablation must be one of: None, 'l' (lambda), 'e' (epsilon)")

    with open(out_csv, "a", newline = "") as f:
        if verbose:
            write_csv_header(f)
        else:
            if to_write != "epslo":
                f.write(f"dataset,{to_write},leakage,utility,mask_size\n")
            else:
                f.write(f"dataset,epsilon,L0,leakage,utility,mask_size\n")

        for ds in DATASETS:
            ds = normalize_dataset_name(ds)
            attr = TARGET_ATTR[ds]
            print(f"[del2ph] Dataset={ds}, attr={attr}")

            for i in range(ITERS):
                key = get_random_key(ds)
                if key is None:
                    continue

                try:
                    raw = two_phase_deletion.two_phase_deletion_main(
                        dataset=ds,
                        key=key,
                        target_cell=attr,
                        epsilon=float(epsilon),
                        lambda_penalty=float(lambda_penalty),
                        L0=L0,
                        leakage_method=str(leakage_method),
                        template_dir=TEMPLATE_DIR,
                        mask_method=mask_method,
                    )
                    if verbose:
                        mask_obj = raw.get("mask", set())
                        upd_t, upd_cnt = update_mask_to_null(ds, key, (mask_obj.update({attr})))
                        raw["del_time"] = upd_t
                        row = standardize_row(
                        method=method_name,
                        dataset=ds,
                        attr=attr,
                        raw=raw,
                        update_time = upd_t
                        )
                        write_csv_row(f, row)
                    else:
                        if to_write == "lambda":
                            f.write(f"{ds},{lambda_penalty},{raw['leakage']},{raw['utility']},{raw['mask_size']}\n")
                        elif to_write == "epsilon":
                            f.write(f"{ds},{epsilon},{raw['leakage']},{raw['utility']},{raw['mask_size']}\n")
                        else:
                            f.write(f"{ds},{epsilon},{L0},{raw['leakage']},{raw['utility']},{raw['mask_size']}\n")


                except Exception as e:
                    print(f"  [del2ph] iter {i+1} error: {e}")
                    continue
            delete_2ph_template(ds, attr, template_dir)



def run_del2ph_canonical(
    out_csv: str,
    *,
    epsilon: float = 10.0,
    lam: float = 0.5,
    L0: float = 0.25,
    leakage_method: str = "greedy_disjoint",
    template_dir: str = "templates_2ph_canonical",
):
    """Convenience wrapper: Del2Ph with canonical mask space (mask_method='canonical')."""
    return run_del2ph(
        out_csv,
        epsilon=epsilon,
        lam=lam,
        mask_method="canonical",
        leakage_method=leakage_method,
        template_dir=template_dir,
        method_name="del2ph_canonical",
    )

# ----------------------------
# Main
# ----------------------------

def main():
    #RUN MIN (BASELINE DELETION)
    # setup_database_copies()
    # run_delmin(f"min_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    # cleanup_database_copies()

    # #RUN GUMBEL
    # setup_database_copies()
    # run_delgum(f"gum_{time.strftime('%Y%m%d-%H%M%S')}.csv", leakage_method="greedy_disjoint", verbose=True)
    # cleanup_database_copies()
    # setup_database_copies()
    # run_del2ph(f"2ph_{time.strftime('%Y%m%d-%H%M%S')}.csv", leakage_method = "greedy_disjoint",
    #            verbose = True)
    # setup_database_copies()
    # for epsilon in [.1, 1, 10]:
    #     for l0 in [.1, .2, .25,  .3, .4, .5, .75]:
    #         run_delgum(f"data_jan15_elo/delgum_{epsilon}_{l0}", epsilon=epsilon, lam=0.75, L0=l0, leakage_method="greedy_disjoint", which_ablation = "eo")
    # cleanup_database_copies()
    """with epsilon = 1, Lambda = .25, and L0 = .25.

Then keeping other parameters fixed as above:
1. eblate with L0 = .1, .3, .6, .9
2. Eblate \lambda = 0, .25, .5, 1"""
    # setup_database_copies()
    # for l0 in [0.1, .3, .6, .9]:
    #     run_del2ph(f"data_ablation_jan22_l0_del2ph_{l0}_v2", epsilon = 1, lam = 0.75, L0 = l0,
    #         leakage_method = "greedy_disjoint", which_ablation = "eo")
    # cleanup_database_copies()
    import os
    from datetime import datetime

    # Create data/<YYYY-MM-DD>/ directory
    # TODAY = datetime.now().strftime("%Y-%m-%d")
    # BASE_DIR = os.path.join("data", TODAY)
    # os.makedirs(BASE_DIR, exist_ok = True)
    #
    # # ---- L0 ablation ----
    #
    # setup_database_copies()
    # for l0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    #     run_del2ph(
    #         f"{BASE_DIR}/l0_ablation_{l0}",
    #         epsilon = 1,
    #         lambda_penalty = 100,
    #         L0 = l0,
    #         leakage_method = "greedy_disjoint",
    #         which_ablation = "eo",
    #     )
    # cleanup_database_copies()
    #
    # # ---- lambda ablation ----
    # setup_database_copies()
    # for lam in [1, 10, 100, 1000]:
    #     run_del2ph(
    #         f"{BASE_DIR}/lambda_ablation_{lam}",
    #         epsilon = 1,
    #         lambda_penalty = lam,
    #         L0 = 0.75,
    #         leakage_method = "greedy_disjoint",
    #         which_ablation = "l",
    #     )
    # cleanup_database_copies()
    #
    # # ---- epsilon ablation ----
    # setup_database_copies()
    # for ep in [0.1, 1, 10, 100, 1000]:
    #     run_del2ph(
    #         f"{BASE_DIR}/epsilon_ablation_{ep}",
    #         epsilon = ep,
    #         lambda_penalty = 100,
    #         L0 = 0.3,
    #         leakage_method = "greedy_disjoint",
    #         which_ablation = "e",
    #     )
    # cleanup_database_copies()

    TODAY = datetime.now().strftime("%Y-%m-%d")
    BASE_DIR = os.path.join("data", TODAY, "gum_ablations")
    os.makedirs(BASE_DIR, exist_ok = True)

    # ---- L0 ablation ----

    setup_database_copies()
    for l0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        run_delgum(
            f"{BASE_DIR}/l0_ablation_{l0}",
            epsilon = 1,
            lam = 100,
            L0 = l0,
            leakage_method = "greedy_disjoint",
            which_ablation = "eo",
        )
    cleanup_database_copies()

    # ---- lambda ablation ----
    setup_database_copies()
    for lam in [1, 10, 100, 1000]:
        run_delgum(
            f"{BASE_DIR}/lambda_ablation_{lam}",
            epsilon = 1,
            lam = lam,
            L0 = 0.75,
            leakage_method = "greedy_disjoint",
            which_ablation = "l",
        )
    cleanup_database_copies()

    # ---- epsilon ablation ----
    setup_database_copies()
    for ep in [0.1, 1, 10, 100, 1000]:
        run_delgum(
            f"{BASE_DIR}/epsilon_ablation_{ep}",
            epsilon = ep,
            lam = 100,
            L0 = 0.3,
            leakage_method = "greedy_disjoint",
            which_ablation = "e",
        )
    cleanup_database_copies()

    # setup_database_copies()
    # for lam in [0.1]:
    #     run_del2ph(f"data_ablation_jan21_lam_del2ph_{lam}", epsilon = 1, lam = lam, L0 = .25,
    #                leakage_method = "greedy_disjoint", which_ablation = "l")
    # cleanup_database_copies()
    # setup_database_copies()
    # for i in range(0, 11):
    #     run_delgum(f"data3/delgum_{i}.csv",
    #                      epsilon = 0.1, lam = i/10, leakage_method = "greedy_disjoint", which_ablation = "l")
    # cleanup_database_copies()
    # setup_database_copies()
    # for i in range(0, 11):
    #     run_del2ph(f"data3/del2ph_{i}.csv", epsilon =0.1, lam =i/10, leakage_method = "greedy_disjoint", which_ablation = "l")
    # cleanup_database_copies()
    # values = [0.05, 0.1, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1, 2, 8, 16, 32]
    # setup_database_copies()
    # for i in values:
    #     run_delgum(f"data3/edelgum_{i}.csv",
    #                epsilon = i, lam = 0.75, leakage_method = "greedy_disjoint",
    #                which_ablation = "e")
    # cleanup_database_copies()
    # setup_database_copies()
    # for i in values:
    #     run_del2ph(f"data3/edel2ph_{i}.csv",
    #                epsilon = i, lam = 0.75, leakage_method = "greedy_disjoint",
    #                which_ablation = "e")
    # cleanup_database_copies()

if __name__ == "__main__":
    main()



# cleanup_database_copies()
    # setup_database_copies()
    # for i in range(5, 100, 5):
    #     lam = i/100
    #     run_delgum("ablation_delgum_lambda_v1.csv", lam=lam, which_ablation="l", epsilon=50)
    # cleanup_database_copies()
    # setup_database_copies()
    # for i in range(5, 100, 5):
    #     lam = i / 100
    #     run_delexp("ablation_delexp_lambda_v1.csv", lam = lam, which_ablation = "l", verbose = False, epsilon=50)
    # cleanup_database_copies()
    # #
    # # #exp 2
    # setup_database_copies()
    # VALUES = [0.1, 0.51, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]
    # for i in VALUES:
    #     run_delgum("ablation_delgum_epsilon_v1.csv", lam = 0.5, which_ablation = "e", epsilon = i)
    # cleanup_database_copies()
    # setup_database_copies()
    # for i in VALUES:
    #     run_delexp("ablation_delexp_epsilon_v1.csv", lam = 0.5, which_ablation = "e", verbose = False, epsilon = i)
    # cleanup_database_copies()

    # setup_database_copies()
    # for i in [1, 2, 5, 10, 15, 20]:
    #     run_delgum("ablation_delexp_epsilon.csv", lam = 2 / 3, which_ablation = "k",
    #                K = i)
    # cleanup_database_copies()
    # print("\nDone.")
    # setup_database_copies()
    # for i in range(1, 301):
    #     run_delgum("ablation_delgum_epsilon.csv", epsilon = i, which_ablation = "e", verbose = False)
    # cleanup_database_copies()
    # setup_database_copies()
    # for i in range(1, 301):
    #     run_delexp("ablation_delexp_epsilon.csv", epsilon = i, which_ablation = "e", verbose = False)
    # cleanup_database_copies()

    # setup_database_copies()
    # for i in range(1, 301):
    #     run_delgum("ablation_delgum_alpha.csv", alpha = i, which_ablation = "a", verbose = False)
    # cleanup_database_copies()
    # setup_database_copies()
    # for i in range(1, 301):
    #     run_delexp("ablation_delexp_alpha.csv", alpha= i, which_ablation = "a", verbose = False)
    # cleanup_database_copies()
    #
    # setup_database_copies()
    # for i in range(1, 301):
    #     run_delgum("ablation_delgum_beta.csv", beta = i/2, which_ablation = "b", verbose = False)
    # cleanup_database_copies()
    # setup_database_copies()
    # for i in range(1, 301):
    #     run_delexp("ablation_delexp_beta.csv", beta = i/2, which_ablation = "b", verbose = False)
    # cleanup_database_copies()

    # print("=" * 60)
    # print("Standardized Deletion Experiments (delmin/delexp/delgum)")
    # print("=" * 60)
    #
    # setup_database_copies()
    # run_delmin("delmin_data_standarized_f3.csv")
    # cleanup_database_copies()
    # #
    # setup_database_copies()
    # run_delexp("delexp_data_standardized_non_canonical_or_leakage.csv", verbose=True)
    # cleanup_database_copies()
    # #
    # setup_database_copies()
    # run_delgum("delgum_data_standardized_vFinal.csv", verbose=True)
    # cleanup_database_copies()
    # #
    # setup_database_copies()
    # run_del2ph("del2ph_data_standardized_v2.csv")
    # cleanup_database_copies()
    #

    #ablation studies
    #exp 1
