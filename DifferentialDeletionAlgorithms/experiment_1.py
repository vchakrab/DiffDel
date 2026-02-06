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
import os
from datetime import datetime


#!/usr/bin/env python3

import os
from datetime import datetime
from typing import Callable

# You already have these somewhere in your repo:
# from your_module import setup_database_copies, cleanup_database_copies, run_delgum, run_del2ph


def with_db_copies(fn: Callable[[], None]) -> None:
    """Run fn() with fresh DB copies, always cleaning up afterward."""
    setup_database_copies()
    try:
        fn()
    finally:
        cleanup_database_copies()


def main() -> None:
    epsilons = [0.1, 0.2, 0.4, 0.8, 1, 2, 3, 4, 5, 10, 100]
    # l0s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # lams = [10000]

    RUN_DATE = datetime.now().strftime("%Y-%m-%d")
    RUN_TIMESTAMP = datetime.now().strftime("%H-%M-%S")
    BASE_OUTPUT_DIR = "experiment_outputs"
    RUN_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_DATE, RUN_TIMESTAMP)
    ABL_DIR = os.path.join(RUN_DIR, "ablation")
    os.makedirs(ABL_DIR, exist_ok=True)

    # -------------------------
    # delgum: epsilon ablation
    # -------------------------
    def delgum_eps() -> None:
        for epsilon in epsilons:
            out = os.path.join(ABL_DIR, f"delgum_epsilon_{epsilon}.csv")
            run_delgum(
                out,
                epsilon=epsilon,
                L0=0.3,
                lam=100,
                verbose=False,
                leakage_method="greedy_disjoint",
                which_ablation="e",
            )

    with_db_copies(delgum_eps)

    # ----------------------
    # delgum: lambda ablation
    # ----------------------
    # def delgum_lam() -> None:
    #     for lam in lams:
    #         out = os.path.join(ABL_DIR, f"delgum_lam_{lam}.csv")
    #         run_delgum(
    #             out,
    #             epsilon=0.1,
    #             L0=0.3,
    #             lam=lam,
    #             verbose=False,
    #             leakage_method="greedy_disjoint",
    #             which_ablation="l",
    #         )

    # with_db_copies(delgum_lam)

    # ------------------
    # delgum: L0 ablation
    # ------------------
    # def delgum_l0() -> None:
    #     for l0 in l0s:
    #         out = os.path.join(ABL_DIR, f"delgum_L0_{l0}.csv")
    #         run_delgum(
    #             out,
    #             epsilon=0.1,
    #             L0=l0,
    #             lam=100,
    #             verbose=False,
    #             leakage_method="greedy_disjoint",
    #             which_ablation="eo",
    #         )
    #
    # with_db_copies(delgum_l0)
    #
    # # -------------------------
    # # del2ph: epsilon ablation
    # # -------------------------
    def del2ph_eps() -> None:
        for epsilon in epsilons:
            out = os.path.join(ABL_DIR, f"del2ph_epsilon_{epsilon}.csv")
            run_del2ph(
                out,
                epsilon=epsilon,
                L0=0.3,
                lambda_penalty=100,
                verbose=False,
                leakage_method="greedy_disjoint",
                which_ablation="e",
            )

    with_db_copies(del2ph_eps)

    # ----------------------
    # del2ph: lambda ablation
    # ----------------------
    # def del2ph_lam() -> None:
    #     for lam in lams:
    #         out = os.path.join(ABL_DIR, f"del2ph_lam_{lam}.csv")
    #         run_del2ph(
    #             out,
    #             epsilon=0.1,
    #             L0=0.3,
    #             lambda_penalty=lam,
    #             verbose=False,
    #             leakage_method="greedy_disjoint",
    #             which_ablation="l",
    #         )
    #
    # with_db_copies(del2ph_lam)

    # ------------------
    # del2ph: L0 ablation
    # ------------------
    # def del2ph_l0() -> None:
    #     for l0 in l0s:
    #         out = os.path.join(ABL_DIR, f"del2ph_L0_{l0}.csv")
    #         run_del2ph(
    #             out,
    #             epsilon=0.1,
    #             L0=l0,
    #             lambda_penalty=100,
    #             verbose=False,
    #             leakage_method="greedy_disjoint",
    #             which_ablation="eo",
    #         )
    #
    # with_db_copies(del2ph_l0)

    print(f"Done. Wrote ablations to: {ABL_DIR}")


if __name__ == "__main__":
    main()

