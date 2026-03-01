#!/usr/bin/env python3
"""
run_standardized_experiments.py
"""

from __future__ import annotations

import os
import time
from typing import Callable, Any, Dict, Optional, Tuple

import mysql.connector
import config

import two_phase_deletion,marginal_em
import baseline_deletion_3


# ============================
# CONFIG
# ============================

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]

ORIGINAL_TABLE_NAMES = {
    "airport": "airports",
    "hospital": "hospital_data",
    "adult": "adult_data",
    "flight": "flight_data",
    "tax": "tax_data",
}

TARGET_ATTR = {
    "airport": "continent",
    "hospital": "ProviderNumber",
    "tax": "city",
    "adult": "education",
    "flight": "FlightDate",
}

ITERS = 100


# ============================
# UTILITIES
# ============================

def normalize_dataset_name(ds: str) -> str:
    return ds

def get_db_config_robust(dataset: str) -> Dict[str, Any]:
    db = config.get_database_config(dataset)

    return {
        "host": db["host"],
        "user": db["user"],
        "password": db["password"],
        "database": db["database"],
        "ssl_disabled": db.get("ssl_disabled", True),
    }
def get_random_key(dataset: str) -> Optional[int]:
    db = get_db_config_robust(dataset)
    conn = mysql.connector.connect(**db)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT ID FROM {dataset}_copy_data ORDER BY RAND() LIMIT 1;")
        row = cursor.fetchone()
        return int(row[0]) if row else None
    finally:
        cursor.close()
        conn.close()


# ============================
# UPDATE TO NULL
# ============================

def update_mask_to_null(dataset: str, key: int, mask: set) -> Tuple[float, int]:
    t0 = time.time()
    db = get_db_config_robust(dataset)

    conn = None
    cursor = None
    updated = 0

    try:
        conn = mysql.connector.connect(**db)
        cursor = conn.cursor()

        cols = [c for c in mask if c.replace("_", "").isalnum()]
        if cols:
            set_clause = ", ".join([f"`{c}` = NULL" for c in cols])
            sql = f"UPDATE `{dataset}_copy_data` SET {set_clause} WHERE `ID` = %s"
            cursor.execute(sql, (key,))
            updated = int(cursor.rowcount or 0)

        conn.commit()

    except Exception:
        if conn:
            conn.rollback()

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return time.time() - t0, updated


# ============================
# CSV
# ============================

def write_csv_header(f):
    f.write(
        "method,dataset,target_attribute,epsilon_m,lambda,L0,"
        "total_time,init_time,model_time,del_time,"
        "leakage,baseline_leakage_empty_mask,utility,"
        "total_paths,mask_size,"
        "memory_overhead_bytes,num_instantiated_cells\n"
    )


def write_csv_row(f, row: Dict[str, Any]):
    def fmt(x):
        return "" if x is None else str(x)

    f.write(",".join([
        fmt(row.get("method")),
        fmt(row.get("dataset")),
        fmt(row.get("target_attribute")),
        fmt(row.get("epsilon_m")),
        fmt(row.get("lambda")),
        fmt(row.get("L0")),
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
    ]) + "\n")


def standardize_row(
    *,
    method: str,
    dataset: str,
    attr: str,
    raw: Dict[str, Any],
    update_time: float,
    epsilon_m: float,
    lambda_val: float,
    L0: float,
) -> Dict[str, Any]:

    init_time = raw.get("init_time", 0)
    model_time = float(raw.get("model_time", 0))
    del_time = float(update_time)
    total_time = init_time + model_time + del_time

    return {
        "method": method,
        "dataset": dataset,
        "target_attribute": attr,
        "epsilon_m": epsilon_m,
        "lambda": lambda_val,
        "L0": L0,
        "total_time": total_time,
        "init_time": init_time,
        "model_time": model_time,
        "del_time": del_time,
        "leakage": raw.get("leakage"),
        "baseline_leakage_empty_mask": raw.get("baseline_leakage"),
        "utility": raw.get("utility"),
        "total_paths": raw.get("num_paths", -1),
        "mask_size": raw.get("mask_size", 0),
        "memory_overhead_bytes": raw.get("memory_overhead_bytes"),
        "num_instantiated_cells": raw.get("num_instantiated_cells"),
    }

def run_delmin(out_csv, ds):

    file_exists = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0

    with open(out_csv, "a", newline="") as f:
        if not file_exists:
            write_csv_header(f)

        attr = TARGET_ATTR[ds]

        for _ in range(ITERS):
            key = get_random_key(ds)
            if key is None:
                continue

            raw = baseline_deletion_3.baseline_deletion_3(
                dataset=ds,
                key=key,
                target=attr,
                threshold=0.0,
            )

            # delmin mask is just attributes (like others)
            upd_mask = set(raw.get("mask", set())) | {attr}
            upd_t, _ = update_mask_to_null(ds, key, upd_mask)

            row = standardize_row(
                method="delmin",
                dataset=ds,
                attr=attr,
                raw=raw,
                update_time=upd_t,
                epsilon_m = None,  # not used
                lambda_val = None,  # not used
                L0 = None,
            )

            write_csv_row(f, row)
# ============================
# EXPERIMENT RUNNERS
# ============================

def run_marginal_em(out_csv, ds, epsilon, lam, L0):

    file_exists = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0

    with open(out_csv, "a", newline="") as f:
        if not file_exists:
            write_csv_header(f)

        attr = TARGET_ATTR[ds]

        for _ in range(ITERS):
            key = get_random_key(ds)
            if key is None:
                continue

            raw = marginal_em.marginal_em_main(
                dataset=ds,
                target_cell=attr,
                epsilon=float(epsilon),
                lam=float(lam),
                L0=float(L0),
            )

            upd_mask = set(raw["mask"]) | {attr}
            upd_t, _ = update_mask_to_null(ds, key, upd_mask)

            row = standardize_row(
                method="marginal_em",
                dataset=ds,
                attr=attr,
                raw=raw,
                update_time=upd_t,
                epsilon_m=epsilon,
                lambda_val=lam,
                L0=L0,
            )

            write_csv_row(f, row)


def run_del2ph(out_csv, ds, epsilon, lam, L0):

    file_exists = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0

    with open(out_csv, "a", newline="") as f:
        if not file_exists:
            write_csv_header(f)

        attr = TARGET_ATTR[ds]

        for _ in range(ITERS):
            key = get_random_key(ds)
            if key is None:
                continue

            raw = two_phase_deletion.two_phase_deletion_main(
                dataset=ds,
                key=key,
                target_cell=attr,
                epsilon=float(epsilon),
                lambda_penalty=float(lam),
                L0=float(L0),
            )

            upd_mask = set(raw.get("mask", set())) | {attr}
            upd_t, _ = update_mask_to_null(ds, key, upd_mask)

            row = standardize_row(
                method="del2ph",
                dataset=ds,
                attr=attr,
                raw=raw,
                update_time=upd_t,
                epsilon_m=epsilon,
                lambda_val=lam,
                L0=L0,
            )

            write_csv_row(f, row)


# ============================
# DB COPY WRAPPER (SAFE)
# ============================

def setup_database_copies(dataset: str):

    dataset = normalize_dataset_name(dataset)
    conn = None
    cursor = None

    try:
        db_config = get_db_config_robust(dataset)
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        original_table = ORIGINAL_TABLE_NAMES[dataset]
        copied_table = f"{dataset}_copy_data"

        cursor.execute(f"DROP TABLE IF EXISTS `{copied_table}`;")
        cursor.execute(f"CREATE TABLE `{copied_table}` LIKE `{original_table}`;")
        cursor.execute(f"INSERT INTO `{copied_table}` SELECT * FROM `{original_table}`;")

        original_time_table = f"{original_table}_insertiontime"
        copied_time_table = f"{copied_table}_insertiontime"

        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = DATABASE()
              AND table_name = %s
        """, (original_time_table,))

        if int(cursor.fetchone()[0]) == 1:
            cursor.execute(f"DROP TABLE IF EXISTS `{copied_time_table}`;")
            cursor.execute(f"CREATE TABLE `{copied_time_table}` LIKE `{original_time_table}`;")
            cursor.execute(f"INSERT INTO `{copied_time_table}` SELECT * FROM `{original_time_table}`;")

        conn.commit()

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def cleanup_database_copies(dataset: str):

    dataset = normalize_dataset_name(dataset)
    conn = None
    cursor = None

    try:
        db_config = get_db_config_robust(dataset)
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        copied_table = f"{dataset}_copy_data"
        copied_time_table = f"{copied_table}_insertiontime"

        cursor.execute(f"DROP TABLE IF EXISTS `{copied_table}`;")
        cursor.execute(f"DROP TABLE IF EXISTS `{copied_time_table}`;")

        conn.commit()

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def with_db_copies(fn: Callable[[], None], dataset: str) -> None:
    setup_database_copies(dataset)
    try:
        fn()
    finally:
        cleanup_database_copies(dataset)


# ============================
# MAIN LOOP (UNCHANGED)
# ============================

def run_all_experiments():
    EM_VALUES = [0]#, 0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 10]
    L0_VALUES = [0.025]#, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    LAMBDA = 1000

    BASE_OUTPUT_DIR = "experiment_outputs"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    GUM_DIR = os.path.join(BASE_OUTPUT_DIR, "gumbel")
    os.makedirs(GUM_DIR, exist_ok=True)

    for epsilon_m in EM_VALUES:
        for L0 in L0_VALUES:
            for dataset in DATASETS:

                dataset_dir = os.path.join(GUM_DIR, dataset)
                os.makedirs(dataset_dir, exist_ok=True)

                file_path = os.path.join(dataset_dir, "full_data.csv")

                def run_once():
                    run_marginal_em(
                        out_csv=file_path,
                        ds=dataset,
                        epsilon=epsilon_m,
                        L0=L0,
                        lam=LAMBDA,
                    )

                with_db_copies(run_once, dataset=dataset)

    EXP_DIR = os.path.join(BASE_OUTPUT_DIR, "exp")
    os.makedirs(EXP_DIR, exist_ok=True)

    for epsilon_m in EM_VALUES:
        for L0 in L0_VALUES:
            for dataset in DATASETS:

                dataset_dir = os.path.join(EXP_DIR, dataset)
                os.makedirs(dataset_dir, exist_ok=True)

                file_path = os.path.join(dataset_dir, "full_data.csv")

                def run_once():
                    run_del2ph(
                        out_csv=file_path,
                        ds=dataset,
                        epsilon=epsilon_m,
                        L0=L0,
                        lam=LAMBDA,
                    )

                with_db_copies(run_once, dataset=dataset)
    # --------------------------
    # DELMIN (BASELINE)
    # --------------------------
    MIN_DIR = os.path.join(BASE_OUTPUT_DIR, "min")
    os.makedirs(MIN_DIR, exist_ok=True)

    for dataset in DATASETS:

        dataset_dir = os.path.join(MIN_DIR, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        file_path = os.path.join(dataset_dir, "full_data.csv")

        def run_once():
            run_delmin(
                out_csv=file_path,
                ds=dataset,
            )

        with_db_copies(run_once, dataset=dataset)

if __name__ == "__main__":
    run_all_experiments()