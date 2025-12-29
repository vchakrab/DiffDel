#!/usr/bin/env python3
"""
Benchmark offline template generation time + memory for 6 datasets.

What it does:
- For each dataset:
  - repeats N_ITERS times:
    - creates a fresh temp save_dir
    - calls build_template_two_phase(dataset, target_attr, save_dir=..., epsilon=50, lam=0.5)
    - measures build time + memory
    - deletes the temp save_dir (so next run is cold)
    - measures delete time + memory
- Appends results to a CSV.

How to use:
1) Put this file somewhere in your repo (e.g., scripts/bench_offline_template.py)
2) Edit IMPORT below so build_template_two_phase is imported correctly.
3) Run: python bench_offline_template.py
"""

import csv
import gc
import os
import shutil
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime

# -----------------------------
# USER SETTINGS (edit these)
# -----------------------------

# 1) IMPORT: change this to your actual module that defines build_template_two_phase
# Example:
#   from del2ph import build_template_two_phase
#   from mypackage.two_phase import build_template_two_phase
try:
    from two_phase_deletion import build_template_two_phase  # <-- CHANGE ME
except Exception as e:
    print("[FATAL] Failed to import build_template_two_phase. Edit the IMPORT section.")
    print("Error:", repr(e))
    sys.exit(1)

# 2) Your 6 datasets (edit if needed)
DATASETS = ["airport", "hospital", "ncvoter", "onlineretail", "adult", "tax"]

# 3) Target attribute per dataset (edit to match your codebase)
#    (You likely already have TARGET_ATTR somewhere; you can import it instead.)
TARGET_ATTR = {
    "airport": "latitude_deg",
    "hospital": "zip",
    "ncvoter": "zip",
    "onlineretail": "CustomerID",
    "adult": "education",
    "tax": "zipcode",
}

# 4) DP settings used in offline template build
EPSILON = 50.0
LAM = 0.5

# 5) Repetitions per dataset
N_ITERS = 100

# 6) Where to write results
DEFAULT_OUT_CSV = "offline_template_bench.csv"

# 7) Base directory where temp template dirs will be created
#    (kept stable so you can inspect; each iter creates and then deletes a subdir)
BASE_BENCH_DIR = "bench_offline_templates_tmp"

# -----------------------------
# Memory helpers
# -----------------------------

def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None

_PSUTIL = _try_import_psutil()

def rss_bytes() -> int:
    """
    Current process RSS in bytes (best-effort).
    - If psutil is available: exact RSS
    - Else: returns -1
    """
    if _PSUTIL is None:
        return -1
    p = _PSUTIL.Process(os.getpid())
    return int(p.memory_info().rss)

def fmt_bytes(n: int) -> str:
    if n is None or n < 0:
        return "NA"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}PB"

# -----------------------------
# Filesystem helpers
# -----------------------------

def rm_tree(path: str) -> None:
    """Delete a directory tree or file if it exists."""
    if not path or not os.path.exists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=False)
    else:
        os.remove(path)

# -----------------------------
# Benchmark
# -----------------------------

CSV_FIELDS = [
    "timestamp",
    "dataset",
    "target_attribute",
    "iter",

    "epsilon",
    "lam",

    "build_time_s",
    "delete_time_s",

    # RSS (process resident memory) before/after each phase (if psutil installed; else -1)
    "rss_before_build_bytes",
    "rss_after_build_bytes",
    "rss_delta_build_bytes",

    "rss_before_delete_bytes",
    "rss_after_delete_bytes",
    "rss_delta_delete_bytes",

    # tracemalloc (python allocs) peak during build/delete
    "py_peak_build_bytes",
    "py_peak_delete_bytes",

    # size of save_dir before deletion (bytes, best-effort)
    "save_dir_size_bytes",
]

def dir_size_bytes(root: str) -> int:
    """Best-effort directory size in bytes (walk)."""
    if not os.path.exists(root):
        return 0
    total = 0
    for base, _, files in os.walk(root):
        for fn in files:
            fp = os.path.join(base, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def append_row(out_csv: str, row: dict) -> None:
    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})

def bench_one(out_csv: str) -> None:
    os.makedirs(BASE_BENCH_DIR, exist_ok=True)

    for ds in DATASETS:
        attr = TARGET_ATTR.get(ds)
        if not attr:
            print(f"[WARN] No TARGET_ATTR for dataset='{ds}'. Skipping.")
            continue

        print(f"\n=== Dataset={ds} attr={attr} ===")

        for it in range(1, N_ITERS + 1):
            # Encourage a colder run (best-effort)
            gc.collect()

            # Fresh unique save_dir for this run
            save_dir = tempfile.mkdtemp(prefix=f"{ds}_{attr}_", dir=BASE_BENCH_DIR)

            # ----------------
            # BUILD PHASE
            # ----------------
            rss0 = rss_bytes()
            tracemalloc.start()
            t0 = time.perf_counter()
            try:
                _ = build_template_two_phase(
                    ds,
                    attr,
                    save_dir=save_dir,
                    epsilon=float(EPSILON),
                    lam=float(LAM),
                )
            finally:
                build_time = time.perf_counter() - t0
                _, peak_build = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            rss1 = rss_bytes()

            # Size of what we created (so you can sanity-check)
            sd_size = dir_size_bytes(save_dir)

            # ----------------
            # DELETE PHASE
            # ----------------
            gc.collect()

            rss2 = rss_bytes()
            tracemalloc.start()
            t1 = time.perf_counter()
            try:
                rm_tree(save_dir)
            finally:
                del_time = time.perf_counter() - t1
                _, peak_del = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            rss3 = rss_bytes()

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset": ds,
                "target_attribute": attr,
                "iter": it,
                "epsilon": EPSILON,
                "lam": LAM,

                "build_time_s": f"{build_time:.6f}",
                "delete_time_s": f"{del_time:.6f}",

                "rss_before_build_bytes": rss0,
                "rss_after_build_bytes": rss1,
                "rss_delta_build_bytes": (rss1 - rss0) if (rss0 >= 0 and rss1 >= 0) else -1,

                "rss_before_delete_bytes": rss2,
                "rss_after_delete_bytes": rss3,
                "rss_delta_delete_bytes": (rss3 - rss2) if (rss2 >= 0 and rss3 >= 0) else -1,

                "py_peak_build_bytes": int(peak_build),
                "py_peak_delete_bytes": int(peak_del),

                "save_dir_size_bytes": int(sd_size),
            }
            append_row(out_csv, row)

            # progress line
            msg = (
                f"[{ds} {it:03d}/{N_ITERS}] "
                f"build={build_time:.4f}s (py_peak={fmt_bytes(peak_build)}, rssΔ={fmt_bytes(row['rss_delta_build_bytes'])}) "
                f"del={del_time:.4f}s (py_peak={fmt_bytes(peak_del)}, rssΔ={fmt_bytes(row['rss_delta_delete_bytes'])}) "
                f"dir={fmt_bytes(sd_size)}"
            )
            print(msg)

def main():
    out_csv = input(f"Output CSV filename (blank => {DEFAULT_OUT_CSV}): ").strip()
    if not out_csv:
        out_csv = DEFAULT_OUT_CSV

    print("\nNotes:")
    print("- RSS memory requires psutil. If you see -1, run: pip install psutil")
    print(f"- Writing results to: {out_csv}")
    print(f"- Temp dirs under: {BASE_BENCH_DIR} (created+deleted every iter)\n")

    bench_one(out_csv)
    print("\nDone.")

if __name__ == "__main__":
    main()
