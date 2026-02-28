#!/usr/bin/env python3
"""
make_main_data_copy.py

Creates a new directory:

    main_data/

containing:
    - exp/      (epsilon_m = 0.1, L0 = 0.1 only)
    - gumbel/   (epsilon_m = 0.1, L0 = 0.1 only)
    - min/      (all runs)

Must be placed in the same directory as:
    exp/
    gumbel/
    min/
"""

import os
import shutil
import pandas as pd

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]

EPSILON_M = 0.1
L0 = 0.1

BASE_DIR = os.getcwd()
OUT_DIR = os.path.join(BASE_DIR, "main_data")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def process_exp_or_gumbel(method):
    for d in DATASETS:
        src_file = os.path.join(BASE_DIR, method, d, "full_data.csv")

        if not os.path.exists(src_file):
            print(f"Skipping missing file: {src_file}")
            continue

        df = pd.read_csv(src_file)

        # Filter for epsilon_m = 0.1 and L0 = 0.1
        df_filtered = df[
            (df["epsilon_m"] == EPSILON_M) &
            (df["L0"] == L0)
        ]

        out_dir = os.path.join(OUT_DIR, method, d)
        ensure_dir(out_dir)

        out_file = os.path.join(out_dir, "full_data.csv")
        df_filtered.to_csv(out_file, index=False)

        print(f"Wrote filtered {method}/{d} → {out_file}")


def process_min():
    for d in DATASETS:
        src_file = os.path.join(BASE_DIR, "min", d, "final_data.csv")

        if not os.path.exists(src_file):
            print(f"Skipping missing file: {src_file}")
            continue

        out_dir = os.path.join(OUT_DIR, "min", d)
        ensure_dir(out_dir)

        out_file = os.path.join(out_dir, "final_data.csv")
        shutil.copy2(src_file, out_file)

        print(f"Copied min/{d} → {out_file}")


def main():
    ensure_dir(OUT_DIR)

    print("\nProcessing exp...")
    process_exp_or_gumbel("exp")

    print("\nProcessing gumbel...")
    process_exp_or_gumbel("gumbel")

    print("\nProcessing min...")
    process_min()

    print("\nDone. main_data/ created successfully.\n")


if __name__ == "__main__":
    main()