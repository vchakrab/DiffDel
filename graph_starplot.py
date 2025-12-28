#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# INPUT FILES (yours)
# =========================
INPUT_FILES = [
    "delmin_data_standardized_v2.csv",
    "delgum_data_standardized_v2.csv",
    "delexp_data_standardized_v2.csv",
]

OUTPUT_PDF = "mask_size_vs_instantiated_cells_scatter_by_method.pdf"
# =========================


REQUIRED_COLS = [
    "method",
    "dataset",
    "mask_size",
    "num_instantiated_cells",
]


def read_standardized_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df[REQUIRED_COLS].copy()

    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()

    df["mask_size"] = pd.to_numeric(df["mask_size"], errors="coerce")
    df["num_instantiated_cells"] = pd.to_numeric(df["num_instantiated_cells"], errors="coerce")

    df = df.dropna(subset=["mask_size", "num_instantiated_cells"])
    df = df[(df["mask_size"] >= 0) & (df["num_instantiated_cells"] >= 0)]

    return df


def main():
    frames = []
    for path in INPUT_FILES:
        print(f"Loading {path}")
        frames.append(read_standardized_csv(path))

    df = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=(11, 7))

    # One color per method (dataset ignored)
    methods = sorted(df["method"].unique())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(methods)}

    for method, g in df.groupby("method"):
        plt.scatter(
            g["mask_size"],
            g["num_instantiated_cells"],
            s=32,
            alpha=0.6,
            color=color_map[method],
            label=method,
        )

    plt.xlabel("mask_size")
    plt.ylabel("num_instantiated_cells")
    plt.title("Mask Size vs Instantiated Cells (scatter, colored by method)")
    plt.grid(True, alpha=0.25)

    plt.legend(title="Deletion method")
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
