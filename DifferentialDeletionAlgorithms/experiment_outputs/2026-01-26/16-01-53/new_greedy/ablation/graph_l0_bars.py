#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

K_SIZE = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}

PANEL_ORDER = ["airport", "hospital", "adult", "flight", "tax"]

MASK_COLOR = "#1f77b4"
LEAK_COLOR = "#ff7f0e"


def parse_l0(name):
    m = re.search(r"del2ph_L0_([0-9]*\.?[0-9]+)\.csv$", name)
    return float(m.group(1)) if m else None


def load_data(l0_dir):
    files = sorted(Path(l0_dir).glob("del2ph_L0_*.csv"))

    pairs = []
    for f in files:
        l0 = parse_l0(f.name)
        if l0 is not None:
            pairs.append((l0, f))

    pairs.sort(key=lambda x: x[0])
    l0_vals = [p[0] for p in pairs]

    data = {ds: {"mask": [], "leak": []} for ds in K_SIZE}

    for l0, f in pairs:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        df["dataset"] = df["dataset"].str.strip().str.lower()

        for ds in K_SIZE:
            sub = df[df["dataset"] == ds]
            data[ds]["mask"].append(sub["mask_size"].mean())
            data[ds]["leak"].append(sub["leakage"].mean())

    return l0_vals, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l0_dir", required=True)
    parser.add_argument("--out_pdf", default="L0_side_by_side.pdf")
    args = parser.parse_args()

    l0_vals, data = load_data(args.l0_dir)

    fig, axes = plt.subplots(
        1, 5,
        figsize=(22, 4),   # wide landscape
        sharex=True
    )

    for ax, ds in zip(axes, PANEL_ORDER):

        base = K_SIZE[ds]
        mask_improve = 100 * (1 - (np.array(data[ds]["mask"]) / base))
        leaks = np.array(data[ds]["leak"])

        x = np.arange(len(l0_vals))
        width = 0.35

        bars1 = ax.bar(x - width/2, mask_improve, width,
                       color=MASK_COLOR, label="Mask Improvement (%)")

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, leaks, width,
                        color=LEAK_COLOR, label="Leakage")

        ax.set_title(ds.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(l0_vals)
        ax.set_xlabel("L0")

        if ds == "airport":
            ax.set_ylabel("Mask Improvement (%)")
        if ds == "tax":
            ax2.set_ylabel("Leakage")

        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(args.out_pdf)
    print("Wrote:", args.out_pdf)


if __name__ == "__main__":
    main()
