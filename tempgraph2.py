#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT_FILE = "delexp_data_collect_epsilon.csv"
OUTPUT_PDF = "graph_x.pdf"

# Your file DOES NOT have dataset headers; it repeats a CSV header between dataset blocks.
# Each block belongs to datasets in this fixed repeating order:
DATASETS_IN_ORDER = ["airport", "hospital", "ncvoter", "Onlineretail", "adult"]

def parse_tempdata(path: str) -> pd.DataFrame:
    rows = []

    current_ds_idx = -1
    current_ds = None
    saw_any_header = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Header line marks a NEW dataset block
            if line.lower().startswith("epsilon,"):
                saw_any_header = True
                current_ds_idx = (current_ds_idx + 1) % len(DATASETS_IN_ORDER)
                current_ds = DATASETS_IN_ORDER[current_ds_idx]
                continue

            # Ignore data until we see the first header
            if current_ds is None:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue

            try:
                eps = float(parts[0])
                leakage = float(parts[1])
                utility = float(parts[2])
                mask = float(parts[3])
            except ValueError:
                continue

            rows.append((current_ds, eps, leakage, utility, mask))

    if not saw_any_header:
        raise ValueError("No 'epsilon,leakage,utility,mask_size' headers found.")
    if not rows:
        raise ValueError("No data parsed. Check file format.")

    return pd.DataFrame(rows, columns=["dataset", "epsilon", "leakage", "utility", "mask_size"])


def smooth_lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth y vs x using LOWESS on the ORIGINAL scale.
    Returns sorted (x_s, y_s).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if len(np.unique(x)) < 3:
        return x, y

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(y, x, frac=frac, it=1, return_sorted=True)
        return sm[:, 0], sm[:, 1]
    except Exception:
        s = pd.Series(y)
        w = max(3, (len(y) // 5) | 1)  # odd window
        y_med = s.rolling(window=w, center=True, min_periods=1).median()
        y_smooth = y_med.rolling(window=w, center=True, min_periods=1).mean()
        return x, y_smooth.to_numpy()


def main():
    df = parse_tempdata(INPUT_FILE)

    # Mean over runs per dataset/epsilon
    g = (
        df.groupby(["dataset", "epsilon"], as_index=False)
          .agg(
              leakage=("leakage", "mean"),
              utility=("utility", "mean"),
              mask_size=("mask_size", "mean"),
              n=("mask_size", "size"),
          )
          .sort_values(["dataset", "epsilon"])
          .reset_index(drop=True)
    )

    # Keep legend/order exactly as specified (only include datasets that appear)
    datasets = [d for d in DATASETS_IN_ORDER if d in set(g["dataset"])]

    with PdfPages(OUTPUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

        specs = [
            ("leakage", "Leakage"),
            ("utility", "Utility"),
            ("mask_size", "Mask Size"),
        ]

        frac = 0.40

        for ax, (col, ylabel) in zip(axes, specs):
            for ds in datasets:
                sub = g[g["dataset"] == ds].sort_values("epsilon")
                x = sub["epsilon"].to_numpy()
                y = sub[col].to_numpy()

                xs, ys = smooth_lowess(x, y, frac=frac)
                ax.plot(xs, ys, linewidth=2, alpha=0.95, label=ds)

            ax.set_xlabel("epsilon")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(5, len(datasets)), frameon=True)

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
