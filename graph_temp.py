#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT_FILE = "delgum_data_epsilon_leakage_graph_v4.csv"
OUTPUT_PDF = "graph_x_gum.pdf"

DATASET_HEADER_RE = re.compile(r"^-{2,}\s*([A-Za-z0-9_]+)\s*-{2,}\s*$")


def parse_tempdata(path: str) -> pd.DataFrame:
    rows = []
    current_ds = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = DATASET_HEADER_RE.match(line)
            if m:
                current_ds = m.group(1)
                continue

            # ignore header lines (may repeat)
            if line.lower().startswith("epsilon,"):
                continue

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

    if not rows:
        raise ValueError("No data parsed. Check file format / headers.")

    return pd.DataFrame(
        rows, columns=["dataset", "epsilon", "leakage", "utility", "mask_size"]
    )


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
        # fallback: rolling median + mean
        s = pd.Series(y)
        w = max(3, (len(y) // 5) | 1)
        y_med = s.rolling(window=w, center=True, min_periods=1).median()
        y_smooth = y_med.rolling(window=w, center=True, min_periods=1).mean()
        return x, y_smooth.to_numpy()


# ----------------------------
# Matplotlib config
# ----------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.variant": "small-caps",
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": 13,
})


def main():
    df = parse_tempdata(INPUT_FILE)

    # Mean over runs per dataset/epsilon (denoising step)
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

    # --- Standard process for "integer step" mask sizes ---
    # 1) aggregate (already done)
    # 2) round to integer (so no fractional mask sizes after averaging)
    # 3) plot with ax.step (so it visually looks like a step function)
    g["mask_int"] = np.rint(g["mask_size"]).astype(int)
    # safety: no negatives
    g["mask_int"] = g["mask_int"].clip(lower=0)

    datasets = sorted(g["dataset"].unique())
    frac = 0.40  # smoothing for leakage/utility only

    with PdfPages(OUTPUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))

        # ----------------------------
        # Subplot 1: Leakage + Mask Size (step)
        # ----------------------------
        ax1 = axes[0]
        ax2 = ax1.twinx()

        for ds in datasets:
            sub = g[g["dataset"] == ds].sort_values("epsilon")
            x = sub["epsilon"].to_numpy()

            # Leakage (smooth line)
            y_leakage = sub["leakage"].to_numpy()
            xs, ys_leakage = smooth_lowess(x, y_leakage, frac=frac)
            ax1.plot(
                xs, ys_leakage,
                linewidth=2, alpha=0.95,
                label=ds.capitalize() if ds != "ncvoter" else "NCVoter"
            )

            # Mask size (INTEGER STEP, no smoothing)
            y_mask_int = sub["mask_int"].to_numpy()
            ax2.step(
                x, y_mask_int,
                where="post",           # step look (post is common for sweeps)
                linewidth=2,
                alpha=0.95,
                linestyle="--"
            )

        ax1.set_xlabel("epsilon")
        ax1.set_ylabel("Leakage")
        ax2.set_ylabel("Mask Size (integer)")

        ax1.grid(True, alpha=0.25)

        # Make the mask axis look like discrete integers
        y_max = int(g["mask_int"].max()) if len(g) else 1
        ax2.set_ylim(-0.5, y_max + 0.5)
        ax2.set_yticks(np.arange(0, y_max + 1, 1))

        # ----------------------------
        # Subplot 2: Utility
        # ----------------------------
        ax3 = axes[1]
        for ds in datasets:
            sub = g[g["dataset"] == ds].sort_values("epsilon")
            x = sub["epsilon"].to_numpy()
            y_utility = sub["utility"].to_numpy()

            xs, ys_utility = smooth_lowess(x, y_utility, frac=frac)
            ax3.plot(
                xs, ys_utility,
                linewidth=2, alpha=0.95,
                label=ds.capitalize() if ds != "ncvoter" else "NCVoter"
            )

        ax3.set_xlabel("epsilon")
        ax3.set_ylabel("Utility")
        ax3.grid(True, alpha=0.25)

        # Legend from leakage lines only (cleaner)
        lines1, labels1 = ax1.get_legend_handles_labels()
        fig.legend(
            lines1, labels1,
            loc="upper center", bbox_to_anchor=(0.5, 1.02),
            ncol=len(datasets)
        )

        # Tiny key in corner
        fig.text(
            0.98, 0.98,
            "-- Mask Size (steps)\n— Leakage, Utility",
            ha="right", va="top", fontsize=9
        )

        fig.tight_layout(rect=[0, 0, 1, 0.9])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
