#!/usr/bin/env python3
"""
Better plots showing BOTH leakage and mask size from:
  leakage_trends_actuals_fixed_8.csv

Generates:
  1) Per-dataset: leakage vs mask_size (mean + IQR band) line plot
  2) Global: joint plot (scatter + marginals) leakage vs mask_size colored by dataset
  3) Dataset summary: dual-axis bar/line (mean leakage + mean mask_size)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# FIXED INPUT
# -------------------------
CSV_FILE = "leakage_data/leakage_trends_actuals_fixed_10.csv"
OUTDIR = Path("plots_better")
OUTDIR.mkdir(exist_ok=True)

TAU = 1.0  # your file is tau=1.0 only, but keep filter anyway


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def load_df():
    df = pd.read_csv(CSV_FILE)
    # filter tau if present
    if "tau" in df.columns:
        df["tau"] = _to_num(df["tau"])
        df = df[df["tau"] == TAU]

    df["leakage"] = _to_num(df["leakage"])
    df["mask_size"] = _to_num(df["mask_size"])

    # drop bad rows
    df = df[df["leakage"].notna() & df["mask_size"].notna()]
    df["mask_size"] = df["mask_size"].astype(int)

    # stable dataset order
    df["dataset"] = df["dataset"].astype(str)
    return df


def plot_leakage_vs_masksize_binned(df: pd.DataFrame):
    """
    For each dataset:
      group by mask_size
      plot mean leakage with an IQR band (25-75%)
    """
    datasets = sorted(df["dataset"].unique().tolist())
    n = len(datasets)

    # 1 row of panels if <=5 else grid
    if n <= 5:
        rows, cols = 1, n
    else:
        cols = min(3, n)
        rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), squeeze=False)
    axes = axes.flatten()

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df["dataset"] == ds].copy()

        g = sub.groupby("mask_size")["leakage"]
        x = g.mean().index.values
        mean = g.mean().values
        q25 = g.quantile(0.25).values
        q75 = g.quantile(0.75).values
        npts = g.size().values

        ax.plot(x, mean, marker="o", linewidth=2)
        ax.fill_between(x, q25, q75, alpha=0.2)

        ax.set_title(f"{ds}: leakage vs mask_size")
        ax.set_xlabel("mask_size")
        ax.set_ylabel("leakage")

        # annotate counts lightly
        for xx, yy, nn in zip(x, mean, npts):
            ax.annotate(str(nn), (xx, yy), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("τ=1.0: Mean leakage vs mask size (band = IQR, labels = count at mask_size)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTDIR / "panel_leakage_vs_masksize_binned.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_joint_scatter_with_marginals(df: pd.DataFrame):
    """
    One “publication-style” figure:
      - scatter of (mask_size, leakage), colored by dataset
      - marginal hist for mask_size (top)
      - marginal hist for leakage (right)
    Pure matplotlib (no seaborn).
    """
    datasets = sorted(df["dataset"].unique().tolist())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], wspace=0.05, hspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # scatter
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        ax_scatter.scatter(
            sub["mask_size"].values,
            sub["leakage"].values,
            s=18,
            alpha=0.65,
            label=ds,
            color=colors[i % len(colors)],
        )

    ax_scatter.set_xlabel("mask_size")
    ax_scatter.set_ylabel("leakage")
    ax_scatter.grid(True, alpha=0.25)
    ax_scatter.legend(loc="best", frameon=True)

    # top hist of mask_size
    all_ms = df["mask_size"].values
    bins_ms = np.arange(all_ms.min() - 0.5, all_ms.max() + 1.5, 1.0)
    ax_top.hist(all_ms, bins=bins_ms)
    ax_top.set_ylabel("count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.grid(True, alpha=0.15)

    # right hist of leakage (horizontal)
    ax_right.hist(df["leakage"].values, bins=30, orientation="horizontal")
    ax_right.set_xlabel("count")
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.grid(True, alpha=0.15)

    fig.suptitle("τ=1.0: leakage vs mask_size (scatter + marginals)")
    out = OUTDIR / "joint_leakage_vs_masksize.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_dataset_dual_axis_summary(df: pd.DataFrame):
    """
    Per dataset:
      bar = mean leakage (left y-axis)
      line = mean mask_size (right y-axis)
    This is a quick “executive summary” chart.
    """
    summary = (
        df.groupby("dataset", as_index=False)
        .agg(mean_leakage=("leakage", "mean"),
             mean_mask_size=("mask_size", "mean"),
             n=("leakage", "size"))
        .sort_values("dataset")
    )

    x = np.arange(len(summary))
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()

    ax1.bar(x, summary["mean_leakage"].values)
    ax2.plot(x, summary["mean_mask_size"].values, marker="o", linewidth=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["dataset"].tolist(), rotation=0)
    ax1.set_ylabel("mean leakage")
    ax2.set_ylabel("mean mask_size")

    # annotate n
    for xi, n in zip(x, summary["n"].values):
        ax1.annotate(f"n={int(n)}", (xi, summary["mean_leakage"].iloc[list(x).index(xi)]),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax1.grid(True, axis="y", alpha=0.25)
    fig.suptitle("τ=1.0: Dataset summary (mean leakage + mean mask size)")
    fig.tight_layout()
    out = OUTDIR / "dataset_dual_axis_summary.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main():
    df = load_df()
    if df.empty:
        raise SystemExit("No rows after filtering. Check CSV contents / tau column.")

    plot_leakage_vs_masksize_binned(df)
    plot_joint_scatter_with_marginals(df)
    plot_dataset_dual_axis_summary(df)

    print(f"✅ Wrote plots to: {OUTDIR.resolve()}")
    print("  - panel_leakage_vs_masksize_binned.png")
    print("  - joint_leakage_vs_masksize.png")
    print("  - dataset_dual_axis_summary.png")


if __name__ == "__main__":
    main()
