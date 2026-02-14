#!/usr/bin/env python3
"""
Expected tradeoff plots from a directory of CSV runs.

Reads CSVs whose filenames match:
  *_(L0|l0|lam|lambda|epsilon|eps)_<number>.csv

CSV columns (case-insensitive):
  dataset, mask_size, leakage

Computes (per run):
  mask_improvement_pct = abs(mask_size - baseline_k) / baseline_k * 100
  leakage_pct = leakage * 100

Aggregates expected values (mean) per (dataset, param, param_value).

Outputs (PDF):
  1) Combined expected scatter per parameter:
       scatter_expected_<param>.pdf
     Encodings:
       x = expected mask improvement (%)
       y = expected leakage (%)
       shape = dataset
       color = ablated value (DISCRETE steps), dataset-specific colormap
     Adds "directional centroids" per dataset:
       open circle = centroid at min(param_value)
       filled circle = centroid at max(param_value)
       arrow from min -> max (net drift)

  2) Per-dataset dual-axis bar charts per parameter:
       bars_<param>_<dataset>.pdf
     x = ablated values (categorical)
     left y-axis = expected mask improvement (%)  [blue bars]
     right y-axis = expected leakage (%)          [red bars]
     Legends placed above plot to avoid collisions.

Color scaling for scatter (rank -> color step):
  - L0: linear
  - epsilon & lam: log-scaled (log10)

Usage:
  python make_tradeoff_plots.py --dir . --out plots
  python make_tradeoff_plots.py --dir runs --out plots --only L0
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


# ---- Baseline mask sizes (edit as needed) ----
BASELINE_K_SIZE: Dict[str, int] = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}

# ---- Visual encodings ----
DATASET_MARKERS = {
    "airport": "o",
    "hospital": "s",
    "adult": "^",
    "flight": "D",
    "tax": "P",
}

# dataset-specific colormaps (each dataset gets its own gradient family)
DATASET_CMAPS = {
    "airport": "Blues",
    "hospital": "Reds",
    "adult": "Greens",
    "flight": "Purples",
    "tax": "Oranges",
}

# Parameters whose colors should be log-scaled
LOG_COLOR_PARAMS = {"epsilon", "lam"}


def canonical_param(p: str) -> str:
    p = p.lower()
    if p == "l0":
        return "L0"
    if p in {"lam", "lambda"}:
        return "lam"
    if p in {"eps", "epsilon"}:
        return "epsilon"
    return p


def parse_param_value_from_filename(fname: str) -> Optional[Tuple[str, float]]:
    """
    Match filenames like:
      del2ph_L0_0.2.csv
      del2ph_lam_1000.csv
      delgum_epsilon_0.01.csv
      run_eps_1e-2.csv
      run_lambda_0.1.csv
    """
    m = re.match(
        r"^.*_(L0|l0|lam|lambda|epsilon|eps)_([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\.csv$",
        fname,
    )
    if not m:
        return None
    return canonical_param(m.group(1)), float(m.group(2))


def resolve_columns_case_insensitive(df: pd.DataFrame, wanted: Dict[str, str]) -> Dict[str, str]:
    lower_to_actual = {c.lower(): c for c in df.columns}
    resolved = {}
    for logical, desired in wanted.items():
        if desired.lower() in lower_to_actual:
            resolved[logical] = lower_to_actual[desired.lower()]
        elif logical.lower() in lower_to_actual:
            resolved[logical] = lower_to_actual[logical.lower()]
        else:
            raise KeyError(
                f"Missing column for '{logical}'. Tried '{desired}' and '{logical}'. "
                f"Available: {list(df.columns)}"
            )
    return resolved


def load_runs(dirpath: Path, dataset_col: str, mask_col: str, leakage_col: str) -> pd.DataFrame:
    rows = []
    for fp in sorted(dirpath.glob("*.csv")):
        parsed = parse_param_value_from_filename(fp.name)
        if parsed is None:
            continue
        param, val = parsed

        df = pd.read_csv(fp)
        col = resolve_columns_case_insensitive(
            df, {"dataset": dataset_col, "mask_size": mask_col, "leakage": leakage_col}
        )

        out = pd.DataFrame(
            {
                "dataset": df[col["dataset"]].astype(str).str.lower(),
                "mask_size": pd.to_numeric(df[col["mask_size"]], errors="coerce"),
                "leakage": pd.to_numeric(df[col["leakage"]], errors="coerce"),
                "param": param,
                "param_value": val,
            }
        )
        rows.append(out)

    if not rows:
        raise FileNotFoundError(
            f"No matching CSVs found in {dirpath}. Expected *_(L0|lam|epsilon)_<num>.csv"
        )

    data = pd.concat(rows, ignore_index=True)
    data = data.dropna(subset=["dataset", "mask_size", "leakage", "param_value"])
    return data


def compute_expected(data: pd.DataFrame) -> pd.DataFrame:
    baseline = {k.lower(): float(v) for k, v in BASELINE_K_SIZE.items()}

    missing = sorted(set(data["dataset"]) - set(baseline.keys()))
    if missing:
        raise KeyError(f"Missing baselines for datasets: {missing}. Update BASELINE_K_SIZE.")

    data = data.copy()
    data["baseline_k"] = data["dataset"].map(baseline)

    data["mask_improvement_pct"] = (
        (data["mask_size"] - data["baseline_k"]).abs() / data["baseline_k"] * 100.0
    )
    data["leakage_pct"] = data["leakage"] * 100.0

    exp = (
        data.groupby(["param", "param_value", "dataset"], as_index=False)[
            ["mask_improvement_pct", "leakage_pct"]
        ]
        .mean()
        .sort_values(["param", "dataset", "param_value"])
    )
    return exp


def make_discrete_cmap(base_cmap_name: str, n: int) -> ListedColormap:
    """Stark discrete colors sampled from the saturated part of the colormap."""
    base = plt.get_cmap(base_cmap_name)
    positions = np.linspace(0.25, 0.95, n)
    colors = [base(p) for p in positions]
    return ListedColormap(colors, name=f"{base_cmap_name}_disc_{n}")


def value_to_color_rank(values: np.ndarray, param: str) -> Dict[float, int]:
    """
    Map each ablated value -> discrete color index.
      - log10 scaling for epsilon & lam
      - linear scaling for others (L0)
    """
    vals = np.array(sorted({float(v) for v in values}), dtype=float)
    n = len(vals)
    if n == 1:
        return {float(vals[0]): 0}

    if param in LOG_COLOR_PARAMS:
        if np.any(vals <= 0):
            raise ValueError(f"Log color scaling for {param} requires all values > 0. Got: {vals}")
        key = np.log10(vals)
    else:
        key = vals

    order = np.argsort(key)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n)
    return {float(vals[i]): int(ranks[i]) for i in range(n)}


def add_directional_centroid(ax, x0, y0, x1, y1, color):
    """
    Drift arrow from (x0,y0) -> (x1,y1) plus endpoint markers.
      open circle  = min(param_value) centroid
      filled circle= max(param_value) centroid
    """
    # Arrow
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.6,
            alpha=0.9,
            shrinkA=4,
            shrinkB=4,
        ),
        zorder=3,
    )

    # Endpoints
    ax.scatter(
        [x0],
        [y0],
        s=140,
        marker="o",
        facecolors="none",
        edgecolors=color,
        linewidths=2.0,
        zorder=4,
    )
    ax.scatter(
        [x1],
        [y1],
        s=140,
        marker="o",
        c=[color],
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )


def plot_expected_scatter(exp: pd.DataFrame, param: str, outdir: Path) -> Path:
    sub = exp[exp["param"] == param].copy()
    if sub.empty:
        raise ValueError(f"No rows for param={param}")

    values = np.array(sorted(sub["param_value"].unique()), dtype=float)
    n = len(values)
    val_to_idx = value_to_color_rank(values, param)

    fig, ax = plt.subplots(figsize=(9, 6.8))

    dataset_handles = []
    value_handles = []

    for ds in sorted(sub["dataset"].unique()):
        ds_sub = sub[sub["dataset"] == ds].copy()

        marker = DATASET_MARKERS.get(ds, "o")
        disc_cmap = make_discrete_cmap(DATASET_CMAPS.get(ds, "viridis"), n)

        # Scatter points (no connecting lines)
        for _, r in ds_sub.iterrows():
            idx = val_to_idx[float(r["param_value"])]
            ax.scatter(
                r["mask_improvement_pct"],
                r["leakage_pct"],
                s=90,
                marker=marker,
                c=[disc_cmap(idx)],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.98,
            )

        # Directional centroids: min -> max param_value
        low_val = float(ds_sub["param_value"].min())
        high_val = float(ds_sub["param_value"].max())

        low_centroid = ds_sub[ds_sub["param_value"] == low_val][
            ["mask_improvement_pct", "leakage_pct"]
        ].mean()
        high_centroid = ds_sub[ds_sub["param_value"] == high_val][
            ["mask_improvement_pct", "leakage_pct"]
        ].mean()

        arrow_color = disc_cmap(n - 1)  # strongest shade for the dataset
        add_directional_centroid(
            ax,
            float(low_centroid["mask_improvement_pct"]),
            float(low_centroid["leakage_pct"]),
            float(high_centroid["mask_improvement_pct"]),
            float(high_centroid["leakage_pct"]),
            color=arrow_color,
        )

        # Dataset legend entry (shape)
        dataset_handles.append(
            Line2D(
                [], [],
                marker=marker,
                linestyle="None",
                markerfacecolor=disc_cmap(max(0, n - 1)),
                markeredgecolor="black",
                markersize=10,
                label=ds,
            )
        )

    ax.set_xlabel("Expected mask improvement (%)")
    ax.set_ylabel("Expected leakage (%)")
    ax.set_title(f"{param} ablation (expected)")

    # Dataset legend
    leg1 = ax.legend(handles=dataset_handles, title="Dataset", frameon=True, loc="upper left")
    ax.add_artist(leg1)

    # Value legend (neutral swatches; keeps plot uncluttered)
    grey = plt.get_cmap("Greys")
    for i, v in enumerate(values):
        value_handles.append(
            Line2D(
                [], [],
                marker="o",
                linestyle="None",
                markerfacecolor=grey(0.2 + 0.6 * i / max(1, (n - 1))),
                markeredgecolor="black",
                markersize=8,
                label=f"{v:g}",
            )
        )
    ax.legend(
        handles=value_handles,
        title=param,
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"scatter_expected_{param}.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_dual_axis_bars(exp: pd.DataFrame, param: str, outdir: Path) -> Dict[str, Path]:
    """
    One bar chart per dataset:
      x = ablated values
      left y = expected mask improvement (%)  [blue]
      right y = expected leakage (%)          [red]
    Legends placed above axes to avoid collisions with bars.
    """
    sub = exp[exp["param"] == param].copy()
    if sub.empty:
        raise ValueError(f"No rows for param={param}")

    outdir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    for ds in sorted(sub["dataset"].unique()):
        d = sub[sub["dataset"] == ds].copy().sort_values("param_value")

        x_vals = d["param_value"].values.astype(float)
        x = np.arange(len(x_vals))
        mask = d["mask_improvement_pct"].values
        leak = d["leakage_pct"].values

        fig, ax1 = plt.subplots(figsize=(9, 4.8))
        ax2 = ax1.twinx()

        width = 0.42
        ax1.bar(
            x - width / 2,
            mask,
            width=width,
            color="tab:blue",
            label="Mask improvement",
        )
        ax2.bar(
            x + width / 2,
            leak,
            width=width,
            color="tab:red",
            label="Leakage",
        )

        ax1.set_xlabel(param)
        ax1.set_xticks(x, [f"{v:g}" for v in x_vals])
        ax1.set_ylabel("Expected mask improvement (%)")
        ax2.set_ylabel("Expected leakage (%)")
        ax1.set_title(f"{ds} — {param} ablation (expected)")

        # Legends ABOVE the plot area (prevents collisions)
        ax1.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, 1.15),
            frameon=True,
            handlelength=1.2,
        )
        ax2.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 1.15),
            frameon=True,
            handlelength=1.2,
        )

        fig.tight_layout()
        outpath = outdir / f"bars_{param}_{ds}.pdf"
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        outputs[ds] = outpath

    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing CSVs")
    ap.add_argument("--out", default="plots", help="Output directory")
    ap.add_argument("--only", default="", help="Comma-separated params to plot: L0,lam,epsilon")
    ap.add_argument("--dataset_col", default="dataset")
    ap.add_argument("--mask_col", default="mask_size")
    ap.add_argument("--leakage_col", default="leakage")
    args = ap.parse_args()

    data = load_runs(Path(args.dir), args.dataset_col, args.mask_col, args.leakage_col)
    exp = compute_expected(data)

    params = sorted(exp["param"].unique())
    if args.only.strip():
        wanted = [canonical_param(p.strip()) for p in args.only.split(",") if p.strip()]
        params = [p for p in params if p in wanted]

    if not params:
        raise ValueError("No parameters selected to plot.")

    outdir = Path(args.out)
    print("Plotting params:", params)

    for p in params:
        scatter_pdf = plot_expected_scatter(exp, p, outdir)
        print("Wrote:", scatter_pdf)

        bar_paths = plot_dual_axis_bars(exp, p, outdir)
        for _, path in bar_paths.items():
            print("Wrote:", path)


if __name__ == "__main__":
    main()
