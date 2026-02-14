#!/usr/bin/env python3
"""
plot_grid_mask_improvement_vs_leakage_ci_gradient.py

Changes requested (implemented):
1) Make gradient MORE visible:
   - Gamma-boosted colormap (default --gamma 3.0)
   - Not pure white at low end (uses a faint tinted start color)
   - Higher CI-band alpha and slightly larger points

2) Legend text:
   - Uses STIXGeneral (VLDB-ish) fonts globally
   - Single global legend placed above plots, with enough top margin to avoid overlap

3) Axis labels:
   - Each subplot gets its own y-axis label: "leakage (%)"
   - Each subplot gets its own x-axis label: "mask improvement"

Other requirements preserved:
- exp = blue family, marginal = red family
- epsilon + lam: LOG-scaled gradient
- l0: LINEAR gradient
- leakage plotted as percentage (leakage * 100)
- x = |mask_size - baseline| / baseline
- Grid: 3 rows (epsilon, lam, l0), columns = datasets
- One legend for the whole figure
- Gradient bars (colorbars) for BOTH methods next to each row (one pair per row)
"""

from __future__ import annotations

import argparse
import os
import re
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.ticker as mticker
import glob


# -----------------------------
# Global styling (VLDB-ish)
# -----------------------------
mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# -----------------------------
# Baselines (user-provided)
# -----------------------------
BASELINE_MASK = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}


# -----------------------------
# Filename parsing
# -----------------------------
ABL_RE = re.compile(
    r"(?i)(?:^|_)(epsilon|lam|lambda|l0|L0)(?:_)([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?:\.csv$|_)",
)

def parse_ablation_from_filename(fname: str) -> Optional[Tuple[str, float]]:
    m = ABL_RE.search(fname)
    if not m:
        return None
    abl = m.group(1).lower()
    val = float(m.group(2))
    if abl == "lambda":
        abl = "lam"
    if abl in ("l0", "L0".lower()):
        abl = "l0"
    return (abl, val)


def infer_method_label(fname: str, df: pd.DataFrame) -> str:
    method_str = ""
    if "method" in df.columns:
        method_str = str(df["method"].iloc[0]).lower()

    low = fname.lower()

    if "marginal" in method_str or "marginal" in low:
        return "marginal"

    if any(k in method_str for k in ["del2ph", "delexp", "exponential", "exp"]) or any(
        k in low for k in ["del2ph", "delexp", "exponential"]
    ):
        return "exp"

    return method_str if method_str else "unknown"


# -----------------------------
# Robust CSV loading
# -----------------------------
def load_csv_robust(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if {"dataset", "mask_size", "leakage"}.issubset(df.columns):
            return df
    except Exception:
        pass

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Fix "header glued to first row" (observed in sample marginal_em file)
    text2 = re.sub(
        r"(num_instantiated_cells)(marginal_em,)",
        r"\1\n\2",
        text,
        flags=re.IGNORECASE,
    )
    text2 = re.sub(r"(cells)([A-Za-z0-9_]+,)", r"\1\n\2", text2)

    return pd.read_csv(StringIO(text2))


def coerce_needed_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["dataset", "leakage", "mask_size"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}")

    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip().str.lower()
    out["leakage"] = pd.to_numeric(out["leakage"], errors="coerce")
    out["mask_size"] = pd.to_numeric(out["mask_size"], errors="coerce")
    return out.dropna(subset=needed)


# -----------------------------
# Color + gradient utilities
# -----------------------------
METHOD_COLORS = {
    "exp": "#1f77b4",        # blue
    "marginal": "#d62728",   # red
}

def tinted_low_color(hex_color: str, tint: float = 0.92) -> str:
    """
    Returns a very light tint of the base color (tint close to 1 -> very light).
    This avoids starting from pure white, which often looks like "no gradient".
    """
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    out = white * tint + rgb * (1.0 - tint)
    return mpl.colors.to_hex(out)

def method_cmap(base_hex: str, gamma: float) -> LinearSegmentedColormap:
    """
    More visible gradient:
      - start from a faint tinted version of base color (not pure white)
      - gamma warp to increase contrast (gamma > 1)
    """
    start = tinted_low_color(base_hex, tint=0.92)  # faint tint
    base = LinearSegmentedColormap.from_list("cmap_base", [start, base_hex], N=256)
    xs = np.linspace(0, 1, 256) ** gamma
    colors = base(xs)
    return LinearSegmentedColormap.from_list("cmap_gamma", colors)

def make_norm_for_ablation(values: np.ndarray, ablation_type: str):
    """
    epsilon + lam: LogNorm (values must be >0)
    l0: Normalize (linear)
    """
    values = np.asarray(values, dtype=float)

    if ablation_type in ("epsilon", "lam"):
        pos = values[values > 0]
        if len(pos) == 0:
            return Normalize(vmin=0.0, vmax=1.0)
        vmin = float(pos.min())
        vmax = float(pos.max())
        if vmin == vmax:
            vmax = vmin * 10.0
        return LogNorm(vmin=vmin, vmax=vmax)

    # l0 linear
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        vmax = vmin + 1e-12
    return Normalize(vmin=vmin, vmax=vmax)

def draw_gradient_ci_band(ax, x, y_lo, y_hi, vals, cmap, norm, alpha=0.45):
    """
    Gradient band between y_lo and y_hi; each segment colored by cmap(norm(v_mid)).
    Assumes vals are sorted in the same order as x/y arrays.
    """
    x = np.asarray(x, dtype=float)
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    vals = np.asarray(vals, dtype=float)

    if len(x) < 2:
        return

    polys = []
    colors = []

    for i in range(len(x) - 1):
        polys.append([
            (x[i],   y_lo[i]),
            (x[i+1], y_lo[i+1]),
            (x[i+1], y_hi[i+1]),
            (x[i],   y_hi[i]),
        ])
        v_mid = 0.5 * (vals[i] + vals[i+1])
        try:
            colors.append(cmap(norm(v_mid)))
        except Exception:
            colors.append(cmap(0.5))

    pc = PolyCollection(polys, facecolors=colors, edgecolors="none", alpha=alpha)
    ax.add_collection(pc)


# -----------------------------
# Aggregation
# -----------------------------
def build_aggregated_points(in_dir: str, pattern: str) -> pd.DataFrame:
    """
    Returns tidy dataframe with per-(dataset, ablation_type, ablation_value, method):
      mask_imp_mean, leak_pct_mean, leak_pct_lo, leak_pct_hi
    """
    csv_paths = sorted(glob.glob(os.path.join(in_dir, pattern)))
    rows: List[Dict] = []

    for p in csv_paths:
        fname = os.path.basename(p)
        abl = parse_ablation_from_filename(fname)
        if abl is None:
            continue
        ablation_type, ablation_value = abl

        try:
            df = coerce_needed_columns(load_csv_robust(p))
        except Exception:
            continue

        method_label = infer_method_label(fname, df)

        # baseline mapping
        df["baseline_mask"] = df["dataset"].map(BASELINE_MASK)
        df = df.dropna(subset=["baseline_mask"]).copy()
        if df.empty:
            continue

        # derived metrics
        df["mask_improvement"] = (df["mask_size"] - df["baseline_mask"]).abs() / df["baseline_mask"]
        df["leak_pct"] = df["leakage"] * 100.0

        # aggregate per dataset over runs
        grouped = (
            df.groupby("dataset", as_index=False)
              .agg(
                  mask_imp_mean=("mask_improvement", "mean"),
                  leak_pct_mean=("leak_pct", "mean"),
                  leak_pct_std=("leak_pct", "std"),
                  leak_pct_n=("leak_pct", "count"),
              )
        )

        # 95% CI (normal approx)
        z = 1.96
        grouped["leak_ci_half"] = (
            z * grouped["leak_pct_std"].fillna(0.0) / np.sqrt(grouped["leak_pct_n"].clip(lower=1))
        )
        grouped["leak_pct_lo"] = grouped["leak_pct_mean"] - grouped["leak_ci_half"]
        grouped["leak_pct_hi"] = grouped["leak_pct_mean"] + grouped["leak_ci_half"]

        for _, r in grouped.iterrows():
            rows.append({
                "dataset": str(r["dataset"]).lower(),
                "ablation_type": ablation_type,
                "ablation_value": float(ablation_value),
                "method_label": method_label,
                "mask_imp_mean": float(r["mask_imp_mean"]),
                "leak_pct_mean": float(r["leak_pct_mean"]),
                "leak_pct_lo": float(r["leak_pct_lo"]),
                "leak_pct_hi": float(r["leak_pct_hi"]),
            })

    if not rows:
        raise SystemExit(f"No usable ablation CSVs found in {os.path.abspath(in_dir)} matching {pattern}")

    data = pd.DataFrame(rows)
    data = data[data["method_label"].isin(["exp", "marginal"])].copy()
    if data.empty:
        raise SystemExit("Found files, but could not detect any 'exp' or 'marginal' methods.")
    return data


# -----------------------------
# Plot grid page
# -----------------------------
def plot_grid_page(
    pdf: PdfPages,
    data: pd.DataFrame,
    datasets: List[str],
    ablation_rows: List[str],
    page_title: str,
    gamma: float,
):
    nrows = len(ablation_rows)
    ncols = len(datasets)

    # +2 thin columns for colorbars (exp, marginal)
    fig_w = max(11.0, 3.1 * ncols + 2.2)
    fig_h = 2.9 * nrows + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 2,
        width_ratios=[1.0] * ncols + [0.07, 0.07],
        wspace=0.28,
        hspace=0.38,
    )

    # Single global legend (won't overlap)
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="None",
                         markerfacecolor=METHOD_COLORS["exp"],
                         markeredgecolor="black", markeredgewidth=0.7,
                         markersize=8, label="exp"),
        mpl.lines.Line2D([0], [0], marker="o", linestyle="None",
                         markerfacecolor=METHOD_COLORS["marginal"],
                         markeredgecolor="black", markeredgewidth=0.7,
                         markersize=8, label="marginal"),
    ]

    # Leave ample top space for title+legend
    fig.suptitle(page_title, y=0.995, fontsize=12)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        borderaxespad=0.2,
        handletextpad=0.6,
        columnspacing=1.6,
    )

    # Build each row
    for r, abl in enumerate(ablation_rows):
        row_data = data[data["ablation_type"] == abl]
        if row_data.empty:
            continue

        # Shared norm across this entire row
        norm = make_norm_for_ablation(row_data["ablation_value"].values, abl)

        # Method-specific colormaps with same norm
        cm_exp = method_cmap(METHOD_COLORS["exp"], gamma=gamma)
        cm_marg = method_cmap(METHOD_COLORS["marginal"], gamma=gamma)

        # Plot each dataset column
        for c, ds in enumerate(datasets):
            ax = fig.add_subplot(gs[r, c])

            sub = row_data[row_data["dataset"] == ds]
            if sub.empty:
                ax.set_title(ds)
                ax.axis("off")
                continue

            # Scatter + CI bands for both methods
            for method_label, cmap in [("exp", cm_exp), ("marginal", cm_marg)]:
                msub = sub[sub["method_label"] == method_label].sort_values("ablation_value")
                if msub.empty:
                    continue

                x = msub["mask_imp_mean"].to_numpy()
                y = msub["leak_pct_mean"].to_numpy()
                y_lo = msub["leak_pct_lo"].to_numpy()
                y_hi = msub["leak_pct_hi"].to_numpy()
                vals = msub["ablation_value"].to_numpy()

                # CI band (more visible)
                draw_gradient_ci_band(ax, x, y_lo, y_hi, vals, cmap=cmap, norm=norm, alpha=0.45)

                # scatter (more visible)
                ax.scatter(
                    x, y,
                    c=vals, cmap=cmap, norm=norm,
                    edgecolors="black",
                    linewidth=0.7,
                    s=62,
                    zorder=3,
                )

            ax.set_title(ds)

            # Axis labels for EACH subplot (as requested)
            ax.set_xlabel("mask improvement")
            ax.set_ylabel("leakage (%)")

            ax.grid(True, linewidth=0.35, alpha=0.35)

            # Small row label inside leftmost plot (nice for 3-row grid)
            if c == 0:
                ax.text(
                    0.02, 0.98,
                    f"{abl}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    fontweight="bold",
                )

        # Colorbars to the right for this row
        cax_exp = fig.add_subplot(gs[r, ncols])
        cax_m = fig.add_subplot(gs[r, ncols + 1])

        sm_exp = mpl.cm.ScalarMappable(norm=norm, cmap=cm_exp)
        sm_m = mpl.cm.ScalarMappable(norm=norm, cmap=cm_marg)

        cb1 = fig.colorbar(sm_exp, cax=cax_exp)
        cb2 = fig.colorbar(sm_m, cax=cax_m)

        cb1.set_label("exp", rotation=90, labelpad=6)
        cb2.set_label("marginal", rotation=90, labelpad=6)

        # Nice tick formatting for log norms
        if isinstance(norm, LogNorm):
            for cb in (cb1, cb2):
                cb.formatter = mticker.LogFormatterMathtext()
                cb.update_ticks()

        # Put a small "ablation value" label above the two bars (once per row)
        # (Placed above the exp colorbar axis)
        cax_exp.set_title("ablation\nvalue", fontsize=9, pad=6)

    # Tight layout with room for title+legend
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=".")
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--out_pdf", default="grid_mask_improvement_vs_leakage_ci_gradient.pdf")
    ap.add_argument("--cols_per_page", type=int, default=5, help="Max dataset columns per page")
    ap.add_argument("--gamma", type=float, default=3.0, help="Increase for more distinct gradient (try 3.5 or 4.0)")
    args = ap.parse_args()

    data = build_aggregated_points(args.in_dir, args.glob)

    present_datasets = sorted([d for d in data["dataset"].unique().tolist() if d in BASELINE_MASK])
    if not present_datasets:
        raise SystemExit("No datasets matched the baseline list after parsing.")

    # Fixed row order: 3 rows
    desired_rows = ["epsilon", "lam", "l0"]
    ablation_rows = [a for a in desired_rows if a in set(data["ablation_type"].unique())]
    if not ablation_rows:
        raise SystemExit("No ablation types (epsilon/lam/l0) found in filenames.")

    cols_per_page = max(1, int(args.cols_per_page))
    chunks = [present_datasets[i:i + cols_per_page] for i in range(0, len(present_datasets), cols_per_page)]

    with PdfPages(args.out_pdf) as pdf:
        for idx, ds_chunk in enumerate(chunks, start=1):
            title = "Mask improvement vs Leakage (%) — exp vs marginal"
            title += "\nRows: epsilon (log color), lambda (log color), L0 (linear color)"
            if len(chunks) > 1:
                title += f"  |  page {idx}/{len(chunks)}"
            plot_grid_page(
                pdf=pdf,
                data=data,
                datasets=ds_chunk,
                ablation_rows=ablation_rows,
                page_title=title,
                gamma=args.gamma,
            )

    print(f"Wrote: {args.out_pdf}")


if __name__ == "__main__":
    main()
