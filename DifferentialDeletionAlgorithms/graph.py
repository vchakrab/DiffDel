#!/usr/bin/env python3
"""
DiffDel paper figures (LaTeX-friendly split) — roman numerals restart PER PDF.

PDF 1: Ratio plots ONLY (stacked):
  (i) Tradeoff: Leakage vs |M|
  (ii) Deletion Ratio vs #Constraints (gamma=0.25)
  -> Legend: METHOD SHAPES ONLY (labels: DEL, EXP, GUM)

PDF 2: Ablation ONLY (2x6) — NO axis titles:
  (i) ... (xii) across the 12 panels (row-major)
  -> Legend: DATASET COLORS as RECTANGLES + MECHANISM LINESTYLES (EXP solid, GUM dashed)

PDF 3: Radar (top row) + Runtime (bottom row):
  - Roman numerals for ALL 10 panels (row-major): (i)...(x)
  - Dataset titles ONLY on top row (radar); none on bottom row bars.
  - NO figure-level title.
  - Phase legend labels: Instantiation, Modeling, Update To NULL
  - Cells legend labels: Instantiated Cells, Mask Size

Legend overlap fix:
  - Put legend in a dedicated top row axes (not fig.legend floating).
  - This eliminates overlap while keeping whitespace minimal.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter

# =============================================================================
# STYLE (match your reference style)
# =============================================================================
# =============================================================================
# STYLE (force EVERYTHING to 10pt)
# =============================================================================
# =============================================================================
# STYLE (force EVERYTHING to 12pt)
# =============================================================================
FS = 12  # global font size

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FS,

    # axes
    "axes.labelsize": FS,
    "axes.titlesize": FS,

    # legend
    "legend.fontsize": FS,
    "legend.title_fontsize": FS,

    # ticks
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,

    # figure/export
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    # grid / spacing
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelpad": 2.0,
})
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"


# =============================================================================
# Roman numerals (small-caps)
# =============================================================================
_ROMAN = [
    "i","ii","iii","iv","v","vi","vii","viii","ix","x",
    "xi","xii","xiii","xiv","xv","xvi","xvii","xviii","xix","xx"
]

def roman(k1_based: int) -> str:
    if 1 <= k1_based <= len(_ROMAN):
        return f"({_ROMAN[k1_based-1]})"
    return f"({k1_based})"

def add_roman_outside(ax, txt: str, x: float = 0.5, y: float = 1.02):
    ax.text(
        x, y, txt,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=FS,
        fontvariant="small-caps",
        clip_on=False,
    )


def _style_ticks(ax):
    ax.tick_params(axis="both", which="both", direction="out", length=3.0, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# =============================================================================
# CONSISTENT COLOR / MARKER / LINESTYLE SCHEME
# =============================================================================
DATASET_ORDER = ["Airport", "Hospital", "Adult", "Flight", "Tax"]
DATASET_COLORS = {
    "Airport": "#1f77b4",
    "Hospital": "#ff7f0e",
    "Adult": "#2ca02c",
    "Flight": "#d62728",
    "Tax": "#9467bd",
}

METHOD_ORDER = ["DelMin", "Del2Ph", "DelGum"]
METHOD_LABEL = {"DelMin": "MIN", "Del2Ph": "EXP", "DelGum": "GUM"}
MARKERS = {"DelMin": "s", "Del2Ph": "^", "DelGum": "o"}
LINESTYLES = {"DelMin": "-", "Del2Ph": "--", "DelGum": ":"}
METHOD_COLORS = {"DelMin": "#2ecc71", "Del2Ph": "#3498db", "DelGum": "#e74c3c"}

Z_95 = 1.96

# Runtime extras
ZONE_COLOR_LIGHT = "#D2B48C"
ZONE_COLOR_DARK  = "#8B4513"

# =============================================================================
# Roman numerals (OUTSIDE, LEFT-aligned) + small-caps
# =============================================================================

# =============================================================================
# Helpers
# =============================================================================
def _maybe_lambda_to_value(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x
    if finite.min() >= 0 and finite.max() <= 10 and np.allclose(finite, np.round(finite)):
        return np.power(10.0, x)
    return x

def _leakage_to_percent(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return y
    return 100.0 * y if np.nanmax(finite) <= 1.5 else y

# =============================================================================
# DATA LOADING (main)
# =============================================================================
def load_main_results(data_dir: Path, delmin_csv: str, del2ph_csv: str, delgum_csv: str) -> pd.DataFrame:
    new_columns = [
        "method", "dataset", "target_attribute", "total_time", "init_time",
        "model_time", "del_time", "leakage", "baseline_leakage_empty_mask",
        "utility", "total_paths", "mask_size", "model_size", "num_instantiated_cells"
    ]

    del2ph = pd.read_csv(data_dir / del2ph_csv, header=None, skiprows=1, names=new_columns)
    delgum = pd.read_csv(data_dir / delgum_csv, header=None, skiprows=1, names=new_columns)
    del2ph["method"] = "Del2Ph"
    delgum["method"] = "DelGum"

    delmin = pd.read_csv(data_dir / delmin_csv)
    delmin.columns = [str(c).strip() for c in delmin.columns]
    delmin["method"] = "DelMin"

    aliases = {
        "Dataset": "dataset", "ds": "dataset",
        "deletion_time": "del_time", "delete_time": "del_time", "update_time": "del_time",
        "masked_cells": "mask_size", "num_masked_cells": "mask_size", "mask": "mask_size",
        "instantiated_cells": "num_instantiated_cells", "model_cells": "num_instantiated_cells", "num_inst_cells": "num_instantiated_cells",
        "instantiated_model_size": "model_size",
        "paths": "total_paths",
    }
    for src, dst in aliases.items():
        if src in delmin.columns and dst not in delmin.columns:
            delmin = delmin.rename(columns={src: dst})

    for col in ["init_time", "model_time", "del_time"]:
        if col not in delmin.columns:
            delmin[col] = 0.0
    for col in ["total_time", "leakage", "utility", "total_paths", "mask_size", "model_size", "num_instantiated_cells"]:
        if col not in delmin.columns:
            delmin[col] = np.nan

    df = pd.concat([delmin, del2ph, delgum], ignore_index=True)
    df["dataset"] = df["dataset"].astype(str).str.strip().str.capitalize()

    numeric_cols = [
        "total_time", "init_time", "model_time", "del_time",
        "mask_size", "model_size", "num_instantiated_cells",
        "leakage", "utility", "total_paths"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["init_time_ms"] = df["init_time"] * 1000.0
    df["model_time_ms"] = df["model_time"] * 1000.0
    df["update_time_ms"] = df["del_time"] * 1000.0
    df["time_ms"] = df["init_time_ms"] + df["model_time_ms"] + df["update_time_ms"]

    df["memory_kb"] = df["model_size"] / 1024.0

    denom = df["num_instantiated_cells"].astype(float).clip(lower=1.0)
    df["deletion_ratio"] = (df["mask_size"].astype(float) / denom).clip(0.0, 1.0)

    return df

# =============================================================================
# Legend helper: dedicated legend row (prevents overlap)
# =============================================================================
def _legend_row(fig, gs_cell, handles, labels, ncol, handlelength=1.6, columnspacing=1.0, handletextpad=0.4):
    ax_leg = fig.add_subplot(gs_cell)
    ax_leg.axis("off")
    ax_leg.legend(
        handles, labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=handlelength,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        borderaxespad=0.0
    )
    return ax_leg

# =============================================================================
# PDF 1: Ratio plots
# =============================================================================
def plot_leakage_tradeoff(ax, df: pd.DataFrame) -> None:
    summary = df.groupby(["dataset", "method"]).agg(mask_size=("mask_size", "mean"),
                                                    leakage=("leakage", "mean")).reset_index()
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            row = summary[(summary["dataset"] == dataset) & (summary["method"] == method)]
            if row.empty:
                continue
            ax.scatter(
                float(row["leakage"].values[0]),
                float(row["mask_size"].values[0]),
                s=55,
                c=DATASET_COLORS[dataset],
                marker=MARKERS[method],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )

    ax.annotate("", xy=(0.38, 3), xytext=(0.02, 11),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.set_xlabel(r"Leakage $\mathcal{L}$")
    ax.xaxis.set_label_coords(0.5, -0.12)

    ax.set_ylabel(r"Auxiliary Deletions $|M|$")
    ax.set_xlim(-0.03, 0.95)
    ax.set_ylim(0, 12)
    _style_ticks(ax)

def plot_deletion_ratio(ax, df: pd.DataFrame) -> None:
    fallback = {"Airport": 7, "Hospital": 40, "Adult": 57, "Flight": 112, "Tax": 31}
    has_paths = df["total_paths"].notna().any()
    xmap = df.groupby("dataset").agg(x=("total_paths", "mean"))["x"].to_dict() if has_paths else fallback

    summary = df.groupby(["dataset", "method"]).agg(deletion_ratio=("deletion_ratio", "mean")).reset_index()

    xs = []
    for dataset in DATASET_ORDER:
        x = float(xmap.get(dataset, np.nan))
        for method in METHOD_ORDER:
            row = summary[(summary["dataset"] == dataset) & (summary["method"] == method)]
            if row.empty or not np.isfinite(x):
                continue
            y = float(row["deletion_ratio"].values[0])
            xs.append(x)
            ax.scatter(
                x, y,
                s=55,
                c=DATASET_COLORS[dataset],
                marker=MARKERS[method],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"Number of Constraints ($\gamma=0.25$)")
    ax.set_ylabel(r"Deletion Ratio $|M|/|\mathcal{I}(c^*)|$")
    if xs:
        ax.set_xlim(min(xs) - 5, max(xs) + 5)
    ax.set_ylim(0.0, 1.15)
    _style_ticks(ax)

def legend_shapes_methods():
    handles = [
        Line2D([0], [0], marker=MARKERS[m], linestyle="None",
               markerfacecolor="white", markeredgecolor="black",
               markersize=6, label=METHOD_LABEL[m])
        for m in METHOD_ORDER
    ]
    return handles, [h.get_label() for h in handles]

def build_ratio_pdf(df: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(8.2, 3.6))

    # 2 rows: legend + plots
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.16, 1.0],
        hspace=0.22,
        wspace=0.28,
        left=0.08, right=0.995,
        top=0.95, bottom=0.16
    )

    # --- Combined legend (methods + datasets) ---
    handles_m, labels_m = legend_shapes_methods()
    handles_d, labels_d = legend_ablation_datasets_rectangles()
    handles = handles_m + [Line2D([0],[0], color="none", label=" ")] + handles_d
    labels  = labels_m + [" "] + labels_d

    _legend_row(
        fig, gs[0, :],
        handles, labels,
        ncol=len(handles),
        handlelength=1.6,
        columnspacing=1.0
    )

    # --- Side-by-side plots ---
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    plot_leakage_tradeoff(ax1, df)
    plot_deletion_ratio(ax2, df)
    add_roman_outside(ax1, roman(1), y = 1.04)
    add_roman_outside(ax2, roman(2), y = 1.04)

    # --- Roman numerals ABOVE panels (safe, no collisions) ---


    out_file = out_dir / "fig_ratio_plots.pdf"
    fig.savefig(out_file)
    plt.close(fig)
    print("Wrote", out_file)



# =============================================================================
# PDF 2: Ablation only
# =============================================================================
def _read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, header=None)

def load_sweep(d: Path, key: str) -> Dict[str, pd.DataFrame]:
    files = sorted(d.glob(f"*{key}*.csv"))
    if not files:
        raise FileNotFoundError(f"No ablation CSVs found matching '*{key}*.csv' in {d}")
    buckets: Dict[str, list] = {"Exp": [], "Gum": []}
    for f in files:
        name = f.name.lower()
        mech = "Gum" if ("gum" in name or "gumbel" in name) else "Exp"
        df = _read_csv_safely(f)
        df["__file__"] = f.name
        buckets[mech].append(df)
    out: Dict[str, pd.DataFrame] = {}
    for mech, lst in buckets.items():
        if lst:
            out[mech] = pd.concat(lst, ignore_index=True)
    return out

def compute_stats(df: pd.DataFrame, sweep_col: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    ren = {
        "Dataset": "dataset", "ds": "dataset",
        "lambda": "lambda", "lam": "lambda",
        "eps": "epsilon",
        "L0": "L0",
        "masked_cells": "mask_size", "num_masked_cells": "mask_size", "mask": "mask_size",
    }
    for a, b in ren.items():
        if a in df.columns and b not in df.columns:
            df = df.rename(columns={a: b})

    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.strip().str.capitalize()

    for c in [sweep_col, "mask_size", "leakage"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = df.dropna(subset=[sweep_col, "mask_size", "leakage", "dataset"])
    g = keep.groupby(["dataset", sweep_col]).agg(
        n=("mask_size", "count"),
        mask_mean=("mask_size", "mean"),
        mask_std=("mask_size", "std"),
        leak_mean=("leakage", "mean"),
        leak_std=("leakage", "std"),
    ).reset_index().sort_values(["dataset", sweep_col])

    g["mask_ci"] = Z_95 * (g["mask_std"].fillna(0.0) / np.sqrt(g["n"].clip(lower=1)))
    g["leak_ci"] = Z_95 * (g["leak_std"].fillna(0.0) / np.sqrt(g["n"].clip(lower=1)))
    return g

def _format_lambda_ticks(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    def _exp_fmt(val, pos):
        if val <= 0:
            return ""
        e = np.log10(val)
        if np.isfinite(e) and abs(e - round(e)) < 1e-9:
            return f"{int(round(e))}"
        return ""
    ax.xaxis.set_major_formatter(FuncFormatter(_exp_fmt))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$\lambda\$")

def _format_epsilon_ticks(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    def _exp_fmt(val, pos):
        if val <= 0:
            return ""
        e = np.log10(val)
        if np.isfinite(e) and abs(e - round(e)) < 1e-9:
            return f"{int(round(e))}"
        return ""
    ax.xaxis.set_major_formatter(FuncFormatter(_exp_fmt))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("𝜖")

def _format_linear_ticks(ax, lab):
    ax.set_xscale("linear")
    ax.set_xlabel(lab)

def plot_ablation_block(fig, subspec, lam, eps, l0, baseline_mask: Dict[str, float]) -> Dict[Tuple[str,str,str], plt.Axes]:
    gs = subspec.subgridspec(2, 6, hspace=0.46, wspace=0.36)

    def ax_at(r, c): return fig.add_subplot(gs[r, c])

    axes = {
        ("Exp","lam","mask"): ax_at(0,0), ("Exp","lam","leak"): ax_at(0,1),
        ("Exp","eps","mask"): ax_at(0,2), ("Exp","eps","leak"): ax_at(0,3),
        ("Exp","l0","mask"):  ax_at(0,4), ("Exp","l0","leak"):  ax_at(0,5),
        ("Gum","lam","mask"): ax_at(1,0), ("Gum","lam","leak"): ax_at(1,1),
        ("Gum","eps","mask"): ax_at(1,2), ("Gum","eps","leak"): ax_at(1,3),
        ("Gum","l0","mask"):  ax_at(1,4), ("Gum","l0","leak"):  ax_at(1,5),
    }

    def drop_lambda1(dfstats):
        return dfstats[dfstats["lambda"] != 1] if "lambda" in dfstats.columns else dfstats

    def mask_to_improvement(mask_mean: np.ndarray, mask_ci: np.ndarray, base: float):
        base = float(base) if (base is not None and np.isfinite(base) and base > 0) else np.nan
        if not np.isfinite(base):
            return mask_mean, mask_mean - mask_ci, mask_mean + mask_ci
        mean_imp = 100.0 * (1.0 - (mask_mean / base))
        lower_mask = mask_mean - mask_ci
        upper_mask = mask_mean + mask_ci
        lower_imp = 100.0 * (1.0 - (upper_mask / base))
        upper_imp = 100.0 * (1.0 - (lower_mask / base))
        return mean_imp, lower_imp, upper_imp

    def plot_sweep(mech: str, sweep_key: str, sweep_col: str, stat_df: pd.DataFrame, linestyle: str):
        for dataset in DATASET_ORDER:
            ddf = stat_df[stat_df["dataset"] == dataset]
            if ddf.empty:
                continue

            x = ddf[sweep_col].to_numpy(dtype=float)
            if sweep_col == "lambda":
                x = _maybe_lambda_to_value(x)

            m_mean = ddf["mask_mean"].to_numpy(dtype=float)
            m_ci   = ddf["mask_ci"].to_numpy(dtype=float)
            l_mean = _leakage_to_percent(ddf["leak_mean"].to_numpy(dtype=float))
            l_ci   = _leakage_to_percent(ddf["leak_ci"].to_numpy(dtype=float))

            base = baseline_mask.get(dataset, np.nan)
            imp_mean, imp_lo, imp_hi = mask_to_improvement(m_mean, m_ci, base)

            c = DATASET_COLORS[dataset]
            axm = axes[(mech, sweep_key, "mask")]
            axl = axes[(mech, sweep_key, "leak")]

            axm.plot(x, imp_mean, linestyle=linestyle, marker="o", markersize=3.0, linewidth=1.2, color=c)
            axm.fill_between(x, imp_lo, imp_hi, alpha=0.15, color=c, linewidth=0)

            axl.plot(x, l_mean, linestyle=linestyle, marker="o", markersize=3.0, linewidth=1.2, color=c)
            axl.fill_between(x, l_mean - l_ci, l_mean + l_ci, alpha=0.15, color=c, linewidth=0)

    # Mechanism encoding (as requested): Exp = solid, Gum = dashed
    plot_sweep("Exp","lam","lambda", drop_lambda1(lam["Exp"]), "-")
    plot_sweep("Exp","eps","epsilon", eps["Exp"], "-")
    plot_sweep("Exp","l0","L0", l0["Exp"], "-")

    plot_sweep("Gum","lam","lambda", drop_lambda1(lam["Gum"]), "--")
    plot_sweep("Gum","eps","epsilon", eps["Gum"], "--")
    plot_sweep("Gum","l0","L0", l0["Gum"], "--")

    for mech in ["Exp", "Gum"]:
        for sweep in ["lam", "eps", "l0"]:
            axes[(mech, sweep, "mask")].set_ylabel("Mask size improvement (%)")
            axes[(mech, sweep, "leak")].set_ylabel("Leakage (%)")
            _style_ticks(axes[(mech, sweep, "mask")])
            _style_ticks(axes[(mech, sweep, "leak")])

    for mech in ["Exp","Gum"]:
        _format_lambda_ticks(axes[(mech,"lam","mask")]); _format_lambda_ticks(axes[(mech,"lam","leak")])
        _format_epsilon_ticks(axes[(mech,"eps","mask")]); _format_epsilon_ticks(axes[(mech,"eps","leak")])
        _format_linear_ticks(axes[(mech,"l0","mask")], r"$L_0$"); _format_linear_ticks(axes[(mech,"l0","leak")], r"$L_0$")

    # --- Visualization aid: vertical separators between sweep groups (λ | ε | L0) ---
    # Place 2 vertical lines in the gaps between columns (1|2) and (3|4), spanning both rows.
    # --- Visualization aid: vertical separators between sweep groups (λ | ε | L0) ---
    # Place 2 vertical lines in the gaps between columns (1|2) and (3|4), spanning both rows.

    y0 = axes[("Gum", "lam", "mask")].get_position().y0
    y1 = axes[("Exp", "lam", "mask")].get_position().y1

    # trim slightly to avoid tick/label regions
    y0 = y0 + 0.010
    y1 = y1 - 0.006

    def _gap_separator(ax_left, ax_right, frac = 0.15):
        """
        Draw a separator in the whitespace gap between two axes.
        frac: 0 -> at left edge of the gap (right next to ax_left),
              1 -> at right edge of the gap (next to ax_right).
        We want it near ax_left so it sits on the OTHER side of ax_right's y-label.
        """
        xL = ax_left.get_position().x1
        xR = ax_right.get_position().x0
        x = xL + frac * (xR - xL)
        fig.add_artist(Line2D([x, x], [y0, y1],
                              transform = fig.transFigure,
                              color = "black", lw = 1.0, alpha = 0.9, zorder = 0))

    # between λ and ε
    _gap_separator(axes[("Exp", "lam", "leak")], axes[("Exp", "eps", "mask")], frac = 0.15)

    # between ε and L0
    _gap_separator(axes[("Exp", "eps", "leak")], axes[("Exp", "l0", "mask")], frac = 0.15)

    # between λ (col 0-1) and ε (col 2-3)
    # _gap_separator(axes[("Exp", "lam", "leak")], axes[("Exp", "eps", "mask")], frac = 0.72)
    #
    # # between ε (col 2-3) and L0 (col 4-5)
    # _gap_separator(axes[("Exp", "eps", "leak")], axes[("Exp", "l0", "mask")], frac = 0.72)

    return axes

def legend_ablation_datasets_rectangles():
    handles = [Patch(facecolor=DATASET_COLORS[d], edgecolor="black", linewidth=0.4, label=d) for d in DATASET_ORDER]
    return handles, [h.get_label() for h in handles]

def legend_ablation_method_lines():
    # Mechanism legend only (as requested): Exp solid, Gum dashed
    handles = [
        Line2D([0],[0], color="black", lw=1.6, linestyle="-",  label="EXP"),
        Line2D([0],[0], color="black", lw=1.6, linestyle="--", label="GUM"),
    ]
    return handles, [h.get_label() for h in handles]
def drop_epsilon_001(dfstats: pd.DataFrame) -> pd.DataFrame:
    # remove epsilon = 0.01 (tolerant to float issues)
    if "epsilon" not in dfstats.columns:
        return dfstats
    return dfstats[~np.isclose(dfstats["epsilon"].astype(float), 0.01)]


def build_ablation_pdf(df: pd.DataFrame, ablation_dir: Path, out_dir: Path) -> None:
    fig = plt.figure(figsize=(15.2, 4.7))

    gs_outer = fig.add_gridspec(
        2, 1,
        height_ratios=[0.16, 1.0],
        hspace=0.04,
        left=0.04, right=0.995,
        top=0.98, bottom=0.12
    )

    ds_h, ds_l = legend_ablation_datasets_rectangles()
    m_h, m_l = legend_ablation_method_lines()
    handles = ds_h + [Line2D([0],[0], color="none", label=" ")] + m_h
    labels  = ds_l + [" "] + m_l
    _legend_row(
        fig, gs_outer[0, 0],
        handles, labels,
        ncol=len(DATASET_ORDER) + 1 + 2,
        handlelength=1.9,
        columnspacing=1.1
    )

    lam_raw = load_sweep(ablation_dir, "lam")
    eps_raw = load_sweep(ablation_dir, "epsilon")
    l0_raw  = load_sweep(ablation_dir, "L0")

    lam = {k: compute_stats(v, "lambda") for k, v in lam_raw.items()}
    eps = {k: drop_epsilon_001(compute_stats(v, "epsilon")) for k, v in eps_raw.items()}

    l0  = {k: compute_stats(v, "L0") for k, v in l0_raw.items()}

    base = (
        df[df["method"] == "DelMin"]
        .groupby("dataset")
        .agg(base_mask=("mask_size","mean"))["base_mask"]
        .to_dict()
    )

    axes = plot_ablation_block(fig, gs_outer[1, 0], lam, eps, l0, baseline_mask=base)

    order = [
        ("Exp","lam","mask"), ("Exp","lam","leak"),
        ("Exp","eps","mask"), ("Exp","eps","leak"),
        ("Exp","l0","mask"),  ("Exp","l0","leak"),
        ("Gum","lam","mask"), ("Gum","lam","leak"),
        ("Gum","eps","mask"), ("Gum","eps","leak"),
        ("Gum","l0","mask"),  ("Gum","l0","leak"),
    ]
    # Roman numerals for the 12 ablation panels (row-major)
    for i, k in enumerate(order, start = 1):
        add_roman_outside(axes[k], roman(i), y = 1.03)

    # ---- Top row numerals (keep axes-relative, above each top-row panel) ----


    # ---- Bottom row numerals (place between rows using FIGURE coords) ----
    # We place them below the first-row x-axis labels, in the whitespace gap.
    top_ref = axes[("Exp","lam","mask")]

    # One knob to tune vertical placement (bigger => lower)
    y_gap = top_ref.get_position().y0 - 0.1



    out_file = out_dir / "fig_ablation.pdf"
    fig.savefig(out_file)
    plt.close(fig)
    print("Wrote", out_file)

# =============================================================================
# PDF 3: Radar + Runtime (NO FIG TITLE)
# =============================================================================
def plot_radar_row(fig, subspec, df: pd.DataFrame):
    axes = [fig.add_subplot(subspec[0, i], polar=True) for i in range(5)]

    metrics_labels = [r"$|M|/|I|$", r"$\mathcal{L}$", "Mem", "T\n(ms)", "Paths"]
    n = len(metrics_labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        ddf = df[df["dataset"] == dataset].copy()
        if ddf.empty:
            ax.set_axis_off()
            continue

        agg = ddf.groupby("method").agg(
            dr=("deletion_ratio", "mean"),
            leak=("leakage", "mean"),
            mem=("memory_kb", "mean"),
            time=("time_ms", "mean"),
            paths=("total_paths", "mean"),
        ).reindex(METHOD_ORDER)

        mat = agg.to_numpy(dtype=float)
        mins = np.nanmin(mat, axis=0)
        maxs = np.nanmax(mat, axis=0)
        denom = np.where((maxs - mins) < 1e-9, 1.0, (maxs - mins))
        norm = np.nan_to_num((mat - mins) / denom, nan=0.0)

        for mi, method in enumerate(METHOD_ORDER):
            vals = norm[mi, :].tolist() + [norm[mi, 0]]
            ax.plot(angles, vals, color=METHOD_COLORS_PDF3[method], lw=1.2, linestyle=LINESTYLES[method])
            ax.fill(angles, vals, color=METHOD_COLORS_PDF3[method], alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize = FS)
        ax.tick_params(axis = "x", pad = 4)  # pushes labels outward from the circle

        ax.set_yticklabels([])
        # ax.set_title(dataset, y=1.12, fontsize=10, fontweight="bold")
        # ax.set_title(dataset, y=1.12, fontsize=10, fontweight="bold")
        ax.set_title(dataset, fontsize=FS, fontweight="bold", pad=6)


    return axes



PHASE_COLORS = {"Instantiation": "#3498db", "Modeling": "#e74c3c", "Update Masks": "#9b59b6"}
# PDF 3 ONLY: distinct palettes (different from other PDFs)
METHOD_COLORS_PDF3 = {"DelMin": "#5B8FF9", "Del2Ph": "#5AD8A6", "DelGum": "#F6BD16"}
PHASE_COLORS_PDF3  = {"Instantiation": "#6F6F6F", "Modeling": "#A0A0A0", "Update Masks": "#3F3F3F"}
PHASE_HATCH_PDF3   = {"Instantiation": "///", "Modeling": "\\\\", "Update Masks": "..."}

def plot_runtime_row(fig, subspec, df: pd.DataFrame):
    axes = [fig.add_subplot(subspec[0, i]) for i in range(5)]
    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        ddf = df[df["dataset"] == dataset].copy()
        if ddf.empty:
            ax.set_axis_off()
            continue

        s = ddf.groupby("method").agg(
            instantiation=("init_time_ms", "mean"),
            modeling=("model_time_ms", "mean"),
            null=("update_time_ms", "mean"),
            inst_cells=("num_instantiated_cells", "mean"),
            mask_size=("mask_size", "mean"),
        ).reindex(METHOD_ORDER)

        x = np.arange(len(METHOD_ORDER))
        w = 0.25
        ax.grid(False)
        bottom = np.zeros(len(METHOD_ORDER), dtype=float)

        phase_map = [
            ("instantiation", "Instantiation"),
            ("modeling", "Modeling"),
            ("null", "Update Masks"),
        ]

        for key, label in phase_map:
            vals = s[key].to_numpy(dtype=float)
            ax.bar(
                x - w, vals,
                width=w, bottom=bottom,
                color=PHASE_COLORS_PDF3[label],
                hatch=PHASE_HATCH_PDF3[label],
                edgecolor="black", linewidth=0.3,
            )
            bottom += vals

        ax.set_xticks(x - w)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=FS)
        ax.set_ylabel("Time (ms)" if i == 0 else "")
        _style_ticks(ax)

        ax2 = ax.twinx()
        ax2.grid(False)

        inst_cells = s["inst_cells"].to_numpy(dtype=float)
        mask_size  = s["mask_size"].to_numpy(dtype=float)

        ax2.bar(x + w, inst_cells, width=w, color=ZONE_COLOR_LIGHT, edgecolor="black", linewidth=0.3)
        ax2.bar(x + w, mask_size,  width=w, color=ZONE_COLOR_DARK,  edgecolor="black", linewidth=0.3)
        ax2.set_ylabel("Cells" if i == 4 else "")
        ax2.tick_params(axis="y", which="both", direction="out", length=3.0, width=0.8)

    return axes


def legend_methods_and_runtime():
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS_PDF3[m], lw=1.5, linestyle=LINESTYLES[m], label=METHOD_LABEL[m])
        for m in METHOD_ORDER
    ]
    phase_handles = [
        Patch(facecolor = PHASE_COLORS_PDF3["Instantiation"], edgecolor = "black", linewidth = 0.3,
              hatch = PHASE_HATCH_PDF3["Instantiation"], label = "Instantiation"),
        Patch(facecolor = PHASE_COLORS_PDF3["Modeling"], edgecolor = "black", linewidth = 0.3,
              hatch = PHASE_HATCH_PDF3["Modeling"], label = "Modeling"),
        Patch(facecolor = PHASE_COLORS_PDF3["Update Masks"], edgecolor = "black", linewidth = 0.3,
              hatch = PHASE_HATCH_PDF3["Update Masks"], label = "Update Masks"),
    ]

    zone_handles =  [
        Patch(facecolor=ZONE_COLOR_LIGHT, edgecolor="black", linewidth=0.3, label="Instantiated Cells"),
        Patch(facecolor=ZONE_COLOR_DARK,  edgecolor="black", linewidth=0.3, label="Mask Size"),
    ]
    sep1 = Line2D([0], [0], color="none", label=" ")
    sep2 = Line2D([0], [0], color="none", label=" ")
    handles = method_handles + [sep1] + phase_handles + [sep2] + zone_handles
    labels = [h.get_label() for h in handles]
    return handles, labels

def build_radar_runtime_pdf(df: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(16.5, 7.4))

    gs_outer = fig.add_gridspec(
        3, 1,
        height_ratios = [0.14, 1.35, 1.75],  # was ~1.65
        hspace = 0.22,  # was ~0.18
        left = 0.03, right = 0.995,
        top = 0.965, bottom = 0.09
    )

    handles, labels = legend_methods_and_runtime()
    _legend_row(
        fig, gs_outer[0, 0],
        handles, labels,
        ncol=12,
        handlelength=1.7,
        columnspacing=0.95
    )

    radar_spec = gs_outer[1, 0].subgridspec(1, 5, wspace=0.42)
    runtime_spec = gs_outer[2, 0].subgridspec(1, 5, wspace=0.32)

    radar_axes = plot_radar_row(fig, radar_spec, df)
    runtime_axes = plot_runtime_row(fig, runtime_spec, df)

    # Vertical separators between datasets (PDF 3 only)
    y0 = min(a.get_position().y0 for a in runtime_axes)
    y1 = max(a.get_position().y1 for a in radar_axes)
    for i in range(4):
        left = radar_axes[i].get_position().x1
        right = radar_axes[i+1].get_position().x0
        x = 0.5 * (left + right)
        fig.add_artist(Line2D([x, x], [y0, y1], transform=fig.transFigure,
                              color="black", lw=0.8, alpha=0.8))



    out_file = out_dir / "fig_combined_radar_runtime.pdf"
    fig.savefig(out_file)
    plt.close(fig)
    print("Wrote", out_file)

# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".", help="Directory containing jan26_*.csv main result files")
    ap.add_argument("--ablation_dir", default="./ablation", help="Directory containing ablation sweep CSVs")
    ap.add_argument("--out_dir", default=".", help="Output directory for PDFs")
    ap.add_argument("--delmin_csv", default="jan26_min.csv")
    ap.add_argument("--del2ph_csv", default="jan26_del2ph.csv")
    ap.add_argument("--delgum_csv", default="jan26_gum.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_main_results(data_dir, args.delmin_csv, args.del2ph_csv, args.delgum_csv)

    build_ratio_pdf(df, out_dir)
    build_ablation_pdf(df, ablation_dir, out_dir)
    build_radar_runtime_pdf(df, out_dir)

if __name__ == "__main__":
    main()
