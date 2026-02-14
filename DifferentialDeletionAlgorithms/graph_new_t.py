#!/usr/bin/env python3
"""
graph_new_t.py

Creates a VLDB-style combined ablation plot that "merges" the two L0 panels:
  - Leakage(%) vs L0 as a line with 95% CI band
  - Marker dots at each L0 sized by mean mask size at that L0

Inputs:
  --ablation_dir  Directory containing ablation CSVs (filenames include L0)
Optional:
  --out_dir       Output directory (default: .)
  --mech          Which mechanism to plot: Exp, Gum, or Both (default: Exp)
  --title         Optional title (default: none)
  --logy          Use log scale on y (default: off)

Assumptions:
  - Each CSV contains columns: dataset, mask_size, leakage (common in your setup)
  - L0 value is encoded in filename like "...L0_0.2..." (robust parsing included)
  - Mechanism is inferred from filename: gum/gumbel -> Gum else Exp
"""

import os
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Style (match your paper style)
# -----------------------------
FS = 12
plt.rcParams.update({
    "font.family": "serif",
    "font.size": FS,

    "axes.labelsize": FS,
    "axes.titlesize": FS,

    "legend.fontsize": FS,
    "legend.title_fontsize": FS,

    "xtick.labelsize": FS,
    "ytick.labelsize": FS,

    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelpad": 2.0,
})
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

Z_95 = 1.96

# You can adjust to match your exact dataset naming
DATASET_ORDER = ["Airport", "Hospital", "Adult", "Flight", "Tax"]
DATASET_COLORS = {
    "Airport": "#1f77b4",
    "Hospital": "#ff7f0e",
    "Adult": "#2ca02c",
    "Flight": "#d62728",
    "Tax": "#9467bd",
}


def _style_ticks(ax):
    ax.tick_params(axis="both", which="both", direction="out", length=3.0, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# -----------------------------
# Robust filename parsing
# -----------------------------
def parse_l0_from_filename(fname: str) -> Optional[float]:
    """
    Robustly extract L0 from filenames.

    Works for:
      del2ph_L0_0.2.csv
      something_L0_0.2.someTag.csv
      weird_L0_0.2..csv

    Avoids capturing trailing '.' which caused your float("0.2.") crash.
    """
    m = re.search(r"(?:^|[_-])L0[_=]([0-9]+(?:\.[0-9]+)?)", fname)
    if not m:
        return None
    return float(m.group(1))


def infer_mechanism_from_filename(fname: str) -> str:
    """
    Infer mechanism from filename. Matches your ablation loader behavior:
      - If 'gum' or 'gumbel' is in the filename -> Gum
      - Else -> Exp
    """
    s = fname.lower()
    if "gum" in s or "gumbel" in s:
        return "Gum"
    return "Exp"


def normalize_dataset_name(x: str) -> str:
    """
    Normalize dataset strings so they match your plotting scheme.
    """
    s = str(x).strip()
    if not s:
        return s
    s = s.lower()
    # Basic normalization: capitalize first letter
    s = s.capitalize()

    # Optional: map aliases if your CSVs use different names
    aliases = {
        "Ncvoter": "Ncvoter",
        "Onlineretail": "Onlineretail",
    }
    return aliases.get(s, s)


# -----------------------------
# Loading + stats
# -----------------------------
def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except Exception:
        # fallback
        return pd.read_csv(path, header=None)


def load_l0_rows(ablation_dir: Path) -> pd.DataFrame:
    """
    Walks ablation_dir, reads all CSVs that contain an L0 in the filename,
    adds columns: L0, mech, __file__.
    """
    rows: List[pd.DataFrame] = []

    for root, _, files in os.walk(ablation_dir):
        for f in files:
            if not f.endswith(".csv"):
                continue

            l0 = parse_l0_from_filename(f)
            if l0 is None:
                continue

            mech = infer_mechanism_from_filename(f)
            path = Path(root) / f
            df = read_csv_safely(path)
            df.columns = [str(c).strip() for c in df.columns]

            # Must have these to be meaningful
            required = {"dataset", "mask_size", "leakage"}
            if not required.issubset(df.columns):
                continue

            df = df.copy()
            df["L0"] = l0
            df["mech"] = mech
            df["__file__"] = f

            # Normalize dataset names
            df["dataset"] = df["dataset"].apply(normalize_dataset_name)

            # Numeric conversion
            for c in ["mask_size", "leakage", "L0"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=["dataset", "mask_size", "leakage", "L0"])
            rows.append(df)

    if not rows:
        raise RuntimeError(f"No usable L0 ablation CSVs found in: {ablation_dir}")

    return pd.concat(rows, ignore_index=True)


def leakage_to_percent(series: pd.Series) -> pd.Series:
    """
    Convert leakage to percent if it looks like a rate in [0,1].
    If values already look like percents, keep them.
    """
    vals = pd.to_numeric(series, errors="coerce")
    mx = np.nanmax(vals.to_numpy(dtype=float)) if len(vals) else np.nan
    if np.isfinite(mx) and mx <= 1.5:
        return 100.0 * vals
    return vals


def compute_l0_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stats per (mech, dataset, L0):
      - leak_mean (%), leak_ci (95%)
      - mask_mean (raw)
    """
    d = df.copy()
    d["leak_pct"] = leakage_to_percent(d["leakage"])

    g = (
        d.groupby(["mech", "dataset", "L0"])
         .agg(
            n=("leak_pct", "count"),
            leak_mean=("leak_pct", "mean"),
            leak_std=("leak_pct", "std"),
            mask_mean=("mask_size", "mean"),
         )
         .reset_index()
         .sort_values(["mech", "dataset", "L0"])
    )
    g["leak_ci"] = Z_95 * (g["leak_std"].fillna(0.0) / np.sqrt(g["n"].clip(lower=1)))
    return g


# -----------------------------
# Plotting: combined L0 plot
# -----------------------------
def plot_l0_leakage_with_maskdots(stats: pd.DataFrame, mech: str, title: Optional[str], logy: bool):
    """
    One plot:
      x = L0
      y = leakage (%)
      line + CI band
      dots sized by mask_mean
    """
    # Filter mechanism
    if mech != "Both":
        s = stats[stats["mech"] == mech].copy()
        mech_styles = {mech: {"ls": "-", "alpha": 1.0}}
    else:
        s = stats.copy()
        mech_styles = {
            "Exp": {"ls": "-",  "alpha": 1.0},
            "Gum": {"ls": "--", "alpha": 1.0},
        }

    if s.empty:
        raise RuntimeError(f"No rows found for mech={mech}. Check filenames / inference.")

    # Marker size scaling based on mask_mean across all datasets (within selected mech(s))
    mmin = float(np.nanmin(s["mask_mean"].to_numpy(dtype=float)))
    mmax = float(np.nanmax(s["mask_mean"].to_numpy(dtype=float)))
    # Tune these if you want bigger/smaller bubbles
    S_MIN, S_MAX = 40.0, 260.0

    def size_from_mask(m):
        # area encoding
        return S_MIN + (S_MAX - S_MIN) * ((m - mmin) / (mmax - mmin + 1e-9))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    # Ensure consistent dataset ordering but include any extras if present
    present = list(s["dataset"].unique())
    ordered = [d for d in DATASET_ORDER if d in present] + [d for d in present if d not in DATASET_ORDER]

    for dataset in ordered:
        for mech_k, sty in mech_styles.items():
            sub = s[(s["dataset"] == dataset) & (s["mech"] == mech_k)].sort_values("L0")
            if sub.empty:
                continue

            x = sub["L0"].to_numpy(dtype=float)
            y = sub["leak_mean"].to_numpy(dtype=float)
            ci = sub["leak_ci"].to_numpy(dtype=float)
            maskm = sub["mask_mean"].to_numpy(dtype=float)

            color = DATASET_COLORS.get(dataset, None)

            # line
            ax.plot(x, y, linestyle=sty["ls"], linewidth=1.8, color=color, alpha=0.95)

            # CI band (like your ablation plots)
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.18, linewidth=0)

            # mask dots (size encodes mask_mean)
            sizes = np.array([size_from_mask(m) for m in maskm], dtype=float)
            ax.scatter(
                x, y,
                s=sizes,
                color=color,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.6,
                label=dataset if mech_k == list(mech_styles.keys())[0] else None,  # one legend entry per dataset
            )

    ax.set_xlabel(r"$L_0$")
    ax.set_ylabel("Leakage (%)")
    if title:
        ax.set_title(title)

    if logy:
        ax.set_yscale("log")

    _style_ticks(ax)

    # Legend: datasets only (like your reference)
    ax.legend(frameon=False, ncol=2, loc="best")

    fig.tight_layout()
    return fig


def plot_l0_leakage_maskbubble_exaggerated(
    stats: pd.DataFrame,
    mech: str,
    title: Optional[str] = None,
    logy: bool = False,
    y_max: float = 45.0,
    s_min: float = 40.0,
    s_max: float = 1400.0,
    power: float = 0.35,
) -> plt.Figure:
    """
    Same x/y as plot_l0_leakage_with_maskdots, but INTENTIONALLY exaggerates
    marker areas to make mask-size changes visually obvious.

    - Per-dataset normalization (so one dataset can't flatten the others)
    - Power-law expansion with power < 1 amplifies small differences
    - Larger (s_max) bubble range

    This is meant to match the "loud" style you referenced.
    """
    # Filter mechanism
    s = stats[stats["mech"] == mech].copy()
    if s.empty:
        raise ValueError(f"No rows found for mech={mech}")

    # Use actual sweep values for ticks (paper-like)
    l0_ticks = sorted(s["L0"].unique().tolist())

    # Ensure consistent dataset ordering
    present = list(s["dataset"].unique())
    ordered = [d for d in DATASET_ORDER if d in present] + [d for d in present if d not in DATASET_ORDER]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for dataset in ordered:
        sub = s[s["dataset"] == dataset].sort_values("L0")
        if sub.empty:
            continue

        x = sub["L0"].to_numpy(dtype=float)
        y = sub["leak_mean"].to_numpy(dtype=float)
        ci = sub["leak_ci"].to_numpy(dtype=float)
        m = sub["mask_mean"].to_numpy(dtype=float)

        color = DATASET_COLORS.get(dataset, None)

        # Mean curve + CI band (ablation-style)
        ax.plot(x, y, linestyle="-", linewidth=2.2, color=color, alpha=0.95)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.18, linewidth=0)

        # --- EXAGGERATED bubble sizing (per dataset) ---
        mmin, mmax = float(np.nanmin(m)), float(np.nanmax(m))
        t = (m - mmin) / (mmax - mmin + 1e-9)  # normalize to [0,1]
        sizes = s_min + (s_max - s_min) * (t ** power)

        ax.scatter(
            x, y,
            s=sizes,
            color=color,
            alpha=0.90,
            edgecolors="white",
            linewidth=0.8,
            label=dataset,
            zorder=3,
        )

    ax.set_xlabel(r"$L_0$")
    ax.set_ylabel("Leakage (%)")
    if title:
        ax.set_title(title)

    ax.set_xticks(l0_ticks)
    ax.set_xticklabels([f"{t:g}" for t in l0_ticks])

    if logy:
        ax.set_yscale("log")
    else:
        # tight default range like your panels; makes trend readable
        ax.set_ylim(0.0, float(y_max))

    _style_ticks(ax)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    return fig

# ============================================================
# L0 vs MASK IMPROVEMENT (%) with CI band, leakage as bubble size
# Baseline = K_SIZE (your constants)
# ============================================================
# ============================================================
# L0 vs MASK IMPROVEMENT (%) with CI band, leakage as bubble size
# Baseline = K_SIZE (your constants)
# Bubble sizes: SMALL overall, but EXAGGERATED DIFFERENCES via gamma
# ============================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Z_95 = 1.96

DATASET_ORDER = ["Airport", "Hospital", "Adult", "Flight", "Tax"]
DATASET_COLORS = {
    "Airport": "#1f77b4",
    "Hospital": "#ff7f0e",
    "Adult": "#2ca02c",
    "Flight": "#d62728",
    "Tax": "#9467bd",
}

# Your fixed baselines
K_SIZE = {
    "Airport": 5,
    "Hospital": 9,
    "Adult": 9,
    "Flight": 11,
    "Tax": 3,
}

def _leakage_to_percent(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return y
    # if leakage is in [0,1], convert to %
    return 100.0 * y if np.nanmax(finite) <= 1.5 else y

def mask_to_improvement(mask_mean: np.ndarray, mask_ci: np.ndarray, base: float):
    """
    Your requested definition:
      improvement(%) = 100 * (1 - mask/base) = 100*(base - mask)/base
    Positive means mask is smaller than baseline.
    CI bounds flip because improvement decreases as mask increases.
    """
    base = float(base) if (base is not None and np.isfinite(base) and base > 0) else np.nan
    if not np.isfinite(base):
        return mask_mean, mask_mean - mask_ci, mask_mean + mask_ci

    mean_imp = 100.0 * (1.0 - (mask_mean / base))

    lower_mask = mask_mean - mask_ci
    upper_mask = mask_mean + mask_ci

    lower_imp = 100.0 * (1.0 - (upper_mask / base))
    upper_imp = 100.0 * (1.0 - (lower_mask / base))

    return mean_imp, lower_imp, upper_imp

def plot_ablation_mask_improvement(
    ablation_dir: Path,
    out_dir: Path,
    sweep_key: str,   # "L0", "lambda", or "epsilon"
    mech: str = "Exp",
    outfile: str = None,
    s_min: float = 25,
    s_span: float = 220,
    gamma: float = 0.25,
):
    import re

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir = Path(ablation_dir)

    def parse_value(fname: str):
        m = re.search(rf"(?:^|[_-]){sweep_key}[_=]([0-9]+(?:\.[0-9]+)?)", fname, re.IGNORECASE)
        return float(m.group(1)) if m else None

    rows = []
    for f in ablation_dir.glob("*.csv"):
        val = parse_value(f.name)
        if val is None:
            continue

        name = f.name.lower()
        this_mech = "Gum" if ("gum" in name or "gumbel" in name) else "Exp"
        if this_mech != mech:
            continue

        df = pd.read_csv(f)
        df.columns = [str(c).strip() for c in df.columns]
        if not {"dataset", "mask_size", "leakage"}.issubset(df.columns):
            continue

        df["dataset"] = df["dataset"].astype(str).str.strip().str.capitalize()
        df["mask_size"] = pd.to_numeric(df["mask_size"], errors="coerce")
        df["leakage"] = pd.to_numeric(df["leakage"], errors="coerce")
        df[sweep_key] = val
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No {sweep_key} ablation CSVs found.")

    df = pd.concat(rows, ignore_index=True).dropna()

    # ---- stats ----
    stats = (
        df.groupby(["dataset", sweep_key])
          .agg(
              n=("mask_size", "count"),
              mask_mean=("mask_size", "mean"),
              mask_std=("mask_size", "std"),
              leak_mean_raw=("leakage", "mean"),
          )
          .reset_index()
          .sort_values(["dataset", sweep_key])
    )

    stats["mask_ci"] = Z_95 * (stats["mask_std"].fillna(0.0) / np.sqrt(stats["n"].clip(lower=1)))

    # ---- normalize leakage globally ----
    gmin = stats["leak_mean_raw"].min()
    gmax = stats["leak_mean_raw"].max()
    stats["leak_norm"] = (stats["leak_mean_raw"] - gmin) / (gmax - gmin + 1e-12)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for dataset in DATASET_ORDER:
        d = stats[stats["dataset"] == dataset]
        if d.empty:
            continue

        x = d[sweep_key].to_numpy()
        mask_mean = d["mask_mean"].to_numpy()
        mask_ci = d["mask_ci"].to_numpy()

        base = K_SIZE.get(dataset, np.nan)
        y_mean, y_lo, y_hi = mask_to_improvement(mask_mean, mask_ci, base)

        t = d["leak_norm"].to_numpy()
        sizes = s_min + s_span * (t ** gamma)

        c = DATASET_COLORS[dataset]

        ax.plot(x, y_mean, color=c, lw=2)
        ax.fill_between(x, y_lo, y_hi, color=c, alpha=0.18)

        ax.scatter(
            x, y_mean,
            s=sizes,
            color=c,
            edgecolors="white",
            linewidth=0.8,
            alpha=0.92,
            label=dataset,
            zorder=3,
        )

    # -------- LOG SCALE FIX --------
    if sweep_key.lower() in ["epsilon", "lam"]:
        ax.set_xscale("log")

        # cleaner tick formatting
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel(sweep_key)
    ax.set_ylabel("Mask Improvement (%)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    if outfile is None:
        outfile = f"fig_{sweep_key.lower()}_mask_improvement_leak.pdf"

    out_path = out_dir / outfile
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("Wrote", out_path)
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import ScalarFormatter

def compute_sweep_stats(ablation_dir: Path, sweep_key: str, mech: str = "Exp") -> pd.DataFrame:
    """
    Builds stats per (dataset, sweep_value):
      mask_mean, mask_ci, leak_mean, leak_ci
    Reads recursively under ablation_dir.

    Expects each CSV has: dataset, mask_size, leakage
    Sweep value parsed from filename using sweep_key aliases.
    """
    # sweep aliases so you can call sweep_key="lambda" or "lam" and still match
    aliases = {
        "l0": ["l0"],
        "epsilon": ["epsilon", "eps"],
        "lambda": ["lambda", "lam"],
        "lam": ["lambda", "lam"],
    }
    keys = aliases.get(sweep_key.lower(), [sweep_key])

    def parse_sweep_value(fname: str) -> Optional[float]:
        for k in keys:
            m = re.search(rf"(?:^|[_-]){k}[_=]([0-9]+(?:\.[0-9]+)?)", fname, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return None

    rows = []
    for root, _, files in os.walk(ablation_dir):
        for f in files:
            if not f.endswith(".csv"):
                continue
            val = parse_sweep_value(f)
            if val is None:
                continue

            this_mech = infer_mechanism_from_filename(f)
            if mech != "Both" and this_mech != mech:
                continue

            path = Path(root) / f
            df = read_csv_safely(path)
            df.columns = [str(c).strip() for c in df.columns]
            if not {"dataset", "mask_size", "leakage"}.issubset(df.columns):
                continue

            df = df.copy()
            df["dataset"] = df["dataset"].apply(normalize_dataset_name)
            df["mask_size"] = pd.to_numeric(df["mask_size"], errors="coerce")
            df["leakage"] = pd.to_numeric(df["leakage"], errors="coerce")
            df["sweep"] = val
            df = df.dropna(subset=["dataset", "mask_size", "leakage", "sweep"])
            rows.append(df)

    if not rows:
        raise RuntimeError(f"No usable CSVs found for sweep_key={sweep_key} in {ablation_dir}")

    d = pd.concat(rows, ignore_index=True)
    d["leak_pct"] = leakage_to_percent(d["leakage"])

    stats = (
        d.groupby(["dataset", "sweep"])
         .agg(
            n=("leak_pct", "count"),
            leak_mean=("leak_pct", "mean"),
            leak_std=("leak_pct", "std"),
            mask_mean=("mask_size", "mean"),
            mask_std=("mask_size", "std"),
         )
         .reset_index()
         .sort_values(["dataset", "sweep"])
    )
    stats["leak_ci"] = Z_95 * (stats["leak_std"].fillna(0.0) / np.sqrt(stats["n"].clip(lower=1)))
    stats["mask_ci"] = Z_95 * (stats["mask_std"].fillna(0.0) / np.sqrt(stats["n"].clip(lower=1)))
    return stats

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import ScalarFormatter

DATASET_LINESTYLES = {
    "Airport": "-",
    "Hospital": "--",
    "Adult": "-.",
    "Flight": ":",
    "Tax": (0, (3, 1, 1, 1)),  # dash-dot-ish custom
}

DATASET_MARKERS = {
    "Airport": "o",
    "Hospital": "s",
    "Adult": "^",
    "Flight": "D",
    "Tax": "P",
}

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def _regions_from_x(x: np.ndarray):
    """
    Given sorted x points, return left/right boundaries for each point's region
    using midpoints between neighbors.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] - 0.5]), np.array([x[0] + 0.5])

    mids = (x[:-1] + x[1:]) / 2.0
    left = np.empty_like(x)
    right = np.empty_like(x)

    left[0] = x[0] - (mids[0] - x[0])
    right[-1] = x[-1] + (x[-1] - mids[-1])

    left[1:] = mids
    right[:-1] = mids
    return left, right


def plot_mask_vs_leak_with_sweep_regions(
    stats: pd.DataFrame,
    sweep_name: str,         # "L0" / "epsilon" / "lambda"
    out_path: Path,
    title: str = None,
    region_alpha: float = 0.10,
    band_alpha: float = 0.20,
    show_region_legend: bool = True,
    max_regions_in_legend: int = 6,   # don’t spam legend if many sweep values
):
    """
    stats must have columns:
      dataset, sweep, mask_mean, leak_mean, leak_ci

    Produces a plot like your example:
      - dataset colored line
      - dataset colored CI band
      - translucent x-axis regions marking sweep values
    """

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # global sweep colormap for region shading
    cmap = plt.get_cmap("viridis")
    sweep_vals = np.sort(stats["sweep"].unique().astype(float))
    # normalize to [0,1] for colormap
    vmin, vmax = float(np.min(sweep_vals)), float(np.max(sweep_vals))
    denom = (vmax - vmin) if vmax > vmin else 1.0

    # plot each dataset
    present = list(stats["dataset"].unique())
    ordered = [d for d in DATASET_ORDER if d in present] + [d for d in present if d not in DATASET_ORDER]

    for dataset in ordered:
        sub = stats[stats["dataset"] == dataset].copy()
        if sub.empty:
            continue

        # Sort by mask_mean for a smooth curve in mask–leak space
        sub = sub.sort_values("mask_mean")

        x = sub["mask_mean"].to_numpy(dtype=float)
        y = sub["leak_mean"].to_numpy(dtype=float)
        ci = sub["leak_ci"].to_numpy(dtype=float)
        s = sub["sweep"].to_numpy(dtype=float)

        color = DATASET_COLORS.get(dataset, None)

        # CI band (dataset color)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=band_alpha, linewidth=0, zorder=1)

        # dataset curve
        ax.plot(x, y, color=color, lw=2.2, alpha=0.95, label=dataset, zorder=3)

        # region shading per point (colored by sweep value)
        left, right = _regions_from_x(x)
        for i in range(len(x)):
            t = (s[i] - vmin) / denom
            ax.axvspan(left[i], right[i], color=cmap(t), alpha=region_alpha, zorder=0)

    ax.set_xlabel("Mask Size")
    ax.set_ylabel("Leakage (%)")
    if title:
        ax.set_title(title)

    _style_ticks(ax)
    ax.legend(frameon=False, ncol=2, loc="best")

    # Optional mini legend for regions (only a few entries)
    if show_region_legend:
        # pick evenly spaced sweep values for legend
        if len(sweep_vals) <= max_regions_in_legend:
            chosen = sweep_vals
        else:
            idx = np.linspace(0, len(sweep_vals) - 1, max_regions_in_legend).round().astype(int)
            chosen = sweep_vals[idx]

        patches = []
        for v in chosen:
            t = (v - vmin) / denom
            patches.append(Patch(facecolor=cmap(t), alpha=region_alpha * 2.2, edgecolor="none", label=f"{v:g}"))
        # Put this legend below/side so it doesn't fight dataset legend
        ax.add_artist(ax.legend(handles=patches, title=sweep_name, frameon=False,
                                loc="upper right", bbox_to_anchor=(1.22, 1.0)))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("Wrote", out_path)
from matplotlib.patches import Patch

def plot_mask_vs_leak_with_binned_regions(
    stats: pd.DataFrame,
    sweep_name: str,          # e.g. r"$L_0$", r"$\epsilon$", r"$\lambda$"
    out_path: Path,
    title: str = None,
    n_bins: int = 4,          # 3–6 is usually best
    band_alpha: float = 0.20,
    region_alpha: float = 0.14,
):
    """
    stats columns: dataset, sweep, mask_mean, leak_mean, leak_ci
    Produces:
      - dataset mean curve + leakage CI band
      - background regions are FEW bins of sweep values (not one per point)
    """

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # --- create bins on sweep values (quantiles) ---
    sweep_vals = stats["sweep"].astype(float).to_numpy()
    uniq = np.unique(sweep_vals)

    if len(uniq) < n_bins:
        n_bins = len(uniq)

    # quantile edges
    edges = np.quantile(uniq, np.linspace(0, 1, n_bins + 1))
    # make edges strictly increasing (avoid duplicates if many same values)
    edges = np.unique(edges)
    if len(edges) < 3:
        # fallback: just no regions if cannot bin
        edges = None

    cmap = plt.get_cmap("viridis")

    # plot datasets
    present = list(stats["dataset"].unique())
    ordered = [d for d in DATASET_ORDER if d in present] + [d for d in present if d not in DATASET_ORDER]

    all_x = []

    for dataset in ordered:
        sub = stats[stats["dataset"] == dataset].copy()
        if sub.empty:
            continue

        # Sort by mask_mean so it reads like your example
        sub = sub.sort_values("mask_mean")

        x = sub["mask_mean"].to_numpy(dtype=float)
        y = sub["leak_mean"].to_numpy(dtype=float)
        ci = sub["leak_ci"].to_numpy(dtype=float)
        s = sub["sweep"].to_numpy(dtype=float)

        all_x.append(x)

        c = DATASET_COLORS.get(dataset, None)

        # CI band (dataset color)
        ax.fill_between(x, y - ci, y + ci, color=c, alpha=band_alpha, linewidth=0, zorder=2)

        # dataset curve
        ax.plot(x, y, color=c, lw=2.2, alpha=0.95, label=dataset, zorder=3)

        # overlay “area markers” as lightly colored points (bin-coded)
        if edges is not None:
            # assign each point to a bin index
            bin_idx = np.digitize(s, edges[1:-1], right=True)  # 0..(nbins-1)
            # color points by bin (not by exact sweep)
            for b in range(len(edges) - 1):
                m = (bin_idx == b)
                if not np.any(m):
                    continue
                ax.scatter(x[m], y[m], s=28, color=cmap(b / max(1, (len(edges)-2))),
                           edgecolors="white", linewidth=0.6, alpha=0.85, zorder=4)

    # --- shade broad background regions using sweep-bin -> x-range mapping ---
    # We need a mapping from sweep bins to mask-size ranges. Best heuristic:
    # gather all points, for each bin find min/max mask_mean.
    if edges is not None:
        # Build a combined table of points for bin->mask range
        tmp = stats.copy()
        tmp["sweep"] = tmp["sweep"].astype(float)
        tmp["mask_mean"] = tmp["mask_mean"].astype(float)
        tmp["bin"] = np.digitize(tmp["sweep"].to_numpy(), edges[1:-1], right=True)

        # For each bin, shade the x-range covered by that bin's points
        patches = []
        for b in sorted(tmp["bin"].unique()):
            d = tmp[tmp["bin"] == b]
            if d.empty:
                continue
            xmin = float(np.min(d["mask_mean"]))
            xmax = float(np.max(d["mask_mean"]))
            # expand slightly so it reads like an "area"
            pad = 0.04 * (xmax - xmin + 1e-9)
            xmin -= pad
            xmax += pad

            col = cmap(b / max(1, (len(edges)-2)))
            ax.axvspan(xmin, xmax, color=col, alpha=region_alpha, zorder=0)

            lo = edges[b]
            hi = edges[b+1]
            patches.append(Patch(facecolor=col, alpha=region_alpha, edgecolor="none",
                                 label=f"{lo:g} – {hi:g}"))

        # Put bin legend separate from dataset legend
        if patches:
            ax.add_artist(ax.legend(handles=patches, title=f"{sweep_name} bins",
                                    frameon=False, loc="upper right",
                                    bbox_to_anchor=(1.28, 1.0)))

    ax.set_xlabel("Mask Size")
    ax.set_ylabel("Leakage (%)")
    if title:
        ax.set_title(title)

    _style_ticks(ax)
    ax.legend(frameon=False, ncol=2, loc="best")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("Wrote", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_dir", required=True, help="Directory containing ablation CSVs")
    ap.add_argument("--out_dir", default=".", help="Output directory for the PDF")
    ap.add_argument("--mech", default="Exp", choices=["Exp", "Gum", "Both"], help="Mechanism to plot")
    ap.add_argument("--title", default="", help="Optional plot title (default: none)")
    ap.add_argument("--logy", action="store_true", help="Use log scale on leakage axis")
    ap.add_argument("--outfile", default="fig_l0_leakage_maskdots.pdf", help="Output PDF filename")
    ap.add_argument("--exaggerated", action="store_true",
                    help="Also write an exaggerated bubble-size variant (paper storytelling).")
    ap.add_argument("--exag_outfile", default="fig_l0_leakage_maskbubble_exaggerated.pdf",
                    help="Output PDF filename for exaggerated variant")
    ap.add_argument("--exag_smax", type=float, default=600.0, help="Max bubble area (exaggerated plot)")
    ap.add_argument("--exag_power", type=float, default=0.35, help="Power <1 exaggerates small mask differences")
    ap.add_argument("--exag_ymax", type=float, default=45.0, help="Y max for exaggerated plot when not logy")
    args = ap.parse_args()

    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_l0_rows(ablation_dir)
    stats = compute_l0_stats(raw)

    fig = plot_l0_leakage_with_maskdots(
        stats=stats,
        mech=args.mech,
        title=args.title.strip() or None,
        logy=args.logy,
    )

    out_path = out_dir / args.outfile
    fig.savefig(out_path)
    plt.close(fig)
    print("Wrote", out_path)

    if args.exaggerated:
        fig2 = plot_l0_leakage_maskbubble_exaggerated(
            stats=stats,
            mech=args.mech if args.mech != "Both" else "Exp",
            title=(args.title.strip() or None),
            logy=args.logy,
            y_max=args.exag_ymax,
            s_max=args.exag_smax,
            power=args.exag_power,
        )
        out_path2 = out_dir / args.exag_outfile
        fig2.savefig(out_path2)
        plt.close(fig2)
        print("Wrote", out_path2)
    plot_ablation_mask_improvement(ablation_dir, out_dir, "L0", mech = "Exp")
    plot_ablation_mask_improvement(ablation_dir, out_dir, "lam", mech = "Exp")
    plot_ablation_mask_improvement(ablation_dir, out_dir, "epsilon", mech = "Exp")
    # ---- Mask vs Leakage colored by sweep "regions" ----
    # for sweep_key, label, logc in [
    #     ("L0", r"$L_0$", False),
    #     ("epsilon", r"$\epsilon$", True),  # often nice to log-color epsilon
    #     ("lambda", r"$\lambda$", True),  # often nice to log-color lambda
    # ]:
    #     st = compute_sweep_stats(ablation_dir, sweep_key = sweep_key,
    #                              mech = "Exp")  # or args.mech if you want
    #     plot_mask_vs_leak_colored_by_sweep(
    #         stats = st,
    #         sweep_label = label,
    #         out_path = out_dir / f"fig_mask_vs_leak_colored_by_{sweep_key.lower()}.pdf",
    #         use_log_color = logc,
    #         title = f"Mask vs Leakage colored by {label}",
    #     )

    # ----- L0 plot -----
    st_l0 = compute_sweep_stats(ablation_dir, sweep_key = "L0", mech = "Exp")
    plot_mask_vs_leak_with_binned_regions(
        stats = st_l0,
        sweep_name = r"$L_0$",
        out_path = out_dir / "fig_mask_vs_leak_L0_regions.pdf",
        title = r"Mask vs Leakage (regions colored by $L_0$)",
        n_bins = 4,
    )

    # ----- lambda plot -----
    st_lam = compute_sweep_stats(ablation_dir, sweep_key = "lambda", mech = "Exp")
    plot_mask_vs_leak_with_binned_regions(
        stats = st_lam,
        sweep_name = r"$\lambda$",
        out_path = out_dir / "fig_mask_vs_leak_lambda_regions.pdf",
        title = r"Mask vs Leakage (regions colored by $\lambda$)",
        n_bins = 4,
    )

    # ----- epsilon plot -----
    st_eps = compute_sweep_stats(ablation_dir, sweep_key = "epsilon", mech = "Exp")
    plot_mask_vs_leak_with_binned_regions(
        stats = st_eps,
        sweep_name = r"$\epsilon$",
        out_path = out_dir / "fig_mask_vs_leak_epsilon_regions.pdf",
        title = r"Mask vs Leakage (regions colored by $\epsilon$)",
        n_bins = 4,
    )


if __name__ == "__main__":
    main()
