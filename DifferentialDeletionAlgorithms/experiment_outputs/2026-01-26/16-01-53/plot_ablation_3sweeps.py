#!/usr/bin/env python3
"""
VLDB Ablation Figure — 2 rows × 6 panels (utility dropped), with GROUP DIVIDERS.

Row 1 (Exp):  λ(mask, leak) | ε(mask, leak) | L0(mask, leak)
Row 2 (Gum):  λ(mask, leak) | ε(mask, leak) | L0(mask, leak)

Fixes in this version:
  ✅ GUM lines dashed, EXP lines solid
  ✅ Spacer columns create a REAL gap between groups (λ | ε | L0)
  ✅ Divider lines shifted LEFT inside spacer so they don't collide with next plot's y-label
  ✅ Y-labels pulled closer to axes (smaller/negative labelpad) to avoid divider overlap
  ✅ λ ticks: exponent-only with xlabel λ(10^x)
  ✅ ε ticks: 10^k, powers-of-10 only
  ✅ λ=10000 supported (skips λ=1 only)
  ✅ EXP leakage truncated to 0–50% only
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator

# =============================================================================
# Styling
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.25,
    "lines.linewidth": 1.2,
})

SC_EXP = "Exᴘ"
SC_GUM = "Gᴜᴍ"

DATASETS = ["airport", "hospital", "adult", "flight", "tax"]
SAMPLES_PER_DATASET = 100

DATASET_COLORS = {
    "airport": "#E69F00",
    "hospital": "#56B4E9",
    "adult": "#009E73",
    "flight": "#CC79A7",
    "tax": "#D55E00",
}

DELMIN_MASK_SIZES = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}

CI_Z = 1.96
CI_ALPHA = 0.18

LEAKAGE_YLIM_EXP = (0, 50)
LEAKAGE_YLIM_GUM = None

# Log tick controls: powers-of-10 only
LOG_MAJOR_LOC = LogLocator(base=10)
LOG_FORMAT_10K = LogFormatterMathtext(base=10)

# Label padding (key to avoid divider collisions)
XLABEL_PAD = 2.0
YLABEL_PAD = 4  # pull y-label closer to axis (away from spacer/divider)

# =============================================================================
# Data loading
# =============================================================================

def assign_dataset(n: int):
    return [DATASETS[i // SAMPLES_PER_DATASET] for i in range(n)]

def normalize_leakage_to_pct(s: pd.Series) -> pd.Series:
    """Convert leakage to percent units [0,100] if it looks like fractions [0,1]."""
    s = pd.to_numeric(s, errors="coerce")
    q99 = s.quantile(0.99)
    return (100.0 * s) if (pd.notna(q99) and q99 <= 1.5) else s

def load_sweep(data_dir: Path, param: str, xcol: str) -> Dict[str, pd.DataFrame]:
    rx = re.compile(rf"(del2ph|delgum)_{param}_([0-9eE\.\-]+)\.csv")
    mech_map = {"del2ph": "exp", "delgum": "gum"}
    out = {"exp": [], "gum": []}

    for f in sorted(data_dir.iterdir()):
        m = rx.match(f.name)
        if not m:
            continue

        mech = mech_map[m.group(1)]
        xval = float(m.group(2))

        # Skip ONLY lambda=1
        if param == "lam" and np.isclose(xval, 1.0):
            continue

        df = pd.read_csv(f)

        if "dataset" not in df.columns:
            df["dataset"] = assign_dataset(len(df))

        df[xcol] = xval
        df["mask_size"] = pd.to_numeric(df.get("mask_size"), errors="coerce")
        df["leakage_pct"] = normalize_leakage_to_pct(df.get("leakage", np.nan))

        out[mech].append(df)

    return {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}

# =============================================================================
# Stats (mean + 95% CI)
# =============================================================================

def _mean_sem(s: pd.Series) -> Tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(s.shape[0])
    if n <= 1:
        return (float(s.mean()) if n == 1 else np.nan), np.nan
    m = float(s.mean())
    sem = float(s.std(ddof=1) / np.sqrt(n))
    return m, sem

def compute_stats(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    rows = []
    for (xv, ds), g in df.groupby([xcol, "dataset"]):
        leak_m, leak_sem = _mean_sem(g["leakage_pct"])
        mask_m, mask_sem = _mean_sem(g["mask_size"])

        base = DELMIN_MASK_SIZES.get(ds, np.nan)
        mask_impr_m = 100.0 * (base - mask_m) / base if pd.notna(base) and pd.notna(mask_m) else np.nan
        mask_impr_ci = (100.0 * CI_Z * mask_sem / base) if pd.notna(mask_sem) and pd.notna(base) else np.nan

        rows.append({
            xcol: xv,
            "dataset": ds,
            "mask_impr": mask_impr_m,
            "mask_impr_ci": mask_impr_ci,
            "leakage_pct": leak_m,
            "leakage_pct_ci": (CI_Z * leak_sem) if pd.notna(leak_sem) else np.nan,
        })
    return pd.DataFrame(rows).sort_values([xcol, "dataset"]).reset_index(drop=True)

# =============================================================================
# Plot helpers
# =============================================================================

def panel_tag(ax, letter: str):
    ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7)

def set_log_ticks_powers_only(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LOG_MAJOR_LOC)
    ax.xaxis.set_major_formatter(LOG_FORMAT_10K)
    ax.xaxis.set_minor_locator(NullLocator())

def set_lambda_exponent_ticks(ax, xvals: np.ndarray):
    """For λ: ticks only at data powers-of-10, labels are exponent only."""
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())

    xv = np.array([v for v in xvals if np.isfinite(v) and v > 0], dtype=float)
    if xv.size == 0:
        set_log_ticks_powers_only(ax)
        return

    exp = np.round(np.log10(xv)).astype(int)
    ticks = []
    for k in sorted(set(exp.tolist())):
        t = 10.0 ** k
        if np.any(np.isclose(xv, t, rtol=0, atol=1e-12)):
            ticks.append(t)

    if not ticks:
        set_log_ticks_powers_only(ax)
        return

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(np.log10(t))) for t in ticks])

def plot_lines_with_ci(
    ax,
    st: pd.DataFrame,
    xcol: str,
    ycol: str,
    yci_col: str,
    xlabel: str,
    ylabel: str,
    *,
    is_log: bool = False,
    xlim=None,
    ylim=None,
    y0=None,
    lambda_exponent_style: bool = False,
    linestyle: str = "-",
):
    for d in DATASETS:
        sub = st[st.dataset == d].sort_values(xcol)
        if sub.empty:
            continue
        x = sub[xcol].to_numpy()
        y = sub[ycol].to_numpy()
        yci = sub[yci_col].to_numpy()

        ax.plot(x, y, color=DATASET_COLORS[d], linestyle=linestyle)
        if np.any(np.isfinite(yci)):
            ax.fill_between(x, y - yci, y + yci,
                            alpha=CI_ALPHA, linewidth=0, color=DATASET_COLORS[d])

    if y0 is not None:
        ax.axhline(y0, color="black", linestyle=":", linewidth=0.9)

    ax.set_xlabel(xlabel, labelpad=XLABEL_PAD)
    ax.set_ylabel(ylabel, labelpad=YLABEL_PAD)
    ax.grid(True, alpha=0.22)

    if is_log:
        if lambda_exponent_style:
            set_lambda_exponent_ticks(ax, st[xcol].to_numpy())
        else:
            set_log_ticks_powers_only(ax)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

def add_separator_in_spacer(fig, gs, spacer_col_idx: int, ax_bottom, ax_top, *, lw=0.75):
    """
    Divider in spacer column spanning both rows.
    Placed safely away from the next panel's y-label.
    """
    bb = gs[:, spacer_col_idx].get_position(fig)
    w = (bb.x1 - bb.x0)

    x = bb.x0 + 0.20 * w   # ⬅️ shifted LEFT to avoid ylabel intersection

    y0 = ax_bottom.get_position().y0
    y1 = ax_top.get_position().y1

    fig.add_artist(Line2D(
        [x, x], [y0, y1],
        transform=fig.transFigure,
        color="black",
        linewidth=lw,
        zorder=0,
        solid_capstyle="butt",
    ))

# =============================================================================
# Figure
# =============================================================================

def plot_figure(lam, eps, l0, out: str):
    fig = plt.figure(figsize=(14.0, 4.9))

    # 8 columns: [λM, λL, spacer, εM, εL, spacer, L0M, L0L]
    gs = fig.add_gridspec(
        3, 8,
        height_ratios = [0.055, 1, 1],
        width_ratios = [1, 1, 0.22, 1, 1, 0.22, 1, 1],  # ⬅️ WIDER spacers
        hspace = 0.26,
        wspace = 0.35,
        left = 0.055, right = 0.995,
        top = 0.965, bottom = 0.09
    )

    # Legend row
    axl = fig.add_subplot(gs[0, :])
    axl.axis("off")
    handles = [mpatches.Patch(color=DATASET_COLORS[d], label=d.title()) for d in DATASETS]
    handles += [
        Line2D([0], [0], color="gray", lw=1.2, linestyle="-",  label=SC_EXP),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="--", label=SC_GUM),
    ]
    axl.legend(handles=handles, ncol=7, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.10))

    # Plot columns (skip spacer columns 2 and 5)
    plot_cols = [0, 1, 3, 4, 6, 7]
    axs_exp = [fig.add_subplot(gs[1, j]) for j in plot_cols]
    axs_gum = [fig.add_subplot(gs[2, j]) for j in plot_cols]

    columns = [
        (lam, "lambda",  r"$\lambda\,(10^{x})$", True,  None,      "mask", True),
        (lam, "lambda",  r"$\lambda\,(10^{x})$", True,  None,      "leak", True),

        (eps, "epsilon", r"$\varepsilon$",       True,  (0.1,100), "mask", False),
        (eps, "epsilon", r"$\varepsilon$",       True,  (0.1,100), "leak", False),

        (l0,  "L0",      r"$L_0$",               False, None,      "mask", False),
        (l0,  "L0",      r"$L_0$",               False, None,      "leak", False),
    ]

    letters = [chr(ord("a") + i) for i in range(12)]
    li = 0

    # EXP (solid)
    for j, (stats, xcol, xlabel, is_log, xlim, kind, lam_style) in enumerate(columns):
        st = stats["exp"]
        ax = axs_exp[j]
        if kind == "mask":
            plot_lines_with_ci(
                ax, st, xcol, "mask_impr", "mask_impr_ci",
                xlabel, "Mask size improvement (%)",
                is_log=is_log, xlim=xlim, y0=0, lambda_exponent_style=lam_style,
                linestyle="-"
            )
        else:
            plot_lines_with_ci(
                ax, st, xcol, "leakage_pct", "leakage_pct_ci",
                xlabel, "Leakage (%)",
                is_log=is_log, xlim=xlim, ylim=LEAKAGE_YLIM_EXP, lambda_exponent_style=lam_style,
                linestyle="-"
            )
        panel_tag(ax, letters[li]); li += 1

    # GUM (dashed)
    for j, (stats, xcol, xlabel, is_log, xlim, kind, lam_style) in enumerate(columns):
        st = stats["gum"]
        ax = axs_gum[j]
        if kind == "mask":
            plot_lines_with_ci(
                ax, st, xcol, "mask_impr", "mask_impr_ci",
                xlabel, "Mask size improvement (%)",
                is_log=is_log, xlim=xlim, y0=0, lambda_exponent_style=lam_style,
                linestyle="--"
            )
        else:
            plot_lines_with_ci(
                ax, st, xcol, "leakage_pct", "leakage_pct_ci",
                xlabel, "Leakage (%)",
                is_log=is_log, xlim=xlim, ylim=LEAKAGE_YLIM_GUM, lambda_exponent_style=lam_style,
                linestyle="--"
            )
        panel_tag(ax, letters[li]); li += 1

    # Dividers in spacer columns 2 and 5 (left-shifted within spacer)
    add_separator_in_spacer(fig, gs, 2, ax_bottom=axs_gum[1], ax_top=axs_exp[1], lw=0.75)
    add_separator_in_spacer(fig, gs, 5, ax_bottom=axs_gum[3], ax_top=axs_exp[3], lw=0.75)

    plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved → {out}")

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./ablation")
    ap.add_argument("--output", default="fig_ablation_2x6_mask_leak.pdf")
    args = ap.parse_args()

    d = Path(args.data_dir)

    lam_raw = load_sweep(d, "lam", "lambda")
    eps_raw = load_sweep(d, "epsilon", "epsilon")
    l0_raw  = load_sweep(d, "L0", "L0")

    lam = {k: compute_stats(v, "lambda") for k, v in lam_raw.items()}
    eps = {k: compute_stats(v, "epsilon") for k, v in eps_raw.items()}
    l0  = {k: compute_stats(v, "L0") for k, v in l0_raw.items()}

    plot_figure(lam, eps, l0, args.output)

if __name__ == "__main__":
    main()
