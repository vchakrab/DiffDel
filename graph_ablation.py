#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not available. Install with: pip install matplotlib")

from matplotlib.lines import Line2D

# =========================
# USER SETTINGS
# =========================
INPUT_FILE_DELGUM = "ablation_delgum_epsilon_v1.csv"
INPUT_FILE_DELEXP = "ablation_delexp_epsilon_v1.csv"
OUTPUT_FIG = "ablation_epsilon_delgum_vs_delexp.pdf"

X_COL = "epsilon"
EPSILON_CONST = 0.5  # shown in legend only

DATASETS_IN_ORDER = ["airport", "hospital", "ncvoter", "onlineretail", "adult", "tax"]
LINES_PER_DATASET_PER_HEADERBLOCK = 50

USE_LOWESS = True
LOWESS_FRAC = 0.44
LOWESS_IT = 0

Z_95 = 1.96

COLOR_MASK = "tab:blue"
COLOR_LEAK = "tab:red"
COLOR_UTIL = "tab:green"

LINESTYLE_DELGUM = "-"
LINESTYLE_DELEXP = "--"

ALPHA_DELGUM = 0.16
ALPHA_DELEXP = 0.10

FORCE_XLIM = True
XLIM_PAD_FRAC = 0.06
MIN_X_SPAN_FRAC = 0.02
FALLBACK_MIN_X_SPAN = 0.1

# =========================
# STYLE (matches your other script)  :contentReference[oaicite:0]{index=0}
# =========================
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.variant": "small-caps",
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": 12,
})

# Pretty dataset titles
DISPLAY_DS = {
    "airport": "Airport",
    "hospital": "Hospital",
    "ncvoter": "NCVoter",
    "onlineretail": "OnlineRetail",
    "adult": "Adult",
    "tax": "Tax",
}

def norm_ds(ds: str) -> str:
    ds = (ds or "").strip().lower()
    if ds in ("online_retail", "onlineretail"):
        return "onlineretail"
    return ds

def ds_title(ds: str) -> str:
    ds = norm_ds(ds)
    return DISPLAY_DS.get(ds, ds.strip().title())


# =========================
# LOWESS
# =========================
def lowess_smooth(x, y, frac=0.22, it=0):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if (not USE_LOWESS) or len(x) < 4:
        return y
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(y, x, frac=frac, it=it, return_sorted=True)
        return np.interp(x, sm[:, 0], sm[:, 1])
    except Exception:
        return y


def mean_ci(values, z=Z_95):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    m = float(arr.mean())
    if len(arr) == 1:
        return (m, m, m)
    s = float(arr.std(ddof=1))
    sem = s / math.sqrt(len(arr))
    return (m, m - z * sem, m + z * sem)


# =========================
# Parsing
# =========================
def parse_header_plus_600lines(path: str, x_col: str):
    """
    File format: repeated CSV header lines like:
      lambda,leakage,utility,mask_size
    Between headers there are rows; datasets cycle in DATASETS_IN_ORDER,
    with LINES_PER_DATASET_PER_HEADERBLOCK rows per dataset.
    """
    rows = []
    current_idx = None
    x_col = (x_col or "").strip().lower()

    if not os.path.exists(path):
        raise RuntimeError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue

            first = (r[0] or "").strip().lower()

            # header line => reset block index
            if first == x_col or first.startswith(x_col + ",") or first in ("lambda", "epsilon"):
                current_idx = 0
                continue

            if current_idx is None or len(r) < 4:
                continue

            try:
                x = float(r[0])
                leak = float(r[1])
                util = float(r[2])
                mask = float(r[3])
            except Exception:
                continue

            ds = DATASETS_IN_ORDER[
                (current_idx // LINES_PER_DATASET_PER_HEADERBLOCK) % len(DATASETS_IN_ORDER)
            ]
            current_idx += 1

            rows.append({
                "dataset": norm_ds(ds),
                "x": x,
                "leakage": leak,
                "utility": util,
                "mask_size": mask,
            })

    if not rows:
        raise RuntimeError(f"Parsed 0 numeric rows from {path}. Check file format.")
    return rows


def aggregate_by_dataset_x(rows):
    buckets = defaultdict(lambda: {"leak": [], "util": [], "mask": []})
    for r in rows:
        k = (r["dataset"], r["x"])
        buckets[k]["leak"].append(r["leakage"])
        buckets[k]["util"].append(r["utility"])
        buckets[k]["mask"].append(r["mask_size"])

    agg = defaultdict(list)
    for (ds, x), v in buckets.items():
        lm, llo, lhi = mean_ci(v["leak"])
        um, ulo, uhi = mean_ci(v["util"])
        mm, mlo, mhi = mean_ci(v["mask"])
        agg[ds].append({
            "x": x,
            "leak_m": lm, "leak_lo": llo, "leak_hi": lhi,
            "util_m": um, "util_lo": ulo, "util_hi": uhi,
            "mask_m": mm, "mask_lo": mlo, "mask_hi": mhi,
        })

    for ds in agg:
        agg[ds].sort(key=lambda d: d["x"])
    return agg


def compute_xlim(xs):
    xs = np.asarray(xs, dtype=float)
    xmin, xmax = float(xs.min()), float(xs.max())
    if xmax > xmin:
        pad = XLIM_PAD_FRAC * (xmax - xmin)
        return xmin - pad, xmax + pad
    eps = max(abs(xmin) * MIN_X_SPAN_FRAC, FALLBACK_MIN_X_SPAN)
    return xmin - eps, xmax + eps


# =========================
# Plot
# =========================
def plot_all(agg_g, agg_e):
    # tighter figure to reduce whitespace; tweak to taste
    fig, axes = plt.subplots(
        1, len(DATASETS_IN_ORDER),
        figsize=(28, 4.6),
        sharex=False
    )
    fig.patch.set_facecolor("white")

    # reduce gaps between panels
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.16, top=0.82, wspace=0.35)

    for ax, ds in zip(np.atleast_1d(axes), DATASETS_IN_ORDER):
        ax.set_title(ds_title(ds), fontsize=12)
        ax.set_xlabel(X_COL, fontsize=11)

        ax_mask = ax
        ax_ru = ax.twinx()

        # plot both mechanisms
        for agg, ls, alpha in [
            (agg_g, LINESTYLE_DELGUM, ALPHA_DELGUM),
            (agg_e, LINESTYLE_DELEXP, ALPHA_DELEXP),
        ]:
            pts = agg.get(norm_ds(ds), [])
            if not pts:
                continue

            x = np.array([p["x"] for p in pts], dtype=float)

            # smooth mean and CI bounds (same x grid)
            mm  = lowess_smooth(x, [p["mask_m"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)
            mlo = lowess_smooth(x, [p["mask_lo"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)
            mhi = lowess_smooth(x, [p["mask_hi"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)

            lm  = lowess_smooth(x, [p["leak_m"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)
            llo = lowess_smooth(x, [p["leak_lo"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)
            lhi = lowess_smooth(x, [p["leak_hi"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)

            um  = lowess_smooth(x, [p["util_m"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)
            ulo = lowess_smooth(x, [p["util_lo"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)
            uhi = lowess_smooth(x, [p["util_hi"] for p in pts], frac=LOWESS_FRAC, it=LOWESS_IT)

            ax_mask.plot(x, mm, color=COLOR_MASK, ls=ls, lw=1.6)
            ax_mask.fill_between(x, mlo, mhi, color=COLOR_MASK, alpha=alpha)

            ax_ru.plot(x, lm, color=COLOR_LEAK, ls=ls, lw=1.6)
            ax_ru.fill_between(x, llo, lhi, color=COLOR_LEAK, alpha=alpha * 0.8)

            ax_ru.plot(x, um, color=COLOR_UTIL, ls=ls, lw=1.6)
            ax_ru.fill_between(x, ulo, uhi, color=COLOR_UTIL, alpha=alpha * 0.8)

            if FORCE_XLIM and len(x) > 0:
                ax_mask.set_xlim(*compute_xlim(x))

        ax_mask.set_ylabel("mask_size", fontsize=11)
        ax_ru.set_ylabel("leakage / utility", fontsize=11)

        ax_mask.tick_params(labelsize=9)
        ax_ru.tick_params(labelsize=9)

    # -------- GLOBAL TOP LEGEND (actually at the top) --------
    legend_items = [
        Line2D([0], [0], color=COLOR_MASK, lw=2, label="Mask size"),
        Line2D([0], [0], color=COLOR_LEAK, lw=2, label="Leakage"),
        Line2D([0], [0], color=COLOR_UTIL, lw=2, label="Utility"),
        Line2D([0], [0], color="black", lw=2, ls=LINESTYLE_DELGUM, label="DelGum"),
        Line2D([0], [0], color="black", lw=2, ls=LINESTYLE_DELEXP, label="DelExp"),
        Line2D([], [], color="none", label=f"λ = {EPSILON_CONST}"),
    ]

    # Place in figure coords; doesn't get pushed to the side by axes/tight bbox
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        bbox_transform=fig.transFigure,
        ncol=6,
        frameon=False,
        fontsize=12
    )

    # Save with minimal outer padding (trim whitespace)
    fig.savefig(OUTPUT_FIG, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {OUTPUT_FIG}")


def main():
    rows_g = parse_header_plus_600lines(INPUT_FILE_DELGUM, x_col=X_COL)
    rows_e = parse_header_plus_600lines(INPUT_FILE_DELEXP, x_col=X_COL)

    agg_g = aggregate_by_dataset_x(rows_g)
    agg_e = aggregate_by_dataset_x(rows_e)

    plot_all(agg_g, agg_e)


if __name__ == "__main__":
    main()
