#!/usr/bin/env python3
"""
plot_iut_two_deletions_like_plots2.py

- Auto-detect method folders (subdirs containing CSVs) or pass --method_dirs
- Robustly parses ablation var/value from filename (handles 0.8. etc)
- Computes per-run then averages (Expected values):
    mask_impr_pct_run = 100 * |baseline - mask_size| / baseline
    P_min = least_freq/total
    Denom = P_min/(1-P_min)              (constant per dataset)
    Num_run = L_run/(1-L_run)
    Er_run  = max(0, ln(Num_run/Denom))

  Then:
    E_mask_impr = mean(mask_impr_pct_run)
    E_L         = mean(L_run)
    E_Num       = mean(Num_run)
    E_Er        = mean(Er_run)

- Outputs ONE PDF with pages:
    For each METHOD:
      - scatter pages: METHOD × {lam, epsilon, L0}
      - bar pages: METHOD × {lam, epsilon, L0} × {adult, airport, flight, hospital, tax}
      - 1 appendix/info page (counts + Pmin/Denom + E[L]/E[Num]/E[Er])

Run:
  python3 plot_iut_two_deletions_like_plots2.py
Optional:
  python3 plot_iut_two_deletions_like_plots2.py --method_dirs folderA folderB
  python3 plot_iut_two_deletions_like_plots2.py --out_pdf OUT.pdf
  python3 plot_iut_two_deletions_like_plots2.py --leakage_is_percent
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# STYLE (force EVERYTHING to 12pt)
# =============================================================================
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


# -----------------------------------------------------------------------------
# Mask baselines
# -----------------------------------------------------------------------------
BASELINE_MASK: Dict[str, float] = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}

# -----------------------------------------------------------------------------
# Counts -> P_min and Denom
# -----------------------------------------------------------------------------
DATASET_COUNTS: Dict[str, Dict[str, int]] = {
    "hospital": {"total_count": 114_919, "least_frequent_count": 2},
    "tax":      {"total_count": 99_904,  "least_frequent_count": 1},
    "adult":    {"total_count": 32_561,  "least_frequent_count": 49},
    "flight":   {"total_count": 499_308, "least_frequent_count": 137},
    "airport":  {"total_count": 55_100,  "least_frequent_count": 1},
}

DATASET_ORDER = ["adult", "airport", "flight", "hospital", "tax"]
MARKERS = {"adult": "^", "airport": "o", "flight": "D", "hospital": "s", "tax": "P"}
COLORS = {
    "adult": "#1a7f37",
    "airport": "#1f77b4",
    "flight": "#4b2ca3",
    "hospital": "#b11218",
    "tax": "#d65f00",
}

def P_min(ds: str) -> float:
    info = DATASET_COUNTS[ds]
    return float(info["least_frequent_count"]) / float(info["total_count"])

def Denom(ds: str, eps: float = 1e-12) -> float:
    p = P_min(ds)
    p = min(max(p, eps), 1.0 - eps)
    return p / (1.0 - p)

def clamp01(x: float, eps: float = 1e-12) -> float:
    return min(max(float(x), eps), 1.0 - eps)

def num_from_L(L: float) -> float:
    Lc = clamp01(L)
    return Lc / (1.0 - Lc)

def er_from_L(L: float, ds: str) -> float:
    denom = Denom(ds)
    num = num_from_L(L)
    er = math.log(num / denom)
    return max(0.0, er)


# -----------------------------------------------------------------------------
# Robust filename parsing for ablation var/value (fixes 0.8. etc)
# -----------------------------------------------------------------------------
ABL_RE = re.compile(
    r"(?:^|_)(lam|lambda|epsilon|L0)(?:_|-)"
    r"([0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?)"
    r"(?![0-9A-Za-z])",
    re.IGNORECASE,
)

def parse_ablation_from_filename(fname: str) -> Tuple[str, float]:
    stem = Path(fname).stem
    m = ABL_RE.search(stem)
    if not m:
        raise ValueError(f"Could not parse ablation var/value from filename: {fname}")

    var = m.group(1).lower()
    if var == "lambda":
        var = "lam"
    if var == "l0":
        var = "L0"
    elif var == "epsilon":
        var = "epsilon"
    elif var == "lam":
        var = "lam"

    raw = m.group(2).strip().rstrip(".")
    val = float(raw)
    return var, val


def list_method_dirs(parent: Path, explicit: List[str] | None) -> List[Path]:
    if explicit:
        return [parent / d for d in explicit]

    out: List[Path] = []
    for p in sorted(parent.iterdir()):
        if p.is_dir() and p.name != "__MACOSX":
            if any(f.suffix.lower() == ".csv" and not f.name.startswith("._") for f in p.iterdir()):
                out.append(p)
    if not out:
        raise RuntimeError(f"No method folders with CSVs found under: {parent}")
    return out


def read_method(method_dir: Path, leakage_is_percent: bool) -> pd.DataFrame:
    """
    Returns per (method, dataset, ablation, value) expected metrics:
      E_mask_impr_pct, E_L, E_Num, E_Er, plus dataset-constant Pmin/Denom.
    """
    rows = []
    csvs = sorted([p for p in method_dir.glob("*.csv") if not p.name.startswith("._")])
    if not csvs:
        raise RuntimeError(f"No CSVs in {method_dir}")

    for csv_path in csvs:
        ablation, ab_val = parse_ablation_from_filename(csv_path.name)
        df = pd.read_csv(csv_path)

        for c in ("dataset", "mask_size", "leakage"):
            if c not in df.columns:
                raise ValueError(f"{csv_path} missing required column '{c}'")

        # convert leakage if in percent
        leak = df["leakage"].astype(float).to_numpy()
        if leakage_is_percent:
            leak = leak / 100.0

        # compute run-wise derived quantities
        # (we’ll do it per dataset group)
        df2 = df.copy()
        df2["L_run"] = leak

        # run-wise mask improvement
        def _mask_impr_row(row):
            ds = str(row["dataset"])
            if ds not in BASELINE_MASK:
                return np.nan
            b = float(BASELINE_MASK[ds])
            return 100.0 * abs(b - float(row["mask_size"])) / b

        df2["mask_impr_run"] = df2.apply(_mask_impr_row, axis=1)

        # run-wise Num and Er
        def _num_row(row):
            return num_from_L(float(row["L_run"]))

        def _er_row(row):
            ds = str(row["dataset"])
            return er_from_L(float(row["L_run"]), ds)

        df2["Num_run"] = df2.apply(_num_row, axis=1)
        df2["Er_run"]  = df2.apply(_er_row, axis=1)

        # expected values per dataset
        g = df2.groupby("dataset", as_index=False).agg(
            E_mask_impr=("mask_impr_run", "mean"),
            E_L=("L_run", "mean"),
            E_Num=("Num_run", "mean"),
            E_Er=("Er_run", "mean"),
        )

        for _, r in g.iterrows():
            ds = str(r["dataset"])
            if ds not in BASELINE_MASK:
                continue
            if ds not in DATASET_COUNTS:
                raise ValueError(f"Missing counts for dataset '{ds}' in DATASET_COUNTS")

            rows.append({
                "method": method_dir.name,
                "dataset": ds,
                "ablation": ablation,
                "value": float(ab_val),

                "E_mask_impr_pct": float(r["E_mask_impr"]),
                "E_L": float(r["E_L"]),
                "E_Num": float(r["E_Num"]),
                "E_Er": float(r["E_Er"]),

                "Pmin": P_min(ds),
                "Denom": Denom(ds),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No usable rows from {method_dir}")
    return out


# -----------------------------------------------------------------------------
# Plot pages
# -----------------------------------------------------------------------------
def scatter_page(pdf: PdfPages, df: pd.DataFrame, method_name: str, ablation: str) -> None:
    d = df[df["ablation"] == ablation].copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for ds in DATASET_ORDER:
        dd = d[d["dataset"] == ds]
        if dd.empty:
            continue
        ax.scatter(
            dd["E_mask_impr_pct"].values,
            dd["E_Er"].values,
            marker=MARKERS[ds],
            color=COLORS[ds],
            edgecolors="black",
            linewidths=0.6,
            s=120,
            alpha=1.0,
            label=ds,
            zorder=3,
        )

    ax.set_title(f"{method_name} — {ablation} ablation (expected)")
    ax.set_xlabel("Expected mask improvement (%)")
    ax.set_ylabel(r"Expected $E_r$")

    ax.legend(title="Dataset", loc="upper left", frameon=True)
    pdf.savefig(fig)
    plt.close(fig)


def bars_page(pdf: PdfPages, df: pd.DataFrame, method_name: str, ablation: str, dataset: str) -> None:
    d = df[(df["ablation"] == ablation) & (df["dataset"] == dataset)].copy()
    if d.empty:
        return

    d = d.sort_values("value")
    xvals = d["value"].values
    ind = np.arange(len(xvals))
    width = 0.42

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax2 = ax.twinx()

    ax.bar(ind - width/2, d["E_mask_impr_pct"].values, width=width, label="Mask improvement")
    ax2.bar(ind + width/2, d["E_Er"].values, width=width, label=r"Expected $E_r$", color="#d62728")

    ax.set_xticks(ind)
    ax.set_xticklabels([f"{v:g}" for v in xvals])

    ax.set_title(f"{method_name} — {dataset} — {ablation} ablation (expected)")
    ax.set_xlabel(ablation)
    ax.set_ylabel("Expected mask improvement (%)")
    ax2.set_ylabel(r"Expected $E_r$")

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), frameon=True)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.15), frameon=True)

    pdf.savefig(fig)
    plt.close(fig)


def appendix_page(pdf: PdfPages, df: pd.DataFrame, method_name: str) -> None:
    """
    One page at end of each method with:
      - dataset constants: total_count, least_freq_count, Pmin, Denom
      - expected params by ablation: E[L], E[Num], E[E_r]
    """
    fig = plt.figure(figsize=(8.5, 11.0))
    fig.suptitle(f"{method_name} — Appendix (constants + expected parameters)", y=0.995)

    # ---------- Top: dataset constants ----------
    ax1 = fig.add_axes([0.07, 0.72, 0.86, 0.21])
    ax1.axis("off")
    ax1.set_title("Dataset constants")

    const_rows = []
    for ds in DATASET_ORDER:
        info = DATASET_COUNTS[ds]
        pmin = P_min(ds)
        denom = Denom(ds)
        const_rows.append([
            ds,
            f"{info['total_count']}",
            f"{info['least_frequent_count']}",
            f"{pmin:.8g}",
            f"{denom:.8g}",
        ])

    const_cols = ["dataset", "total_count", "least_freq_count", r"$P_{min}$", "Denom"]
    t1 = ax1.table(cellText=const_rows, colLabels=const_cols, loc="center", cellLoc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)
    t1.scale(1.0, 1.3)

    # ---------- Bottom: expected params summary ----------
    ax2 = fig.add_axes([0.07, 0.05, 0.86, 0.62])
    ax2.axis("off")
    ax2.set_title("Expected parameters by ablation")

    d = df.copy().sort_values(["dataset", "ablation", "value"])

    rows = []
    for _, r in d.iterrows():
        rows.append([
            r["dataset"],
            r["ablation"],
            f"{r['value']:g}",
            f"{r['E_L']:.6g}",
            f"{r['E_Num']:.6g}",
            f"{r['E_Er']:.6g}",
        ])

    cols = ["dataset", "ablation", "value", "E[L]", "E[Num]", r"E[$E_r$]"]
    t2 = ax2.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(8.5)
    t2.scale(1.0, 1.15)

    pdf.savefig(fig)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent_dir", type=str, default=".")
    ap.add_argument("--method_dirs", nargs="*", default=None)
    ap.add_argument("--out_pdf", type=str, default="ALL_PLOTS_WITH_APPENDIX.pdf")
    ap.add_argument("--leakage_is_percent", action="store_true")
    args = ap.parse_args()

    parent = Path(args.parent_dir).resolve()
    method_dirs = list_method_dirs(parent, args.method_dirs)

    out_pdf = Path(args.out_pdf).resolve()
    with PdfPages(out_pdf) as pdf:
        for md in method_dirs:
            mdf = read_method(md, leakage_is_percent=args.leakage_is_percent)

            # scatter pages
            for ablation in ["lam", "epsilon", "L0"]:
                scatter_page(pdf, mdf, md.name, ablation)

            # bar pages
            for ablation in ["lam", "epsilon", "L0"]:
                for ds in DATASET_ORDER:
                    bars_page(pdf, mdf, md.name, ablation, ds)

            # appendix page at end of this method
            appendix_page(pdf, mdf, md.name)

    print(f"[ok] wrote {out_pdf}")


if __name__ == "__main__":
    main()
