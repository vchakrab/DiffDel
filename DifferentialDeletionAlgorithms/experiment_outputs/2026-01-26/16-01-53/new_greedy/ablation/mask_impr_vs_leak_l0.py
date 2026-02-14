#!/usr/bin/env python3
"""
CORRECTED: Mask-improvement (%) vs Leakage (%) per dataset (Del2Ph L0 sweep)

What changed vs the previous version:
- We DO NOT connect points in L0 order (which causes “triangles” when the tradeoff isn’t monotone in L0).
- Instead, we connect points in increasing X (mask improvement) order to look like a frontier curve.
- We still color segments light→dark by the corresponding (probability-weighted) L0 value.
- If multiple L0 values land on the same plotted point (after rounding), we compute E[L0 | point]
  with p(L0|point)=count/total, so the color is deterministic (order-independent).

Inputs:
  --l0_dir with files: del2ph_L0_<value>.csv
Expected CSV columns: dataset, mask_size, leakage

Output:
  A single PDF with 5 side-by-side panels.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

K_SIZE = {
    "airport": 5,
    "hospital": 9,
    "adult": 9,
    "flight": 11,
    "tax": 3,
}

PANEL_ORDER = ["airport", "hospital", "adult", "flight", "tax"]


def parse_l0_from_name(name: str) -> float | None:
    m = re.search(r"del2ph_L0_([0-9]*\.?[0-9]+)\.csv$", name)
    return float(m.group(1)) if m else None


def leakage_to_percent(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    return arr * 100.0 if np.nanmax(finite) <= 1.5 else arr


def load_l0_points(l0_dir: Path) -> pd.DataFrame:
    files = sorted(l0_dir.glob("del2ph_L0_*.csv"))
    pairs = []
    for f in files:
        l0 = parse_l0_from_name(f.name)
        if l0 is not None:
            pairs.append((l0, f))
    if not pairs:
        raise FileNotFoundError(f"No files matched del2ph_L0_*.csv in {l0_dir}")

    pairs.sort(key=lambda t: t[0])

    rows = []
    for l0, f in pairs:
        df = pd.read_csv(f)
        df.columns = [str(c).strip().lower() for c in df.columns]
        need = {"dataset", "mask_size", "leakage"}
        if not need.issubset(df.columns):
            raise ValueError(f"{f.name} missing columns. Need {need}, got {set(df.columns)}")

        df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
        df["mask_size"] = pd.to_numeric(df["mask_size"], errors="coerce")
        df["leakage"] = pd.to_numeric(df["leakage"], errors="coerce")

        for ds, base in K_SIZE.items():
            sub = df[df["dataset"] == ds]
            if sub.empty:
                continue
            mask_mean = float(sub["mask_size"].mean())
            leak_mean = float(sub["leakage"].mean())

            mask_impr = 100.0 * (1.0 - (mask_mean / float(base)))
            rows.append(
                {
                    "dataset": ds,
                    "L0": float(l0),
                    "mask_improvement_pct": float(mask_impr),
                    "leakage_raw": float(leak_mean),
                }
            )

    out = pd.DataFrame(rows)
    out["leakage_pct"] = leakage_to_percent(out["leakage_raw"].to_numpy(dtype=float))
    return out


def expected_l0_per_point(df_ds: pd.DataFrame, round_decimals: int) -> pd.DataFrame:
    """
    Bucket points by (rounded x, rounded y).
    If multiple L0 map to the same bucket, compute E[L0 | bucket] using count-based probabilities.
    """
    df = df_ds.copy()
    df["_xbin"] = df["mask_improvement_pct"].round(round_decimals)
    df["_ybin"] = df["leakage_pct"].round(round_decimals)

    c = df.groupby(["_xbin", "_ybin", "L0"]).size().reset_index(name="cnt")
    tot = c.groupby(["_xbin", "_ybin"])["cnt"].sum().reset_index(name="tot")
    c = c.merge(tot, on=["_xbin", "_ybin"], how="left")
    c["p"] = c["cnt"] / c["tot"].clip(lower=1)
    c["pL0"] = c["p"] * c["L0"]
    exp_l0 = c.groupby(["_xbin", "_ybin"])["pL0"].sum().reset_index(name="l0_expect")

    df = df.merge(exp_l0, on=["_xbin", "_ybin"], how="left")
    return df


def draw_colored_path(ax, x, y, l0_color, linewidth=7.5, outline=1.8, cmap="Greys"):
    """
    Thick polyline with per-segment colors + a black outline.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    l0_color = np.asarray(l0_color, dtype=float)

    if len(x) < 2:
        ax.scatter(x, y, s=28, facecolor="white", edgecolor="black", linewidth=0.8, zorder=3)
        return None

    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    # Per-segment color value = mean of endpoints
    seg_l0 = 0.5 * (l0_color[:-1] + l0_color[1:])
    norm = Normalize(vmin=np.nanmin(l0_color), vmax=np.nanmax(l0_color))

    # Outline underlay
    under = LineCollection(segs, colors="black", linewidths=linewidth + outline, alpha=1.0, zorder=1)
    ax.add_collection(under)

    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=linewidth, zorder=2)
    lc.set_array(seg_l0)
    ax.add_collection(lc)

    ax.scatter(x, y, s=28, facecolor="white", edgecolor="black", linewidth=0.8, zorder=3)
    return lc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0_dir", required=True)
    ap.add_argument("--out_pdf", default="mask_impr_vs_leak_L0_frontier_colored.pdf")
    ap.add_argument("--round_decimals", type=int, default=3)
    ap.add_argument("--linewidth", type=float, default=8.0)
    ap.add_argument("--outline", type=float, default=2.0)
    ap.add_argument("--cmap", default="Greys")
    ap.add_argument(
        "--connect_by",
        choices=["mask_improvement", "leakage"],
        default="mask_improvement",
        help="How to order points when drawing the path (frontier-like).",
    )
    args = ap.parse_args()

    l0_dir = Path(args.l0_dir).expanduser().resolve()
    df = load_l0_points(l0_dir)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    last_lc = None

    for ax, ds in zip(axes, PANEL_ORDER):
        d = df[df["dataset"] == ds].copy()
        if d.empty:
            ax.set_axis_off()
            continue

        d = expected_l0_per_point(d, round_decimals=args.round_decimals)

        # >>> KEY FIX: connect points by a frontier coordinate, NOT by L0
        if args.connect_by == "mask_improvement":
            d = d.sort_values(["mask_improvement_pct", "leakage_pct"], ascending=[True, True])
        else:
            d = d.sort_values(["leakage_pct", "mask_improvement_pct"], ascending=[True, True])

        x = d["mask_improvement_pct"].to_numpy(dtype=float)
        y = d["leakage_pct"].to_numpy(dtype=float)
        l0c = d["l0_expect"].to_numpy(dtype=float)

        last_lc = draw_colored_path(
            ax, x, y, l0c,
            linewidth=args.linewidth,
            outline=args.outline,
            cmap=args.cmap,
        )

        ax.set_title(ds.capitalize())
        ax.set_xlabel("Mask improvement (%)")
        if ds == "airport":
            ax.set_ylabel("Leakage (%)")
        ax.grid(True, alpha=0.25)

    if last_lc is not None:
        cbar = fig.colorbar(last_lc, ax=axes, location="right", fraction=0.028, pad=0.02)
        cbar.set_label("L0 (lighter → lower, darker → higher)")

    plt.tight_layout()
    fig.savefig(args.out_pdf)
    print("Wrote:", args.out_pdf)


if __name__ == "__main__":
    main()
