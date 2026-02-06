#!/usr/bin/env python3
"""
Paper-style summarizer (Fig. 6c / Fig. 9 table format) — GUM-safe.

Outputs one row per dataset with columns:
Dataset, #attr, #cells, #DC,
MIN:   T(ms) |M| Mem η L
EXP:   T(ms) |M| Mem η L |Π|
GUM:   T(ms) |M| Mem η L |Π|
T_build(ms)

Definitions:
  total_time = init_time + model_time + del_time
  T(ms)      = total_time * 1000
  η          = (1 - L) / (|M| + 1)

Leakage L printed as mean±95%CI.
Other metrics printed as means.

Key robustness:
- Does NOT require "model_size" to keep a file (so GUM won't get skipped).
- Unifies memory into column "__mem__" from multiple candidates.
- Unifies |Π| from total_paths (and some alternates).
"""

import argparse
import glob
import math
import os
from typing import Dict, Tuple, List, Optional

import pandas as pd

DATASET_ORDER = ["Airport", "Hospital", "Adult", "Flight", "Tax"]

# ----------------------------- Stats helpers -----------------------------

def t_critical(df: int, ci: float = 0.95) -> float:
    alpha = 1.0 - ci
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        return 1.96


def mean_ci(series: pd.Series, ci: float = 0.95) -> Tuple[float, float, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(s.mean())
    if n == 1:
        return m, float("nan"), 1
    sd = float(s.std(ddof=1))
    half = t_critical(n - 1, ci) * sd / math.sqrt(n)
    return m, float(half), n


def fmt_mean(x: float, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}"


def fmt_mean_pm(mean: float, half: float, nd: int = 2) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return ""
    if half is None or (isinstance(half, float) and math.isnan(half)):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f}±{half:.{nd}f}"


def first_present(cols: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


# ----------------------------- Column detection -----------------------------

# Minimal columns required to keep a CSV (GUM-safe: no model_size requirement)
REQUIRED_BASE = ["init_time", "model_time", "del_time", "mask_size", "leakage"]

# Memory candidates (your pipeline might name this differently)
MEM_CANDIDATES = [
    "model_size",
    "memory_overhead_bytes", "mem_bytes", "memory_bytes",
    "num_entries", "n_entries", "entries",
    "model_entries", "num_model_entries",
]

# Dataset characteristic candidates (optional)
ATTR_CANDIDATES = ["n_attr", "num_attr", "num_attrs", "#attr", "attrs", "n_attributes"]
CELLS_CANDIDATES = ["n_cells", "num_cells", "#cells", "cells", "num_rows", "n_rows"]
DC_CANDIDATES = ["n_dc", "num_dc", "num_dcs", "#DC", "dcs", "constraints", "n_constraints"]

# |Π| candidates (you said it's total_paths)
PI_CANDIDATES = ["total_paths", "total_path", "num_paths", "n_paths", "|Π|", "|Pi|", "pi_size", "Pi_size"]

# Template build time candidates (optional)
TBUILD_CANDIDATES = [
    "T_build_ms", "t_build_ms", "Tbuild_ms", "tbuild_ms",
    "T_build", "t_build", "build_time", "template_time", "template_build_time",
    "hypergraph_build_time", "hg_build_time"
]

# Optional hardcoded dataset info (leave empty if present in CSVs)
DATASET_INFO: Dict[str, Dict[str, object]] = {
    # "Airport": {"#attr": 11, "#cells": "495K", "#DC": 7, "T_build(ms)": 0.6},
    # "Hospital": {"#attr": 15, "#cells": "1.7M", "#DC": 40, "T_build(ms)": 1.1},
    # "Adult": {"#attr": 15, "#cells": "488K", "#DC": 57, "T_build(ms)": 2.4},
    # "Flight": {"#attr": 20, "#cells": "10M", "#DC": 112, "T_build(ms)": 6.9},
    # "Tax": {"#attr": 14, "#cells": "1.4M", "#DC": 31, "T_build(ms)": 0.4},
}


def normalize_method(m: str) -> str:
    s = str(m).strip().lower()

    # exact-ish
    if s in {"min", "del", "delmin"}:
        return "MIN"
    if "2ph" in s or "del2ph" in s or "two" in s:
        return "EXP"
    if s in {"gum", "delgum"}:
        return "GUM"

    # substring-based
    if "gum" in s or "gumbel" in s:
        return "GUM"
    if "exp" in s or "expon" in s:
        return "EXP"
    if "min" in s:
        return "MIN"

    # if your MIN is logged as del2ph
    if "2ph" in s or "del2ph" in s or "two" in s:
        return "MIN"

    return s.upper()


def infer_method_from_filename(path: str) -> str:
    base = os.path.basename(path).lower()
    if "gum" in base or "gumbel" in base or "delgum" in base:
        return "GUM"
    if "2ph" in base or "del2ph" in base or "two" in base:
        return "EXP"

    if "min" in base or "delmin" in base:
        return "MIN"



def coerce_str_cells(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        v = float(x)
        if math.isnan(v):
            return ""
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M".rstrip("0").rstrip(".") + "M"
        if v >= 1_000:
            return f"{v/1_000:.0f}K"
        return str(int(v)) if v.is_integer() else f"{v:.0f}"
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="jan26_*.csv",
                    help="Glob pattern for input CSVs (default: jan26_*.csv)")
    ap.add_argument("--out", default="paper_table_summary.csv",
                    help="Output CSV filename (default: paper_table_summary.csv)")
    ap.add_argument("--ci", type=float, default=0.95,
                    help="CI level for leakage (default: 0.95)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    print(f"Pattern: {args.pattern}")
    print(f"Found {len(files)} candidate CSV(s).")
    if not files:
        raise SystemExit("No CSVs matched the pattern. Check filenames/pattern.")

    kept: List[pd.DataFrame] = []
    skipped: List[Tuple[str, str]] = []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            skipped.append((f, f"unreadable: {e}"))
            continue

        if df is None or df.empty:
            skipped.append((f, "empty (0 rows)"))
            continue

        missing = [c for c in REQUIRED_BASE if c not in df.columns]
        if missing:
            skipped.append((f, f"missing required columns: {missing}"))
            continue

        df = df.copy()
        df["__source__"] = os.path.basename(f)

        if "method" not in df.columns:
            df["method"] = infer_method_from_filename(f)
        df["method"] = df["method"].astype(str).map(normalize_method)

        if "dataset" not in df.columns:
            df["dataset"] = "ALL"
        df["dataset"] = df["dataset"].astype(str).str.strip()

        # unify memory into __mem__ (may be NaN if not present)
        mem_col = first_present(MEM_CANDIDATES, df)
        if mem_col:
            df["__mem__"] = pd.to_numeric(df[mem_col], errors="coerce")
        else:
            df["__mem__"] = pd.NA

        # unify paths into __pi__ (may be NaN)
        pi_col = first_present(PI_CANDIDATES, df)
        if pi_col:
            df["__pi__"] = pd.to_numeric(df[pi_col], errors="coerce")
        else:
            df["__pi__"] = pd.NA

        # unify t_build into __tbuild_ms__ (may be NaN)
        tbuild_col = first_present(TBUILD_CANDIDATES, df)
        if tbuild_col:
            tb = pd.to_numeric(df[tbuild_col], errors="coerce")
            if tbuild_col.lower().endswith("_ms") or tbuild_col.lower().endswith("ms"):
                df["__tbuild_ms__"] = tb
            else:
                med = float(tb.dropna().median()) if tb.notna().any() else float("nan")
                df["__tbuild_ms__"] = tb * 1000.0 if (not math.isnan(med) and med < 50.0) else tb
        else:
            df["__tbuild_ms__"] = pd.NA

        kept.append(df)

    print("\n=== Inclusion report ===")
    print(f"Kept: {len(kept)} file(s)")
    for d in kept:
        src = d["__source__"].iloc[0]
        meths = ",".join(sorted(d["method"].dropna().unique().tolist()))
        dsets = ",".join(sorted(d["dataset"].dropna().unique().tolist())[:5])
        print(f"  ✅ {src:45s} rows={len(d):6d} methods={meths:10s} datasets={dsets}")

    print(f"\nSkipped: {len(skipped)} file(s)")
    for f, why in skipped[:80]:
        print(f"  ❌ {os.path.basename(f):45s}  {why}")
    if len(skipped) > 80:
        print(f"  ... and {len(skipped) - 80} more skipped")

    if not kept:
        raise SystemExit("\nNo valid CSVs were kept. Fix pattern / required columns.")

    data = pd.concat(kept, ignore_index=True)

    # ---- total_time (seconds) and T_ms ----
    data["total_time"] = (
        pd.to_numeric(data["init_time"], errors="coerce")
        + pd.to_numeric(data["model_time"], errors="coerce")
        + pd.to_numeric(data["del_time"], errors="coerce")
    )
    data["T_ms"] = pd.to_numeric(data["total_time"], errors="coerce") * 1000.0

    # ---- η from paper definition ----
    data["ETA"] = (1.0 - pd.to_numeric(data["leakage"], errors="coerce")) / (
        pd.to_numeric(data["mask_size"], errors="coerce") + 1.0
    )

    # dataset characteristic detection (optional)
    attr_col = first_present(ATTR_CANDIDATES, data)
    cells_col = first_present(CELLS_CANDIDATES, data)
    dc_col = first_present(DC_CANDIDATES, data)

    present = data["dataset"].dropna().unique().tolist()

    # Keep paper order first, then any extras (just in case)
    datasets = [d for d in DATASET_ORDER if d in present] + \
               [d for d in present if d not in DATASET_ORDER]

    def get_dataset_info(ds: str) -> Tuple[str, str, str]:
        if ds in DATASET_INFO:
            info = DATASET_INFO[ds]
            return (
                str(info.get("#attr", "")),
                str(info.get("#cells", "")),
                str(info.get("#DC", "")),
            )

        sub = data[data["dataset"] == ds]
        a = ""
        c = ""
        d = ""
        if attr_col and sub[attr_col].notna().any():
            a = str(int(pd.to_numeric(sub[attr_col], errors="coerce").dropna().iloc[0]))
        if cells_col and sub[cells_col].notna().any():
            c = coerce_str_cells(sub[cells_col].dropna().iloc[0])
        if dc_col and sub[dc_col].notna().any():
            d = str(int(pd.to_numeric(sub[dc_col], errors="coerce").dropna().iloc[0]))
        return a, c, d

    def summarize_block(sub: pd.DataFrame) -> Dict[str, str]:
        out: Dict[str, str] = {}
        m_T, _, _ = mean_ci(sub["T_ms"], ci=args.ci)
        m_M, _, _ = mean_ci(sub["mask_size"], ci=args.ci)
        m_mem, _, _ = mean_ci(sub["__mem__"], ci=args.ci)
        m_eta, _, _ = mean_ci(sub["ETA"], ci=args.ci)
        m_L, h_L, _ = mean_ci(sub["leakage"], ci=args.ci)
        m_pi, _, n_pi = mean_ci(sub["__pi__"], ci=args.ci)

        out["T(ms)"] = fmt_mean(m_T, nd=1)
        out["|M|"] = fmt_mean(m_M, nd=1)
        out["Mem"] = fmt_mean(m_mem, nd=1)
        out["η"] = fmt_mean(m_eta, nd=2)
        out["L"] = fmt_mean_pm(m_L, h_L, nd=2)
        out["|Π|"] = fmt_mean(m_pi, nd=1) if n_pi > 0 else ""
        return out

    def summarize_tbuild(ds: str) -> str:
        if ds in DATASET_INFO and "T_build(ms)" in DATASET_INFO[ds]:
            return fmt_mean(float(DATASET_INFO[ds]["T_build(ms)"]), nd=1)
        sub = data[data["dataset"] == ds]
        m, _, n = mean_ci(sub["__tbuild_ms__"], ci=args.ci)
        return fmt_mean(m, nd=1) if n > 0 else ""

    rows: List[Dict[str, str]] = []

    for ds in datasets:
        row: Dict[str, str] = {"Dataset": ds}
        a, c, d = get_dataset_info(ds)
        row["#attr"] = a
        row["#cells"] = c
        row["#DC"] = d

        for mech in ["MIN", "EXP", "GUM"]:
            sub = data[(data["dataset"] == ds) & (data["method"] == mech)]
            block = summarize_block(sub) if len(sub) else {"T(ms)": "", "|M|": "", "Mem": "", "η": "", "L": "", "|Π|": ""}
            row[f"{mech} T(ms)"] = block["T(ms)"]
            row[f"{mech} |M|"] = block["|M|"]
            row[f"{mech} Mem"] = block["Mem"]
            row[f"{mech} η"] = block["η"]
            row[f"{mech} L"] = block["L"]
            if mech in {"EXP", "GUM"}:
                row[f"{mech} |Π|"] = block["|Π|"]

        row["T_build(ms)"] = summarize_tbuild(ds)
        rows.append(row)

    out = pd.DataFrame(rows)

    ordered_cols = [
        "Dataset", "#attr", "#cells", "#DC",
        "MIN T(ms)", "MIN |M|", "MIN Mem", "MIN η", "MIN L",
        "EXP T(ms)", "EXP |M|", "EXP Mem", "EXP η", "EXP L", "EXP |Π|",
        "GUM T(ms)", "GUM |M|", "GUM Mem", "GUM η", "GUM L", "GUM |Π|",
        "T_build(ms)",
    ]
    ordered_cols = [c for c in ordered_cols if c in out.columns]
    out = out[ordered_cols]

    out.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}\n")

    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 260):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
