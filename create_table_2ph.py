#!/usr/bin/env python3
import os
import csv
import math
from collections import defaultdict

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------
INPUT_FILE = "csv_files/del2ph_data_standardized_vFinal.csv"  # <- paste your filename here
OUTPUT_TEX = None  # None => auto: "<input_basename>_table.tex"

# Optional: filter to a single method (set to None to include all rows)
FILTER_METHOD = None  # e.g. "delgum", "delmin", "delexp"

# ------------------------------------------------------------
# Style (kept exactly as you requested, even though we don't plot)
# ------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.variant": "small-caps",
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "legend.fontsize": 12,
    })
except Exception:
    pass

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None

def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)

def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    if abs(v) >= 1000:
        return f"{v:.1f}"
    if abs(v) >= 100:
        return f"{v:.2f}"
    return f"{v:.{nd}f}"

def latex_escape(s: str) -> str:
    # Minimal escaping for common dataset names
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("$", "\\$")
             .replace("#", "\\#")
             .replace("_", "\\_")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}"))

def get_total_time(row: dict):
    # Prefer explicit total_time if present
    tt = _to_float(row.get("total_time"))
    if tt is not None:
        return tt

    # Otherwise sum known components if available
    parts = ["init_time", "model_time", "del_time", "update_time"]
    got_any = False
    total = 0.0
    for p in parts:
        v = _to_float(row.get(p))
        if v is not None:
            got_any = True
            total += v
    return total if got_any else None

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"File not found: {INPUT_FILE}")

    out_tex = OUTPUT_TEX
    if out_tex is None:
        base = os.path.splitext(os.path.basename(INPUT_FILE))[0]
        out_tex = f"{base}_table.tex"

    # Collect per-dataset values
    per_ds = defaultdict(lambda: {
        "mask_size": [],
        "leakage": [],
        "utility": [],
        "total_time": [],
        "paths_blocked": [],
    })

    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("CSV appears to have no header row.")

        # Basic sanity: require dataset column at least
        if "dataset" not in {h.strip() for h in reader.fieldnames}:
            raise SystemExit("CSV must contain a 'dataset' column.")

        for row in reader:
            if FILTER_METHOD is not None:
                if (row.get("method") or "").strip().lower() != FILTER_METHOD.strip().lower():
                    continue

            ds = (row.get("dataset") or "").strip()
            if not ds:
                continue

            per_ds[ds]["mask_size"].append(_to_float(row.get("mask_size")))
            per_ds[ds]["leakage"].append(_to_float(row.get("leakage")))
            per_ds[ds]["utility"].append(_to_float(row.get("utility")))
            per_ds[ds]["total_time"].append(get_total_time(row))
            per_ds[ds]["paths_blocked"].append(_to_float(row.get("paths_blocked")))

    if not per_ds:
        msg = "No rows found."
        if FILTER_METHOD is not None:
            msg += f" (After filtering method={FILTER_METHOD})"
        raise SystemExit(msg)

    # Build LaTeX table
    # Note: adjust caption/label as you like.
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Mask size (avg) & Leakage (avg) & Utility (avg) & Total time (avg) & Paths blocked (avg) \\")
    lines.append(r"\midrule")

    for ds in sorted(per_ds.keys()):
        m_mask = mean(per_ds[ds]["mask_size"])
        m_leak = mean(per_ds[ds]["leakage"])
        m_util = mean(per_ds[ds]["utility"])
        m_time = mean(per_ds[ds]["total_time"])
        m_path = mean(per_ds[ds]["paths_blocked"])

        lines.append(
            f"{latex_escape(ds)} & {fmt(m_mask)} & {fmt(m_leak)} & {fmt(m_util)} & {fmt(m_time)} & {fmt(m_path)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    caption = r"Averages by dataset."
    if FILTER_METHOD is not None:
        caption = rf"Averages by dataset for \texttt{{{latex_escape(FILTER_METHOD)}}}."
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\label{tab:dataset-averages}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines) + "\n"

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex)

    print(latex)
    print(f"\nWrote: {out_tex}")

if __name__ == "__main__":
    main()
