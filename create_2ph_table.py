#!/usr/bin/env python3
import csv
import math
from collections import defaultdict

# ============================================================
# INPUT FILES (EDIT THESE)
# ============================================================

OFFLINE_CSV = "offline_template_bench.csv"
ONLINE_CSV  = "del2ph_data_standardized_vFinal.csv"

OUTPUT_TEX = "offline_online_table.tex"

# Which offline memory column to use:
#   "rss_delta_build_bytes"  -> system RSS delta
#   "py_peak_build_bytes"    -> Python-level peak
OFFLINE_MEMORY_COL = "rss_delta_build_bytes"

# Formatting
TIME_DECIMALS = 3
MEM_DECIMALS  = 2
NUM_DECIMALS  = 3


# ============================================================
# Helpers
# ============================================================

def _to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def _escape_latex(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("#", r"\#")
         .replace("$", r"\$")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("~", r"\textasciitilde{}")
         .replace("^", r"\textasciicircum{}")
    )


def mean(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt_num(x, d=NUM_DECIMALS):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def fmt_time(x):
    return fmt_num(x, TIME_DECIMALS)


def fmt_mem_mb(bytes_val):
    if bytes_val is None or (isinstance(bytes_val, float) and math.isnan(bytes_val)):
        return "--"
    return f"{bytes_val / (1024):.{MEM_DECIMALS}f}"


# ============================================================
# CSV parsing
# ============================================================

def read_offline_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "build_time_s", OFFLINE_MEMORY_COL}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Offline CSV missing columns: {missing}")

        for r in reader:
            ds = (r.get("dataset") or "").strip()
            if not ds:
                continue
            rows.append({
                "dataset": ds,
                "build_time_s": _to_float(r.get("build_time_s")),
                "mem_bytes": _to_float(r.get(OFFLINE_MEMORY_COL)),
            })
    return rows


def read_online_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "leakage", "utility", "paths_blocked", "mask_size"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Online CSV missing columns: {missing}")

        for r in reader:
            ds = (r.get("dataset") or "").strip()
            if not ds:
                continue
            rows.append({
                "dataset": ds,
                "leakage": _to_float(r.get("leakage")),
                "utility": _to_float(r.get("utility")),
                "paths_blocked": _to_float(r.get("paths_blocked")),
                "mask_size": _to_float(r.get("mask_size")),
            })
    return rows


# ============================================================
# Aggregation
# ============================================================

def group_offline_means(rows):
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    out = {}
    for ds, xs in by_ds.items():
        out[ds] = {
            "build_time_s": mean([x["build_time_s"] for x in xs]),
            "mem_bytes": mean([x["mem_bytes"] for x in xs]),
        }
    return out


def group_online_means(rows):
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    out = {}
    for ds, xs in by_ds.items():
        out[ds] = {
            "leakage": mean([x["leakage"] for x in xs]),
            "mask_size": mean([x["mask_size"] for x in xs]),
            "utility": mean([x["utility"] for x in xs]),
            "paths_blocked": mean([x["paths_blocked"] for x in xs]),
        }
    return out


# ============================================================
# LaTeX output
# ============================================================

def write_latex_table(off, on, path):
    datasets = sorted(set(off) | set(on))

    lines = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{l rr rrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Offline phase}} & \multicolumn{4}{c}{\textbf{Online phase}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
        r"\textbf{Dataset} & \textbf{Build time (s)} & \textbf{Mem (MB)}"
        r" & \textbf{Leakage} & \textbf{Mask size} & \textbf{Utility} & \textbf{Paths blocked} \\",
        r"\midrule",
    ]

    for ds in datasets:
        o = off.get(ds, {})
        n = on.get(ds, {})
        lines.append(
            " & ".join([
                _escape_latex(ds),
                fmt_time(o.get("build_time_s")),
                fmt_mem_mb(o.get("mem_bytes")),
                fmt_num(n.get("leakage")),
                fmt_num(n.get("mask_size")),
                fmt_num(n.get("utility")),
                fmt_num(n.get("paths_blocked")),
            ]) + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-dataset averages for offline template construction and online deletion evaluation.}",
        r"\label{tab:offline-online}",
        r"\end{table}",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    offline_rows = read_offline_csv(OFFLINE_CSV)
    online_rows  = read_online_csv(ONLINE_CSV)

    off_means = group_offline_means(offline_rows)
    on_means  = group_online_means(online_rows)

    write_latex_table(off_means, on_means, OUTPUT_TEX)

    print(f"[OK] Wrote LaTeX table → {OUTPUT_TEX}")
    print(f"[OK] Offline memory column used → {OFFLINE_MEMORY_COL}")


if __name__ == "__main__":
    main()
