#!/usr/bin/env python3
"""
Recompute confidence-bounded DC weights and regenerate plots/tables.
Evaluation-only (independent of the paper's leakage model).
"""
import os, json, math, runpy
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# STYLE — matches the paper's global rcParams
# =============================================================================
FS            = 13
SPLIT_TICK_FS = 15
SPLIT_LEG_FS  = 16
SPLIT_SUP_FS  = 15

plt.rcParams.update({
    "text.usetex":           True,
    "font.family":           "serif",
    "text.latex.preamble":   (
        r"\usepackage{amsmath}"
        r"\usepackage{amssymb}"
        r"\usepackage{bm}"
        r"\usepackage{graphicx}"
    ),
    "font.size":             FS,
    "axes.labelsize":        FS,
    "axes.titlesize":        FS,
    "legend.fontsize":       FS,
    "legend.title_fontsize": FS,
    "xtick.labelsize":       FS,
    "ytick.labelsize":       FS,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "axes.grid":             False,
    "grid.alpha":            0.3,
})



# Line colours — darker shades of each dataset's LaTeX panel colour (rowA–rowE)
# airport=rowA, hospital=rowB, adult=rowC, flights=rowD, tax=rowE
LINE_COLORS = {
    "airport":  "#4A7DB5",  # darker rowA RGB(225,235,248) light blue
    "hospital": "#4A9A4A",  # darker rowB RGB(228,243,228) light green
    "adult":    "#C8841A",  # darker rowC RGB(248,234,210) light orange
    "flights":  "#7A4DAA",  # darker rowD RGB(235,225,245) light purple
    "tax":      "#C04A4A",  # darker rowE RGB(245,225,225) light pink
}

# -------------------- HARD-CODED CONFIG --------------------
SEED     = 7
S        = 300_000
CI_LEVEL = 0.90
Z        = {0.90: 1.6448536269514722,
            0.95: 1.959963984540054,
            0.99: 2.5758293035489004}[CI_LEVEL]
M_MAX    = 3
N_V_MIN  = 200

GAMMAS      = [0.25, 0.50]
GAMMA_SWEEP = [0.05, 0.10, 0.15, 0.25, 0.50]

N_E_MIN: Dict[str, int] = {
    "adult":    200,
    "airport":  200,
    "flights":  74000,
    "hospital": 50,
    "tax":      10000,
}

# Add/uncomment datasets as their CSV and DC files become available.
DATASETS = [
    ("adult",    "adult.csv",                   "adult_0.01_dcs",          "jsonl"),
    ("airport",  "airport.csv",                 "airport_0.01_dcs",        "jsonl"),
    ("flights",  "flights.csv",                 "flights_0.01_dcs",        "jsonl"),
    ("hospital", "hospital.csv",                "hospital_0.0_dcs",        "jsonl"),
    ("tax",      "tax.csv",                     "raw_tax_dcs.py", "taxpy"),
]

# Resolve all paths relative to this script's location so it can be
# run from any working directory.
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(SRC_DIR, "outputs_dc_weighting")
FIG_DIR  = os.path.join(OUT_ROOT, "fig")

# -------------------- Core helpers --------------------
def gtag(g: float) -> str:
    return f"{g:.2f}".replace(".", "p")

def wilson_bounds(k: int, n: int, z: float) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat   = k / n
    denom  = 1.0 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    rad    = (z * math.sqrt((phat*(1-phat) + (z*z)/(4*n)) / n)) / denom
    return (max(0.0, center-rad), min(1.0, center+rad))

Predicate = Tuple[int, str, str, int, str]
DC        = List[Predicate]

def load_dcs_jsonl(path: str) -> List[DC]:
    dcs: List[DC] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj   = json.loads(line)
            preds = obj.get("predicates", [])
            if len(preds) == 0 or len(preds) > M_MAX:
                continue
            parsed: DC = []
            ok = True
            for p in preds:
                i1 = int(p["index1"]); i2 = int(p["index2"])
                if i1 not in (0,1) or i2 not in (0,1):
                    ok = False; break
                c1 = p["column1"]["columnIdentifier"]
                c2 = p["column2"]["columnIdentifier"]
                op = p["op"]
                parsed.append((i1, c1, op, i2, c2))
            if ok:
                dcs.append(parsed)
    return dcs

def load_dcs_tax_py(path: str) -> List[DC]:
    mod = runpy.run_path(path)
    denial_constraints = mod["denial_constraints"]
    attr_to_col = {
        "fname":"FName","lname":"LName","gender":"Gender",
        "area_code":"AreaCode","phone":"Phone","city":"City",
        "state":"State","zip":"Zip","marital_status":"MaritalStatus",
        "has_child":"HasChild","salary":"Salary","rate":"Rate",
        "single_exemp":"SingleExemp","married_exemp":"MarriedExemp",
        "child_exemp":"ChildExemp",
    }
    dcs: List[DC] = []
    for dc in denial_constraints:
        if len(dc) == 0 or len(dc) > M_MAX:
            continue
        parsed: DC = []
        for (lhs, op, rhs) in dc:
            _, aL = lhs.split(".")
            _, aR = rhs.split(".")
            parsed.append((0, attr_to_col[aL], op, 1, attr_to_col[aR]))
        dcs.append(parsed)
    return dcs

def dc_to_str(dc: DC) -> str:
    parts = []
    for (i1, c1, op, i2, c2) in dc:
        t1 = "t0" if i1==0 else "t1"
        t2 = "t0" if i2==0 else "t1"
        parts.append(f"{t1}.{c1} {op} {t2}.{c2}")
    return " AND ".join(parts)

def sample_pairs(n_rows: int, S: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    i0 = rng.integers(0, n_rows, size=S, dtype=np.int64)
    i1 = rng.integers(0, n_rows, size=S, dtype=np.int64)
    eq = (i0 == i1)
    while eq.any():
        i1[eq] = rng.integers(0, n_rows, size=int(eq.sum()), dtype=np.int64)
        eq = (i0 == i1)
    return i0, i1

def eval_pred(a: np.ndarray, b: np.ndarray, op: str) -> Tuple[np.ndarray, np.ndarray]:
    valid = ~(pd.isna(a) | pd.isna(b))
    holds = np.zeros(len(a), dtype=bool)
    if op in ("EQUAL","=="):
        holds[valid] = (a[valid] == b[valid])
    elif op in ("UNEQUAL","!="):
        holds[valid] = (a[valid] != b[valid])
    elif op in ("LESS","<"):
        holds[valid] = (a[valid] < b[valid])
    elif op in ("GREATER",">"):
        holds[valid] = (a[valid] > b[valid])
    elif op in ("LESS_EQUAL","LESS_OR_EQUAL","LESSEQUAL","<="):
        holds[valid] = (a[valid] <= b[valid])
    elif op in ("GREATER_EQUAL","GREATER_OR_EQUAL","GREATEREQUAL",">="):
        holds[valid] = (a[valid] >= b[valid])
    else:
        holds[valid] = False
    return valid, holds

def compute_direction_stats(df: pd.DataFrame, dcs: List[DC]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n   = len(df)
    i0, i1 = sample_pairs(n, S, rng)

    cols = set()
    for dc in dcs:
        for (_, c1, _, _, c2) in dc:
            cols.add(c1); cols.add(c2)

    sampled = {}
    for c in cols:
        col = df[c].to_numpy()
        sampled[(0,c)] = col[i0]
        sampled[(1,c)] = col[i1]

    out_rows = []
    for dc_id, dc in enumerate(dcs, start=1):
        m          = len(dc)
        holds_list = []
        valid_e    = np.ones(S, dtype=bool)

        for (idx1, c1, op, idx2, c2) in dc:
            a = sampled[(idx1, c1)]
            b = sampled[(idx2, c2)]
            v, h = eval_pred(a, b, op)
            valid_e &= v
            holds_list.append(h)

        nV = int(valid_e.sum())
        if nV < N_V_MIN:
            for j in range(m):
                out_rows.append({"dc_id":dc_id,"m":m,"head_j":j,"nV":nV,
                                 "nE":0,"UCB0":np.nan,"LCB1":np.nan,"base":0.0})
            continue

        for j in range(m):
            head    = holds_list[j]
            k0      = int((valid_e & (~head)).sum())
            _, UCB0 = wilson_bounds(k0, nV, Z)
            denom   = 1.0 - UCB0
            if denom <= 1e-12:
                out_rows.append({"dc_id":dc_id,"m":m,"head_j":j,"nV":nV,
                                 "nE":0,"UCB0":UCB0,"LCB1":np.nan,"base":0.0})
                continue

            premise = valid_e.copy()
            for i in range(m):
                if i != j:
                    premise &= holds_list[i]
            nE = int(premise.sum())
            if nE <= 0:
                out_rows.append({"dc_id":dc_id,"m":m,"head_j":j,"nV":nV,
                                 "nE":0,"UCB0":UCB0,"LCB1":np.nan,"base":0.0})
                continue

            k1      = int((premise & (~head)).sum())
            LCB1, _ = wilson_bounds(k1, nE, Z)
            base    = (LCB1 - UCB0) / denom
            out_rows.append({"dc_id":dc_id,"m":m,"head_j":j,"nV":nV,
                             "nE":nE,"UCB0":UCB0,"LCB1":LCB1,"base":base})
    return pd.DataFrame(out_rows)

def best_base_by_nE(dir_df: pd.DataFrame, nEmin: int) -> pd.DataFrame:
    d = dir_df.copy()
    d["eligible"] = (d["nE"] >= nEmin)
    d.loc[~d["eligible"], "base"] = -np.inf

    idx  = d.groupby("dc_id")["base"].idxmax()
    best = d.loc[idx, ["dc_id","m","nV","head_j","nE","UCB0","LCB1","base"]].copy()
    best.rename(columns={"head_j":"best_head_j","nE":"nE_best","base":"best_base"}, inplace=True)

    best["best_base"] = best["best_base"].replace(-np.inf, 0.0)
    best.loc[best["best_base"] < 0,       "best_base"]   = 0.0
    best.loc[best["nE_best"] < nEmin,     "best_base"]   = 0.0
    best.loc[best["nE_best"] < nEmin,     "best_head_j"] = -1
    best.loc[best["nE_best"] < nEmin,     "nE_best"]     = 0
    return best.reset_index(drop=True)

# ==============================
# MAIN
# ==============================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)


    FIG_W  = 6.0
    FIG_H  = 2.2    # taller so legend has room above the plot area
    LEFT   = 0.16   # wider: stops y-axis label being clipped
    RIGHT  = 0.985
    BOTTOM = 0.25
    TOP    = 0.76   # lower ceiling leaves more figure-fraction space above for legend

    summary_rows = []
    sweep_rows   = []
    all_best     = {}

    # ==========================================================
    # DATA PROCESSING — skip datasets whose files are missing
    # ==========================================================
    active_datasets = []
    for name, csv_path, dc_path, fmt in DATASETS:
        csv_full = os.path.join(SRC_DIR, csv_path)
        dc_full  = os.path.join(SRC_DIR, dc_path)
        if not os.path.exists(csv_full) or not os.path.exists(dc_full):
            print(f"[skip] {name}: file(s) not found, skipping.")
            continue

        df = pd.read_csv(csv_full)
        if name == "tax":
            for c in ["AreaCode","Zip","Salary","Rate",
                      "SingleExemp","MarriedExemp","ChildExemp"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

        dcs     = (load_dcs_jsonl(dc_full) if fmt == "jsonl"
                   else load_dcs_tax_py(dc_full))
        dc_strs = [dc_to_str(dc) for dc in dcs]

        dir_df  = compute_direction_stats(df, dcs)
        nEmin   = int(N_E_MIN.get(name, 200))
        best_df = best_base_by_nE(dir_df, nEmin)
        best_df.insert(0, "dataset", name)
        best_df.insert(2, "dc_str",  dc_strs)

        for g in GAMMAS:
            best_df[f"w_g{gtag(g)}"] = np.maximum(
                0.0, best_df["best_base"].to_numpy(float) - g
            )

        summary_rows.append({
            "dataset":        name,
            "kept_wpos_g025": int((best_df["w_g0p25"] > 0).sum()),
            "kept_wpos_g050": int((best_df["w_g0p50"] > 0).sum()),
        })
        all_best[name] = best_df
        active_datasets.append((name, csv_path, dc_path, fmt))

    if not active_datasets:
        print("No datasets loaded — nothing to plot.")
        return

    summary_df = pd.DataFrame(summary_rows)

    # ==========================================================
    # FIGURE 1: Gamma Sweep
    # ==========================================================
    for name, *_ in active_datasets:
        bb = all_best[name]["best_base"].to_numpy(float)
        for g in GAMMA_SWEEP:
            sweep_rows.append({
                "dataset":    name,
                "gamma_frac": g,
                "kept_wpos":  int((np.maximum(0.0, bb - g) > 0).sum()),
            })

    sweep_df = pd.DataFrame(sweep_rows)


    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for name, *_ in active_datasets:
        sub          = sweep_df[sweep_df["dataset"] == name].sort_values("gamma_frac")
        display_name = "Flight" if name == "flights" else name.capitalize()
        ax.plot(
            sub["gamma_frac"],
            sub["kept_wpos"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=LINE_COLORS.get(name, "#333333"),
            label=display_name,
        )

    ax.set_xlabel(r"$\gamma_{\mathrm{frac}}$", fontsize=FS + 1, labelpad=2)
    ax.set_ylabel(r"\# Dependencies with $w(e)>0$", fontsize=FS - 2, labelpad=4)
    ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS - 2)
    ax.grid(True, alpha=0.3)

    leg1 = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.40),
        ncol=5,
        frameon=True,
        fontsize=FS - 2,        # smaller than other figures
        borderpad=0.3,
        handlelength=1.2,
        handletextpad=0.3,
        columnspacing=0.5,
        markerscale=0.8,
    )
    for t in leg1.get_texts():
        t.set_fontweight("bold")

    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)
    fig.savefig(os.path.join(FIG_DIR, "gamma_frac_vs_kept_wpos_all_datasets.pdf"))
    plt.close(fig)

    # ==========================================================
    # FIGURE 2: Violin Plot
    # ==========================================================
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    data_025 = [
        all_best[name].loc[all_best[name]["w_g0p25"] > 0, "w_g0p25"].to_numpy(float)
        for name, *_ in active_datasets
    ]
    data_050 = [
        all_best[name].loc[all_best[name]["w_g0p50"] > 0, "w_g0p50"].to_numpy(float)
        for name, *_ in active_datasets
    ]

    pos    = np.arange(1, len(active_datasets)+1, dtype=float)
    offset = 0.18

    vp1 = ax.violinplot(data_025, positions=pos-offset, widths=0.30,
                        showmeans=False, showmedians=False, showextrema=False)
    vp2 = ax.violinplot(data_050, positions=pos+offset, widths=0.30,
                        showmeans=False, showmedians=False, showextrema=False)

    # Violin colours unchanged from original
    c1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    c2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    for b in vp1['bodies']:
        b.set_facecolor(c1); b.set_edgecolor('black')
        b.set_alpha(0.78);   b.set_linewidth(0.6)
    for b in vp2['bodies']:
        b.set_facecolor(c2); b.set_edgecolor('black')
        b.set_alpha(0.78);   b.set_linewidth(0.6)

    xtick_labels = [
        "Flight" if name == "flights" else name.capitalize()
        for name, *_ in active_datasets
    ]
    ax.set_xticks(pos)
    ax.set_xticklabels(xtick_labels, fontsize=SPLIT_TICK_FS - 2,
                       fontweight="bold")
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=SPLIT_TICK_FS - 2)

    ax.set_ylabel(r"Dependency Weight $w(e)$", fontsize=FS - 1, labelpad=4)
    ax.set_xlabel("Datasets",                  fontsize=FS + 1, labelpad=4)

    ax.legend(
        handles=[
            Patch(facecolor=c1, edgecolor='black',
                  label=r"$\gamma_{\mathrm{frac}} = 0.25$"),
            Patch(facecolor=c2, edgecolor='black',
                  label=r"$\gamma_{\mathrm{frac}} = 0.50$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.50),
        ncol=2,
        frameon=True,
        fontsize=SPLIT_LEG_FS - 2,
        borderpad=0.5,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)
    fig.savefig(os.path.join(FIG_DIR, "fig_weight_violin_gamma025.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()