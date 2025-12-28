#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# CONFIG: choose which parameter family you’re plotting
#   Set PARAM_FAMILY to: "epsilon" or "alpha" or "beta"
#   and set the two input files for delgum + delexp.
# ============================================================

PARAM_FAMILY = "epsilon"  # "epsilon" | "alpha" | "beta"

FILES = {
    "epsilon": {
        "delgum": "ablation_delgum_epsilon.csv",
        "delexp": "ablation_delexp_epsilon.csv",
        "output": "combined_epsilon_alpha_beta_grid.pdf",
        "x_label": "epsilon",
    },
    "alpha": {
        "delgum": "ablation_delgum_alpha.csv",
        "delexp": "ablation_delexp_alpha.csv",
        "output": "combined_epsilon_alpha_beta_grid.pdf",
        "x_label": "alpha",
    },
    "beta": {
        "delgum": "ablation_delgum_beta.csv",
        "delexp": "ablation_delexp_beta.csv",
        "output": "combined_epsilon_alpha_beta_grid.pdf",
        "x_label": "beta",
    },
}

# You want a single PDF with:
#   row 1 = epsilon (5 datasets)
#   row 2 = alpha   (5 datasets)
#   row 3 = beta    (5 datasets)
# So we will ALWAYS render all 3 rows using the corresponding files above.
# (If a file is missing, that row will just be skipped.)

OUTPUT_PDF = "delgum_vs_delexp_epsilon_alpha_beta_grid.pdf"

DATASETS_IN_ORDER = ["airport", "hospital", "ncvoter", "Onlineretail", "adult"]

# -----------------------------
# Controls
# -----------------------------
USE_LOG_X = False
LOG_X_MIN = None

SHOW_MEAN_POINTS = True
POINT_SIZE = 22

CI_LEVEL = 0.95
USE_T_DISTRIBUTION = True

MASK_LOG_Y = False
MASK_LOG_EPS = 1e-9

# Delta-utility reference:
#   "min" -> baseline is smallest x (strong privacy)
#   "max" -> baseline is largest x (weak privacy)
DELTA_U_BASELINE = "min"

# Right axis cap requested
MASK_Y_MAX = 15

# Delmin mask sizes (reference black dot on right y-axis)
AIRPORT_DELMIN_MASK = 15
HOSPITAL_DELMIN_MASK = 12
NCVOTER_DELMIN_MASK = 14
ONLINE_RETAILER_DELMIN_MASK = 8
ADULT_DELMIN_MASK = 6

# -----------------------------
# Fixed colors
# -----------------------------
COLOR_LEAKAGE = "red"
COLOR_DELTAU = "green"
COLOR_MASK = "blue"

# Markers (requested)
# delgum: circled dot (hollow circle)
# delexp: triangle (hollow triangle)
MARKER_DELGUM = "o"
MARKER_DELEXP = "^"

# Line styles: keep both connected, NOT dashed.
# We'll differentiate by linewidth/alpha (both solid).
LW_DELGUM = 2.6
LW_DELEXP = 2.0
ALPHA_DELGUM_LINE = 0.95
ALPHA_DELEXP_LINE = 0.75

# CI band visibility
ALPHA_CI_DELGUM = 0.18
ALPHA_CI_DELEXP = 0.12

# Legend size/location
LEGEND_FONT_SIZE = 9

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.variant": "small-caps",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": LEGEND_FONT_SIZE,
})


# ============================================================
# Parsing
# ============================================================
def parse_tempdata(path: str) -> pd.DataFrame:
    """
    File format: repeats a CSV header between dataset blocks, and datasets cycle in DATASETS_IN_ORDER.
      epsilon(or alpha/beta),leakage,utility,mask_size
      <rows...>
      epsilon(or alpha/beta),leakage,utility,mask_size
      <rows...>
      ...
    """
    rows = []
    current_ds_idx = -1
    current_ds = None
    saw_any_header = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Header line marks a NEW dataset block
            if line.lower().startswith(("epsilon,", "alpha,", "beta,")):
                saw_any_header = True
                current_ds_idx = (current_ds_idx + 1) % len(DATASETS_IN_ORDER)
                current_ds = DATASETS_IN_ORDER[current_ds_idx]
                continue

            if current_ds is None:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue

            try:
                x = float(parts[0])
                leakage = float(parts[1])
                utility = float(parts[2])
                mask = float(parts[3])
            except ValueError:
                continue

            rows.append((current_ds, x, leakage, utility, mask))

    if not saw_any_header or not rows:
        raise ValueError(f"Failed to parse input file: {path}")

    return pd.DataFrame(rows, columns=["dataset", "x", "leakage", "utility", "mask_size"])


def pretty_name(ds: str) -> str:
    if ds.lower() == "ncvoter":
        return "NCVoter"
    if ds.lower() == "onlineretail":
        return "Onlineretail"
    return ds.capitalize()


# ============================================================
# Stats / CI
# ============================================================
def t_critical(alpha: float, df: int) -> float:
    if df <= 0:
        return 1.96
    if USE_T_DISTRIBUTION:
        try:
            from scipy.stats import t
            return float(t.ppf(1.0 - alpha / 2.0, df))
        except Exception:
            pass
    return 1.96


def mean_ci(raw: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Returns per (dataset, x): mean, std, n, sem, low, high
    """
    g = (
        raw.groupby(["dataset", "x"], as_index=False)
           .agg(mean=(col, "mean"),
                std=(col, "std"),
                n=(col, "size"))
           .sort_values(["dataset", "x"])
           .reset_index(drop=True)
    )

    g["std"] = g["std"].fillna(0.0)
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))

    alpha = 1.0 - CI_LEVEL
    tvals = np.array([t_critical(alpha, int(n - 1)) for n in g["n"]], dtype=float)
    half = tvals * g["sem"].to_numpy(dtype=float)

    g["low"] = g["mean"] - half
    g["high"] = g["mean"] + half
    return g


def build_delta_u(util_ci: pd.DataFrame) -> pd.DataFrame:
    """
    Build ΔU = mean(U_x) - mean(U_ref) per dataset.
    CI is approximated as unpaired difference:
      SE(Δ) = sqrt(SE_x^2 + SE_ref^2)
      halfwidth = tcrit(df_eff) * SE(Δ)
    """
    util = util_ci.copy()
    util = util.sort_values(["dataset", "x"]).reset_index(drop=True)

    ref_rows = []
    for ds, sub in util.groupby("dataset", sort=False):
        if sub.empty:
            continue
        if DELTA_U_BASELINE.lower() == "max":
            ref = sub.loc[sub["x"].idxmax()]
        else:
            ref = sub.loc[sub["x"].idxmin()]
        ref_rows.append(ref)

    ref_df = pd.DataFrame(ref_rows)[["dataset", "x", "mean", "sem", "n"]].rename(
        columns={"x": "ref_x", "mean": "ref_mean", "sem": "ref_sem", "n": "ref_n"}
    )

    out = util.merge(ref_df, on="dataset", how="left")

    out["d_mean"] = out["mean"] - out["ref_mean"]
    out["d_sem"] = np.sqrt(out["sem"]**2 + out["ref_sem"]**2)

    alpha = 1.0 - CI_LEVEL
    df_eff = (np.minimum(out["n"].astype(int) - 1, out["ref_n"].astype(int) - 1)).clip(lower=0)
    tvals = np.array([t_critical(alpha, int(df)) for df in df_eff], dtype=float)

    half = tvals * out["d_sem"].to_numpy(dtype=float)
    out["d_low"] = out["d_mean"] - half
    out["d_high"] = out["d_mean"] + half

    return out[["dataset", "x", "d_mean", "d_low", "d_high", "ref_x"]]


def compute_summary_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    For one method+family (delgum epsilon, or delexp alpha, etc.)
    produce per dataset,x:
      leak_mean/low/high
      d_mean/d_low/d_high
      mask_mean/low/high
      ref_x
    """
    if USE_LOG_X:
        raw = raw[raw["x"] > 0].copy()

    L = mean_ci(raw, "leakage").rename(columns={"mean": "leak_mean", "low": "leak_low", "high": "leak_high"})
    U = mean_ci(raw, "utility")  # keep sign
    M = mean_ci(raw, "mask_size").rename(columns={"mean": "mask_mean", "low": "mask_low", "high": "mask_high"})

    dU = build_delta_u(U)

    g = (
        L.merge(dU, on=["dataset", "x"], how="inner")
         .merge(M[["dataset", "x", "mask_mean", "mask_low", "mask_high"]], on=["dataset", "x"], how="inner")
         .sort_values(["dataset", "x"])
         .reset_index(drop=True)
    )
    return g


def get_delmin_mask(ds: str) -> float:
    d = ds.lower()
    if d == "airport":
        return float(AIRPORT_DELMIN_MASK)
    if d == "hospital":
        return float(HOSPITAL_DELMIN_MASK)
    if d == "ncvoter":
        return float(NCVOTER_DELMIN_MASK)
    if d == "onlineretail":
        return float(ONLINE_RETAILER_DELMIN_MASK)
    if d == "adult":
        return float(ADULT_DELMIN_MASK)
    return np.nan


# ============================================================
# Plotting helpers
# ============================================================
def plot_one_subplot(ax, ds: str, x_label: str, gum: pd.DataFrame, exp: pd.DataFrame):
    """
    One dataset plot (one panel):
      Left axis: Leakage (red), ΔUtility (green)
      Right axis: Mask size (blue)
      delgum: solid, hollow circle points
      delexp: solid (thinner), hollow triangle points
      CI bands for BOTH
      delmin mask: black dot ON right y-axis spine
    """
    ax2 = ax.twinx()

    # pick method subsets
    sub_g = gum[gum["dataset"] == ds].copy() if gum is not None else pd.DataFrame()
    sub_e = exp[exp["dataset"] == ds].copy() if exp is not None else pd.DataFrame()

    # Sort
    if not sub_g.empty:
        sub_g = sub_g.sort_values("x")
    if not sub_e.empty:
        sub_e = sub_e.sort_values("x")

    # Mask y-limits
    ax2.set_ylim(0, MASK_Y_MAX)

    # LOG/axis options
    if USE_LOG_X:
        ax.set_xscale("log")
        ax2.set_xscale("log")
        if LOG_X_MIN is not None:
            ax.set_xlim(left=LOG_X_MIN)

    if MASK_LOG_Y:
        ax2.set_yscale("log")

    # ----- delgum -----
    if not sub_g.empty:
        x = sub_g["x"].to_numpy(dtype=float)

        # Leakage
        ax.plot(x, sub_g["leak_mean"], color=COLOR_LEAKAGE, lw=LW_DELGUM,
                alpha=ALPHA_DELGUM_LINE, label="Leakage (delgum)")
        ax.fill_between(x, sub_g["leak_low"], sub_g["leak_high"],
                        color=COLOR_LEAKAGE, alpha=ALPHA_CI_DELGUM)

        # ΔU
        base_txt = "min" if DELTA_U_BASELINE.lower() != "max" else "max"
        ax.plot(x, sub_g["d_mean"], color=COLOR_DELTAU, lw=LW_DELGUM,
                alpha=ALPHA_DELGUM_LINE, label=f"ΔUtility (delgum, vs {base_txt})")
        ax.fill_between(x, sub_g["d_low"], sub_g["d_high"],
                        color=COLOR_DELTAU, alpha=ALPHA_CI_DELGUM)

        # Mask on right axis
        yM = sub_g["mask_mean"].to_numpy(dtype=float)
        loM = sub_g["mask_low"].to_numpy(dtype=float)
        hiM = sub_g["mask_high"].to_numpy(dtype=float)
        if MASK_LOG_Y:
            yM = np.maximum(yM, MASK_LOG_EPS)
            loM = np.maximum(loM, MASK_LOG_EPS)
            hiM = np.maximum(hiM, MASK_LOG_EPS)

        ax2.plot(x, yM, color=COLOR_MASK, lw=LW_DELGUM,
                 alpha=ALPHA_DELGUM_LINE, label="Mask Size (delgum)")
        ax2.fill_between(x, loM, hiM, color=COLOR_MASK, alpha=ALPHA_CI_DELGUM)

        if SHOW_MEAN_POINTS:
            ax.scatter(x, sub_g["leak_mean"], s=POINT_SIZE, marker=MARKER_DELGUM,
                       facecolors="none", edgecolors=COLOR_LEAKAGE, linewidths=1.2, zorder=5)
            ax.scatter(x, sub_g["d_mean"], s=POINT_SIZE, marker=MARKER_DELGUM,
                       facecolors="none", edgecolors=COLOR_DELTAU, linewidths=1.2, zorder=5)
            ax2.scatter(x, yM, s=POINT_SIZE, marker=MARKER_DELGUM,
                        facecolors="none", edgecolors=COLOR_MASK, linewidths=1.2, zorder=5)

    # ----- delexp -----
    if not sub_e.empty:
        x = sub_e["x"].to_numpy(dtype=float)

        # Leakage
        ax.plot(x, sub_e["leak_mean"], color=COLOR_LEAKAGE, lw=LW_DELEXP,
                alpha=ALPHA_DELEXP_LINE, label="Leakage (delexp)")
        ax.fill_between(x, sub_e["leak_low"], sub_e["leak_high"],
                        color=COLOR_LEAKAGE, alpha=ALPHA_CI_DELEXP)

        # ΔU
        base_txt = "min" if DELTA_U_BASELINE.lower() != "max" else "max"
        ax.plot(x, sub_e["d_mean"], color=COLOR_DELTAU, lw=LW_DELEXP,
                alpha=ALPHA_DELEXP_LINE, label=f"ΔUtility (delexp, vs {base_txt})")
        ax.fill_between(x, sub_e["d_low"], sub_e["d_high"],
                        color=COLOR_DELTAU, alpha=ALPHA_CI_DELEXP)

        # Mask
        yM = sub_e["mask_mean"].to_numpy(dtype=float)
        loM = sub_e["mask_low"].to_numpy(dtype=float)
        hiM = sub_e["mask_high"].to_numpy(dtype=float)
        if MASK_LOG_Y:
            yM = np.maximum(yM, MASK_LOG_EPS)
            loM = np.maximum(loM, MASK_LOG_EPS)
            hiM = np.maximum(hiM, MASK_LOG_EPS)

        ax2.plot(x, yM, color=COLOR_MASK, lw=LW_DELEXP,
                 alpha=ALPHA_DELEXP_LINE, label="Mask Size (delexp)")
        ax2.fill_between(x, loM, hiM, color=COLOR_MASK, alpha=ALPHA_CI_DELEXP)

        if SHOW_MEAN_POINTS:
            ax.scatter(x, sub_e["leak_mean"], s=POINT_SIZE, marker=MARKER_DELEXP,
                       facecolors="none", edgecolors=COLOR_LEAKAGE, linewidths=1.2, zorder=5)
            ax.scatter(x, sub_e["d_mean"], s=POINT_SIZE, marker=MARKER_DELEXP,
                       facecolors="none", edgecolors=COLOR_DELTAU, linewidths=1.2, zorder=5)
            ax2.scatter(x, yM, s=POINT_SIZE, marker=MARKER_DELEXP,
                        facecolors="none", edgecolors=COLOR_MASK, linewidths=1.2, zorder=5)

    # ----- delmin black dot ON right y-axis spine (not inside plot) -----
    delmin_mask = get_delmin_mask(ds)
    if np.isfinite(delmin_mask):
        y0 = delmin_mask
        if MASK_LOG_Y:
            y0 = max(y0, MASK_LOG_EPS)

        ax2.scatter(
            [1.0], [y0],                      # x=1.0 => right y-axis spine in axis coords
            transform=ax2.get_yaxis_transform(),  # x in axes coords, y in data coords
            s=90,
            color="black",
            marker="o",
            zorder=10,
            clip_on=False,
            label="delmin mask size"
        )

    # Labels, grid, title per subplot
    ax.grid(True, alpha=0.22)
    ax.set_xlabel(x_label)
    ax.set_title(pretty_name(ds))

    # Keep y labels off most subplots to reduce clutter; we’ll label only leftmost/rightmost via outer text.
    return ax2


# ============================================================
# Main: build 3x5 grid (epsilon row, alpha row, beta row)
# ============================================================
def main():
    families = ["epsilon", "alpha", "beta"]

    # Try to load + compute summaries for each family and each method
    summaries = {}  # summaries[(family, method)] = df
    for fam in families:
        for method in ["delgum", "delexp"]:
            path = FILES[fam][method]
            try:
                raw = parse_tempdata(path)
                summaries[(fam, method)] = compute_summary_table(raw)
                print(f"[OK] Loaded {fam}/{method}: {path}")
            except Exception as e:
                summaries[(fam, method)] = None
                print(f"[WARN] Skipping {fam}/{method} ({path}): {e}")

    fig, axes = plt.subplots(3, 5, figsize=(24, 10.5), sharex=False, sharey=False)

    # Create the grid
    for r, fam in enumerate(families):
        x_label = FILES[fam]["x_label"]
        gum = summaries[(fam, "delgum")]
        exp = summaries[(fam, "delexp")]

        for c, ds in enumerate(DATASETS_IN_ORDER):
            ax = axes[r, c]
            ax2 = plot_one_subplot(ax, ds, x_label, gum, exp)

            # only show left y-label on first column
            if c == 0:
                ax.set_ylabel("Leakage / ΔUtility")
            else:
                ax.set_ylabel("")

            # only show right y-label on last column
            if c == 4:
                ax2.set_ylabel("Mask Size")
            else:
                ax2.set_ylabel("")

            # row label on the left side
            if c == 0:
                ax.text(-0.22, 0.5, fam, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=13, fontweight="bold")

    # Build a compact shared legend (upper, smaller)
    # Use proxy artists so the legend is consistent even if some files were skipped.
    import matplotlib.lines as mlines

    proxies = [
        mlines.Line2D([], [], color=COLOR_LEAKAGE, lw=LW_DELGUM, label="Leakage (delgum)",
                      marker=MARKER_DELGUM, markerfacecolor="none", markeredgecolor=COLOR_LEAKAGE),
        mlines.Line2D([], [], color=COLOR_LEAKAGE, lw=LW_DELEXP, alpha=ALPHA_DELEXP_LINE, label="Leakage (delexp)",
                      marker=MARKER_DELEXP, markerfacecolor="none", markeredgecolor=COLOR_LEAKAGE),

        mlines.Line2D([], [], color=COLOR_DELTAU, lw=LW_DELGUM, label="ΔUtility (delgum)",
                      marker=MARKER_DELGUM, markerfacecolor="none", markeredgecolor=COLOR_DELTAU),
        mlines.Line2D([], [], color=COLOR_DELTAU, lw=LW_DELEXP, alpha=ALPHA_DELEXP_LINE, label="ΔUtility (delexp)",
                      marker=MARKER_DELEXP, markerfacecolor="none", markeredgecolor=COLOR_DELTAU),

        mlines.Line2D([], [], color=COLOR_MASK, lw=LW_DELGUM, label="Mask Size (delgum)",
                      marker=MARKER_DELGUM, markerfacecolor="none", markeredgecolor=COLOR_MASK),
        mlines.Line2D([], [], color=COLOR_MASK, lw=LW_DELEXP, alpha=ALPHA_DELEXP_LINE, label="Mask Size (delexp)",
                      marker=MARKER_DELEXP, markerfacecolor="none", markeredgecolor=COLOR_MASK),

        mlines.Line2D([], [], color="black", lw=0, marker="o", markersize=7, label="delmin mask size"),
    ]

    fig.legend(
        handles=proxies,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=7,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.6,
    )

    fig.suptitle(
        f"delgum (solid, ◯) vs delexp (solid thinner, △) — mean ± {int(CI_LEVEL*100)}% CI — Mask y-max={MASK_Y_MAX}",
        y=0.995,
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.965])

    # Save to one PDF page (grid)
    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"Wrote: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
