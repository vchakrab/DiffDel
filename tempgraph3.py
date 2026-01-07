#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT_FILE = "leakage_data/ablation_delgum_epsilon.csv"
OUTPUT_PDF = "delgum_epsilon_per_dataset_with_ci_colored_MAP.pdf"

DATASETS_IN_ORDER = ["airport", "hospital", "ncvoter", "Onlineretail", "adult"]

# -----------------------------
# Controls
# -----------------------------
USE_LOG_X = False
LOG_X_MIN = None

SHOW_MEAN_POINTS = True
POINT_SIZE = 22

CI_LEVEL = 0.95
USE_T_DISTRIBUTION = True  # uses scipy if available; falls back to 1.96

MASK_LOG_Y = False
MASK_LOG_EPS = 1e-9

# Delmin flat mask-size baselines (blue dashed)
AIRPORT_DELMIN_MASK = 15
HOSPITAL_DELMIN_MASK = 12
NCVOTER_DELMIN_MASK = 14
ONLINE_RETAILER_DELMIN_MASK = 8
ADULT_DELMIN_MASK = 6

# -----------------------------
# Fixed colors (as requested)
# -----------------------------
COLOR_LEAKAGE = "red"
COLOR_UTILITY = "green"
COLOR_MASK = "blue"
COLOR_MAP = "black"

# MAP settings
MAP_STABILIZER_TAU = 1e-9   # prevents division by ~0
MAP_SMOOTH = True           # smooth MAP curve lightly
MAP_SMOOTH_WINDOW = 3       # odd integer (3/5/7). small = keep shape

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.variant": "small-caps",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.fontsize": 12,
})


def parse_tempdata(path: str) -> pd.DataFrame:
    rows = []
    current_ds_idx = -1
    current_ds = None
    saw_any_header = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # repeated CSV header indicates next dataset block in fixed order
            if line.lower().startswith("epsilon,"):
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
                eps = float(parts[0])
                leakage = float(parts[1])
                utility = float(parts[2])
                mask = float(parts[3])
            except ValueError:
                continue

            rows.append((current_ds, eps, leakage, utility, mask))

    if not saw_any_header or not rows:
        raise ValueError("Failed to parse input file.")

    return pd.DataFrame(rows, columns=["dataset", "epsilon", "leakage", "utility", "mask_size"])


def pretty_name(ds: str) -> str:
    if ds.lower() == "ncvoter":
        return "NCVoter"
    if ds.lower() == "onlineretail":
        return "Onlineretail"
    return ds.capitalize()


def delmin_mask_for(ds: str) -> float:
    d = ds.lower()
    if d == "airport":
        return AIRPORT_DELMIN_MASK
    if d == "hospital":
        return HOSPITAL_DELMIN_MASK
    if d == "ncvoter":
        return NCVOTER_DELMIN_MASK
    if d == "onlineretail":
        return ONLINE_RETAILER_DELMIN_MASK
    if d == "adult":
        return ADULT_DELMIN_MASK
    return np.nan


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
    g = (
        raw.groupby(["dataset", "epsilon"], as_index=False)
           .agg(mean=(col, "mean"), std=(col, "std"), n=(col, "size"))
           .sort_values(["dataset", "epsilon"])
           .reset_index(drop=True)
    )

    g["std"] = g["std"].fillna(0.0)
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))

    alpha = 1.0 - CI_LEVEL
    tvals = [t_critical(alpha, int(n - 1)) for n in g["n"]]
    half = np.array(tvals) * g["sem"]

    g["low"] = g["mean"] - half
    g["high"] = g["mean"] + half
    return g


def rolling_smooth(y: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=1).median().to_numpy()


def compute_map_curve(sub: pd.DataFrame) -> pd.DataFrame:
    """
    sub: one dataset, sorted by epsilon ascending.
    Returns DataFrame with epsilon_mid and MAP values.
    MAP_i = (L_{i-1}-L_i) / (|M_{i-1}-M_i| + tau)
    using mean series.
    """
    sub = sub.sort_values("epsilon").reset_index(drop=True)
    e = sub["epsilon"].to_numpy(dtype=float)
    L = sub["leak_mean"].to_numpy(dtype=float)
    M = sub["mask_mean"].to_numpy(dtype=float)

    if len(e) < 2:
        return pd.DataFrame(columns=["epsilon_mid", "map"])

    dL = L[:-1] - L[1:]
    dM = M[:-1] - M[1:]
    mp = dL / (np.abs(dM) + MAP_STABILIZER_TAU)

    # place the MAP value between the two eps points for plotting
    e_mid = 0.5 * (e[:-1] + e[1:])

    if MAP_SMOOTH and len(mp) >= 3:
        mp = rolling_smooth(mp, MAP_SMOOTH_WINDOW)

    return pd.DataFrame({"epsilon_mid": e_mid, "map": mp})


def main():
    raw = parse_tempdata(INPUT_FILE)

    if USE_LOG_X:
        raw = raw[raw["epsilon"] > 0].copy()

    # plot absolute utility as requested
    raw["utility"] = raw["utility"].abs()

    L = mean_ci(raw, "leakage").rename(columns={"mean": "leak_mean", "low": "leak_low", "high": "leak_high"})
    U = mean_ci(raw, "utility").rename(columns={"mean": "util_mean", "low": "util_low", "high": "util_high"})
    M = mean_ci(raw, "mask_size").rename(columns={"mean": "mask_mean", "low": "mask_low", "high": "mask_high"})

    g = (
        L.merge(U[["dataset", "epsilon", "util_mean", "util_low", "util_high"]], on=["dataset", "epsilon"])
         .merge(M[["dataset", "epsilon", "mask_mean", "mask_low", "mask_high"]], on=["dataset", "epsilon"])
         .sort_values(["dataset", "epsilon"])
         .reset_index(drop=True)
    )

    with PdfPages(OUTPUT_PDF) as pdf:
        for ds in DATASETS_IN_ORDER:
            sub = g[g["dataset"] == ds].copy()
            if sub.empty:
                continue

            x = sub["epsilon"].to_numpy(dtype=float)

            # ==========================
            # Page 1: main plot with bands
            # ==========================
            fig, ax = plt.subplots(figsize=(9.2, 5.2))
            ax2 = ax.twinx()

            # Leakage (red)
            ax.plot(x, sub["leak_mean"], color=COLOR_LEAKAGE, lw=2.5, label="Leakage")
            ax.fill_between(x, sub["leak_low"], sub["leak_high"], color=COLOR_LEAKAGE, alpha=0.20)

            # |Utility| (green)
            ax.plot(x, sub["util_mean"], color=COLOR_UTILITY, lw=2.5, linestyle="-.", label="|Utility|")
            ax.fill_between(x, sub["util_low"], sub["util_high"], color=COLOR_UTILITY, alpha=0.18)

            # Mask size (blue) on ax2
            yM = sub["mask_mean"].to_numpy(dtype=float)
            loM = sub["mask_low"].to_numpy(dtype=float)
            hiM = sub["mask_high"].to_numpy(dtype=float)

            if MASK_LOG_Y:
                yM = np.maximum(yM, MASK_LOG_EPS)
                loM = np.maximum(loM, MASK_LOG_EPS)
                hiM = np.maximum(hiM, MASK_LOG_EPS)

            ax2.plot(x, yM, color=COLOR_MASK, lw=2.5, linestyle="-", label="Mask Size")
            ax2.fill_between(x, loM, hiM, color=COLOR_MASK, alpha=0.18)

            # delmin flat baseline (blue dashed)
            delmin_mask = delmin_mask_for(ds)
            if np.isfinite(delmin_mask):
                ax2.axhline(delmin_mask, color=COLOR_MASK, linestyle="--", lw=2.0, alpha=0.9,
                            label="delmin Mask (flat)")

            # mean points
            if SHOW_MEAN_POINTS:
                ax.scatter(x, sub["leak_mean"], color=COLOR_LEAKAGE, s=POINT_SIZE)
                ax.scatter(x, sub["util_mean"], color=COLOR_UTILITY, s=POINT_SIZE)
                ax2.scatter(x, yM, color=COLOR_MASK, s=POINT_SIZE)

            ax.set_title(f"{pretty_name(ds)} (mean ± {int(CI_LEVEL*100)}% CI)")
            ax.set_xlabel("epsilon")
            ax.set_ylabel("Leakage / |Utility|")
            ax2.set_ylabel("Mask Size")
            ax.grid(True, alpha=0.25)

            if USE_LOG_X:
                ax.set_xscale("log")
                ax2.set_xscale("log")
                if LOG_X_MIN is not None:
                    ax.set_xlim(left=LOG_X_MIN)

            if MASK_LOG_Y:
                ax2.set_yscale("log")

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ==========================
            # Page 2: MAP curve vs epsilon
            # ==========================
            map_df = compute_map_curve(sub)
            if not map_df.empty:
                fig2, axm = plt.subplots(figsize=(9.2, 4.6))
                axm.plot(map_df["epsilon_mid"], map_df["map"], color=COLOR_MAP, lw=2.4, label="MAP = ΔLeakage / |ΔMask|")

                axm.axhline(0.0, color="gray", lw=1.0, alpha=0.5)

                axm.set_title(f"{pretty_name(ds)} — MAP curve along ε")
                axm.set_xlabel("epsilon (midpoint between steps)")
                axm.set_ylabel("MAP (leakage reduction per unit mask change)")
                axm.grid(True, alpha=0.25)

                if USE_LOG_X:
                    axm.set_xscale("log")
                    if LOG_X_MIN is not None:
                        axm.set_xlim(left=LOG_X_MIN)

                axm.legend(loc="best")
                fig2.tight_layout()
                pdf.savefig(fig2)
                plt.close(fig2)

    print(f"Wrote: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
