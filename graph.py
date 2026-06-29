import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ── Base font size ────────────────────────────────────────────────────────────
FS = 13

# ── Unified font sizes for ALL split / ablation plots ────────────────────────
# Every function that produces a 1-row or 2-row "split" figure reads from
# these four constants so titles, labels, ticks, and legends are identical
# regardless of which function created the figure.
SPLIT_FS      = 17    # dataset titles & axis labels
SPLIT_TICK_FS = 15    # tick-mark numbers (x and y)
SPLIT_LEG_FS  = 16    # legend entries
SPLIT_SUP_FS  = 15    # fig.supxlabel / fig.supylabel

# ── Global rcParams — full LaTeX rendering ────────────────────────────────────
# text.usetex passes every string to your local LaTeX installation, enabling
# \textsc{}, \textbf{}, \emph{}, \mathbb{}, etc.
# Requires: latex, dvipng (or dvisvgm), ghostscript on PATH.
PANEL_COLOR = "#E8E8E8"   # grey shade
PANEL_ALPHA = 0.55        # 0 = invisible, 1 = fully opaque
PANEL_PAD   = 0.012       # breathing room around each axis pair

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
    "axes.grid":             True,
    "grid.alpha":            0.3,
})


# ── Dataset / method constants ────────────────────────────────────────────────
DATASETS_5 = ["airport", "hospital", "adult", "flight", "tax"]
DATASETS_3 = ["airport", "hospital", "flight"]

MIN_MASK = {"airport": 5, "hospital": 9, "adult": 9, "flight": 11, "tax": 3}

PASTEL_COLORS = ["#FFF8F0", "#F0F7EC", "#D4EDDA", "#A8D8B9",
                 "#7BC89C", "#4DAF7A", "#2D8B57"]
PASTEL_CMAP = LinearSegmentedColormap.from_list("pastel_green", PASTEL_COLORS, N=256)

TAU_CONTOURS = [0.15, 0.32, 0.52, 0.73]
TAU_COLORS   = {0.15: "#d62728", 0.32: "#0072B2", 0.52: "#9467bd", 0.73: "#17BECF"}

DATASET_ORDER = ["Airport", "Hospital", "Adult", "Flight", "Tax"]
METHOD_ORDER  = ["DelMin", "DelExp", "DelMarg"]
METHOD_LABEL  = {"DelMin": "Min", "DelExp": "Exp", "DelMarg": "Gum"}

PHASE_COLORS = {"Instantiation": "#6F6F6F", "Modeling": "#A0A0A0", "Update Masks": "#3F3F3F"}
PHASE_HATCH  = {"Instantiation": "///", "Modeling": "\\\\", "Update Masks": "..."}
ZONE_COLOR_LIGHT = "#D2B48C"
ZONE_COLOR_DARK  = "#8B4513"


def tau_fn(eps_m, L0):
    a = np.exp(eps_m) * L0
    return a / ((1 - L0) + a)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_heatmap_data() -> pd.DataFrame:
    required_cols = ["dataset", "mask_size", "leakage", "epsilon_m", "L0"]

    def load_folder(folder_path, mech_name):
        frames = []
        for csv_file in folder_path.rglob("*.csv"):
            df = pd.read_csv(csv_file)
            if not all(c in df.columns for c in required_cols):
                continue
            df = df[df["epsilon_m"] > 0].copy()
            df["mechanism"] = mech_name
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    frames = [load_folder(DATA_DIR / "exp", "Exp"),
              load_folder(DATA_DIR / "gum", "Gum")]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["min_mask"] = df["dataset"].map(MIN_MASK)
    return df


def load_curves_data(datasets) -> pd.DataFrame:
    records = []
    for method in ["exp", "gum"]:
        for dataset in datasets:
            path = DATA_DIR / method / dataset / "full_data.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df["epsilon_m"] != 0]
            if df.empty:
                continue
            delmin = MIN_MASK[dataset]
            for (eps, L0), grp in df.groupby(["epsilon_m", "L0"]):
                n            = len(grp)
                mean_mask    = grp["mask_size"].mean()
                mean_leakage = grp["leakage"].mean()
                std_mask     = grp["mask_size"].std()
                std_leakage  = grp["leakage"].std()
                ci_mask      = 1.96 * std_mask    / np.sqrt(n)
                ci_leakage   = 1.96 * std_leakage / np.sqrt(n)

                # ── debug print ──────────────────────────────────────────
                if abs(L0 - 0.2) < 1e-9 and abs(eps - 0.1) < 1e-9:
                # ─────────────────────────────────────────────────────────

                records.append({
                    "method":         method,
                    "dataset":        dataset,
                    "epsilon_m":      eps,
                    "L0":             L0,
                    "improvement":    100 * abs(delmin - mean_mask) / delmin,
                    "ci_improvement": 100 * ci_mask / delmin,
                    "mean_leakage":   mean_leakage,
                    "ci_leakage":     ci_leakage,
                })
    return pd.DataFrame(records)


def load_main_data() -> pd.DataFrame:
    dfs = []
    for dataset in [d.lower() for d in DATASET_ORDER]:
        path = DATA_DIR / "min" / dataset / "full_data.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["dataset"] = dataset.capitalize()
            df["method"]  = "DelMin"
            dfs.append(df)
    for folder, method in [("exp", "DelExp"), ("gum", "DelMarg")]:
        for dataset in [d.lower() for d in DATASET_ORDER]:
            path = DATA_DIR / folder / dataset / "full_data.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[(df["epsilon_m"].round(5) == 0.1) & (df["L0"].round(5) == 0.2)]
            if not df.empty:
                df["dataset"] = dataset.capitalize()
                df["method"]  = method
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["init_time_ms"]   = df["init_time"]  * 1000.0
    df["model_time_ms"]  = df["model_time"] * 1000.0
    df["update_time_ms"] = df["del_time"]   * 1000.0
    df["time_ms"]        = df["init_time_ms"] + df["model_time_ms"] + df["update_time_ms"]
    df["memory_kb"]      = df["memory_overhead_bytes"] / 1024.0
    denom = df["num_instantiated_cells"].clip(lower=1.0)
    df["deletion_ratio"] = (df["mask_size"] / denom).clip(0.0, 1.0)
    return df


# ── All-methods comparison loader ─────────────────────────────────────────────

def load_all_methods_data(csv_path: str = "all_masks_all_methods.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["dataset"] = df["dataset"].str.capitalize()
    return df


# ── Runtime data loader ───────────────────────────────────────────────────────

def load_runtime_data() -> pd.DataFrame:
    """Load data exclusively for the runtime breakdown figure.

    Reads from data/release_data/main_data/ only — completely independent
    of load_main_data() and load_curves_data().

    Folders:
        main_data/min/<dataset>/full_data.csv    — all rows, no filter
        main_data/exp/<dataset>/full_data.csv    — all rows, no filter
        main_data/gumbel/<dataset>/full_data.csv — all rows, no filter
    """
    MAIN_DATA_DIR = DATA_DIR / "main_data"

    FOLDER_METHOD = [
        ("min",    "DelMin"),
        ("exp",    "DelExp"),
        ("gumbel", "DelMarg"),
    ]

    dfs = []
    for folder, method in FOLDER_METHOD:
        for dataset in [d.lower() for d in DATASET_ORDER]:
            path = MAIN_DATA_DIR / folder / dataset / "full_data.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["dataset"] = dataset.capitalize()
            df["method"]  = method
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Timing columns are in seconds → convert to ms
    df["init_time_ms"]   = df["init_time"]  * 1000.0
    df["model_time_ms"]  = df["model_time"] * 1000.0
    df["update_time_ms"] = df["del_time"]   * 1000.0

    denom = df["num_instantiated_cells"].clip(lower=1.0)
    df["deletion_ratio"] = (df["mask_size"] / denom).clip(0.0, 1.0)

    return df


# ── Ablation data loader ──────────────────────────────────────────────────────

def load_ablation_data(
    ablation_dir=None,
    datasets: list = DATASETS_5,
) -> pd.DataFrame:
    if ablation_dir is None:
        ablation_dir = DATA_DIR / "gum_score_ablation"
    ablation_dir = Path(ablation_dir)
    records = []
    for variant in ["random", "zero"]:
        for dataset in datasets:
            path = ablation_dir / variant / dataset / "full_data.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df["epsilon_m"] != 0]
            if df.empty:
                continue
            delmin = MIN_MASK[dataset]
            for (eps, L0), grp in df.groupby(["epsilon_m", "L0"]):
                n            = len(grp)
                mean_mask    = grp["mask_size"].mean()
                mean_leakage = grp["leakage"].mean()
                std_mask     = grp["mask_size"].std(ddof=1) if n > 1 else 0.0
                std_leakage  = grp["leakage"].std(ddof=1)  if n > 1 else 0.0
                ci_mask      = 1.96 * std_mask    / np.sqrt(n)
                ci_leakage   = 1.96 * std_leakage / np.sqrt(n)
                records.append({
                    "variant":        variant,
                    "dataset":        dataset,
                    "epsilon_m":      eps,
                    "L0":             L0,
                    "improvement":    100 * abs(delmin - mean_mask) / delmin,
                    "ci_improvement": 100 * ci_mask / delmin,
                    "mean_leakage":   mean_leakage,
                    "ci_leakage":     ci_leakage,
                })
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE FUNCTIONS
#  Naming convention:
#    *_paper     — 3-dataset figures that appear in the main paper body
#    *_appendix  — 5-dataset figures that appear in the appendix
#    runtime_paper is the one exception (no dataset-count suffix needed)
# ════════════════════════════════════════════════════════════════════════════

# FIGURE 8 — Sensitivity of Exp and Gum to dependency leakage threshold and
#             mask-pattern privacy budget (paper, 3 datasets, 2-row × 6-col)
def sensitivity_paper(df: pd.DataFrame) -> plt.Figure:
    """Combined mask-size improvement (row 0) and achieved leakage (row 1)
    for the three paper datasets (airport / hospital / flight), with both
    mechanisms (Exp left, Gum right) shown as six column panels.
    """
    # Per-dataset panel colours — order matches `ordered` list below:
    # [airport, hospital, flight, airport, hospital, flight]
    PANEL_COLORS = [
        "#E1EBF8",  # col 0 — Airport
        "#E4F3E4",  # col 1 — Hospital
        "#EBE1F5",  # col 2 — Flight
        "#E1EBF8",  # col 3 — Airport  (Gum repeat)
        "#E4F3E4",  # col 4 — Hospital (Gum repeat)
        "#EBE1F5",  # col 5 — Flight   (Gum repeat)
    ]
    PANEL_ALPHA = 0.75
    PANEL_PAD_X = 0.006        # horizontal margin between columns
    PANEL_PAD_Y = 0.038        # vertical margin above/below axes
    PANEL_PAD_X_LEFT_COL0 = 0.016  # col 0 extra left room for y-tick numbers & ylabel

    ordered = [(ds, "exp") for ds in DATASETS_3] + [(ds, "gum") for ds in DATASETS_3]

    fig, axes = plt.subplots(
        2, 6,
        figsize=(19.8, 6.5),
        sharey="row",
        sharex=False,
    )

    plt.subplots_adjust(
        top=0.87, bottom=0.10,
        left=0.06, right=0.98,
        wspace=0.30, hspace=0.18,
    )

    # ── Row 0: mask-size improvement ─────────────────────────────────────────
    for i, (dataset, mech) in enumerate(ordered):
        ax     = axes[0, i]
        subset = df[(df["dataset"] == dataset) & (df["method"] == mech)]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(curve["L0"], curve["improvement"],
                    marker="o", label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(
                curve["L0"],
                curve["improvement"] - curve["ci_improvement"],
                curve["improvement"] + curve["ci_improvement"],
                alpha=0.18,
            )

        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.set_title(
            rf"{dataset.capitalize()} ($\textsc{{{mech_label}}}$)",
            pad=5, fontsize=FS + 4,
        )
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        if i == 0:
            ax.set_ylabel(r"Mask-Size Reduction" + "\n" + r"M.R. (\%)", fontsize=18)

    # ── Row 1: achieved leakage ───────────────────────────────────────────────
    for i, (dataset, mech) in enumerate(ordered):
        ax     = axes[1, i]
        subset = df[(df["dataset"] == dataset) & (df["method"] == mech)]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve     = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean_leak = 100 * curve["mean_leakage"]
            lower     = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper     = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean_leak, marker="o",
                    label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)

        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])
        if i == 0:
            ax.set_ylabel(r"Achieved Dependency Leakage" + "\n" + r" $\mathcal{L}(M)$ (\%)",
                          fontsize=18, labelpad=10)

    # ── Shared legend ─────────────────────────────────────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=SPLIT_LEG_FS,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
    )

    # ── Shared x-axis label ───────────────────────────────────────────────────
    fig.supxlabel(
        r"Dependency Leakage Threshold $L_0$",
        y=0.01,
        fontsize=SPLIT_SUP_FS + 4,
    )

    # ── Cabinet-panel shading — one rounded rect per column spanning both rows
    # Must call after subplots_adjust so axis positions are final.
    fig.canvas.draw()

    for col in range(6):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        bb_top = ax_top.get_position()
        bb_bot = ax_bot.get_position()

        pad_left = PANEL_PAD_X_LEFT_COL0 if col == 0 else PANEL_PAD_X

        x0 = bb_top.x0 - pad_left
        y0 = bb_bot.y0 - PANEL_PAD_Y
        x1 = bb_top.x1 + PANEL_PAD_X
        y1 = bb_top.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS[col],
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 9 — Pareto frontier: leakage vs mask size (paper, 3 datasets)
def pareto_frontier_paper(df: pd.DataFrame) -> plt.Figure:
    """Leakage–mask-size trade-off for airport / hospital / flight, with two
    ε_m values and two mechanisms (Exp / Gum), plus reference points for
    M_MIN and the empty mask.
    """
    DATASETS_3_PARETO = ["airport", "hospital", "flight"]
    EM_VALS   = [0.1, 1.0]
    EPS_COLOR = {1.0: "#d62728", 0.1: "#1f77b4"}
    TARGET_ONLY_LEAK = {
        "airport":  0.49830908053361767,
        "hospital": 0.6623992294064142,
        "adult":    0.49122554744404323,
        "flight":   0.9826408586840785,
        "tax":      0.5548418449270073,
    }
    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "flight":   "#EBE1F5",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.015
    PANEL_PAD_Y       = 0.09
    PANEL_PAD_X_LEFT0 = 0.043   # col 0 extra left room for y-tick numbers & ylabel
    PANEL_PAD_X_REST  = 0.015

    agg = (
        df[df["epsilon_m"].isin(EM_VALS)]
        .groupby(["dataset", "mechanism", "epsilon_m", "L0"])
        .agg(mean_mask=("mask_size", "mean"), mean_leak=("leakage", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for col, ds in enumerate(DATASETS_3_PARETO):
        ax       = axes[col]
        baseline = MIN_MASK[ds]
        ax.tick_params(axis="y", left=True, labelleft=(col == 0))
        ax.set_title(rf"{ds.capitalize()}", pad=5, fontsize=FS + 6)
        for mech in ["Exp", "Gum"]:
            for em in EM_VALS:
                sub = agg[(agg.dataset == ds) & (agg.mechanism == mech) &
                          (agg.epsilon_m == em)].sort_values("L0")
                if sub.empty:
                    continue
                mask_pct     = 100 * sub["mean_mask"] / baseline
                marker_style = "o" if em == 1.0 else "x"
                marker_face  = "none" if em == 1.0 else None
                line_style   = "-" if mech == "Exp" else ":"
                ax.plot(sub["mean_leak"], mask_pct,
                        marker=marker_style, linestyle=line_style,
                        color=EPS_COLOR[em], markerfacecolor=marker_face,
                        markeredgewidth=1.8, markersize=6, linewidth=1.8,
                        label=rf"$\textsc{{{mech}}}$, $\varepsilon_m={em}$")
        ax.scatter(0, 100, marker="*", s=260, facecolor="black", edgecolor="black",
                   linewidth=1.2, zorder=7,
                   label=r"$M_{\mathrm{MIN}}\,(M_{\mathrm{det}})$" if col == 0 else None)
        ax.scatter(TARGET_ONLY_LEAK[ds], 0, marker="D", s=120,
                   facecolor="black", edgecolor="black", zorder=6,
                   label=r"$M = \emptyset$" if col == 0 else None)
        max_leak    = max(agg[agg.dataset == ds]["mean_leak"].max(), TARGET_ONLY_LEAK[ds])
        rounded_max = np.ceil(max_leak / 0.2) * 0.2
        ax.set_xlim(-0.03, rounded_max + 0.04)
        ax.set_xticks(np.arange(0, rounded_max + 0.001, 0.2))
        ax.tick_params(axis="x", labelsize=FS + 2)
        ax.tick_params(axis="y", labelsize=FS + 2)
        ax.set_ylim(-8, 124)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        if col == 0:
            ax.set_ylabel(r"Relative Mask Size" + "\n" + r"(\% of $|M_{\text{det}}|$)",
                          fontsize=FS + 7)
        else:
            ax.set_ylabel(None)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=True,
               bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=FS + 2,
               handlelength=1.2, handletextpad=0.3, columnspacing=0.6)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.2)

    fig.supxlabel(r"Expected Dependency Leakage $\mathbb{E}[\mathcal{L}(M)]$",
                  y=-0.06, fontsize=FS + 7)

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()
    for col, (ax, ds) in enumerate(zip(axes, DATASETS_3_PARETO)):
        bb       = ax.get_position()
        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X_REST

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y

        rect = matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS[ds],
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 10 — Runtime breakdown: phase timing + cell-deletion ratio (paper)
def runtime_paper() -> plt.Figure:
    """1-row × 5-col runtime decomposition with cabinet-panel shading.

    Loads its own data via load_runtime_data() — takes no df argument.

    Left bars  (at x - w): phase timing stacked (Instantiation / Modeling /
                            Update Masks), one bar per method.
    Right bars (at x + w): cell-deletion breakdown (Instantiated Cells /
                            Mask Size), read off ax2 (0-100 %).

    twinx() resets xlim to (0,1) internally, so xlim is applied to BOTH
    ax and ax2 after twinx() is created — this is why Min's bar at x[0]-w
    = -0.25 was being clipped before.
    """
    df = load_runtime_data()
    if df.empty:
        raise RuntimeError(
            "load_runtime_data() returned an empty DataFrame — "
            "check that data/release_data/main_data/ exists and is populated."
        )

    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "adult":    "#F8EAD2",
        "flight":   "#EBE1F5",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA          = 0.75
    PANEL_PAD_X          = 0.006
    PANEL_PAD_Y          = 0.072
    PANEL_PAD_X_LEFT0    = 0.014
    PANEL_PAD_X_LEFT     = 0.022   # left pad for all non-zero columns
    PANEL_PAD_X_RIGHT_4  = 0.020   # right pad for last panel (covers right y-axis labels + ylabel)

    # Per-dataset left y-axis: (ylim_max, yticks, yticklabels)
    # yticklabels=None means show numbers; [] means ticks present but no labels.
    YLIM_CONFIG = {
        "airport":  (10,  [0, 2, 4, 6, 8, 10],        None),
        "hospital": (20,  [0, 5, 10, 15, 20, 25],      None),
        "adult":    (60,  [0, 10, 20, 30, 40, 50, 60], None),
        "flight":   (450, [0, 100, 200, 300, 400],      None),
        "tax":      (10,  [0, 2, 4, 6, 8, 10],         None),
    }

    fig = plt.figure(figsize=(16.5, 3.5))
    subspec = fig.add_gridspec(1, 1)[0, 0].subgridspec(1, 5, wspace=0.35)
    axes  = [fig.add_subplot(subspec[0, i]) for i in range(5)]
    axes2 = []

    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.06, right=0.95)

    for i, dataset in enumerate(DATASET_ORDER):
        ax  = axes[i]
        ddf = df[df["dataset"] == dataset]

        if ddf.empty:
            ax.set_axis_off()
            axes2.append(None)
            continue

        s = (ddf.groupby("method").mean(numeric_only=True)
                 .reindex(METHOD_ORDER).fillna(0.0))
        x      = np.arange(len(METHOD_ORDER))
        w      = 0.25
        bottom = np.zeros(len(METHOD_ORDER))

        # ── Left bars: phase timing ───────────────────────────────────────────
        for key, label in [("init_time_ms",   "Instantiation"),
                            ("model_time_ms",  "Modeling"),
                            ("update_time_ms", "Update Masks")]:
            vals = s[key].values
            ax.bar(x - w, vals, width=w, bottom=bottom,
                   color=PHASE_COLORS[label], hatch=PHASE_HATCH[label],
                   edgecolor="black", linewidth=0.3)
            bottom += vals

        # ── Right bars on twinx ───────────────────────────────────────────────
        # Create twinx BEFORE setting xlim: twinx() resets it to (0, 1).
        ax2 = ax.twinx()
        axes2.append(ax2)

        deleted_pct      = s["deletion_ratio"] * 100.0
        instantiated_pct = 100.0 - deleted_pct

        ax2.bar(x + w, instantiated_pct, width=w,
                color=ZONE_COLOR_LIGHT, edgecolor="black", linewidth=0.3)
        ax2.bar(x + w, deleted_pct, width=w, bottom=instantiated_pct,
                color=ZONE_COLOR_DARK, edgecolor="black", linewidth=0.3)

        ax2.set_ylim(0, 100)
        ax2.set_yticks([0, 20, 40, 60, 80, 100])

        # Right y-axis: only show labels + spine on the last (rightmost) column.
        if i == len(DATASET_ORDER) - 1:
            ax2.tick_params(axis="y", labelsize=SPLIT_TICK_FS, labelright=True)
            ax2.set_ylabel(r"Deleted Cells (\%)", fontsize=FS + 3)
        else:
            ax2.tick_params(axis="y", labelright=False, right=False)
            ax2.spines["right"].set_visible(False)

        # ── xlim applied to BOTH axes after twinx ────────────────────────────
        n    = len(METHOD_ORDER)
        xlim = (-w - w * 0.8, (n - 1) + w + w * 0.8)
        ax.set_xlim(*xlim)
        ax2.set_xlim(*xlim)

        # ── x-ticks centred between each bar pair ─────────────────────────────
        ax.set_xticks(x)
        ax.set_xticklabels(
            [rf"$\textsc{{{METHOD_LABEL[m]}}}$" for m in METHOD_ORDER],
            fontsize=SPLIT_TICK_FS,
        )

        # ── Left y-axis ───────────────────────────────────────────────────────
        ylim_max, yticks, yticklabels = YLIM_CONFIG.get(dataset.lower(), (None, None, None))
        if ylim_max is not None:
            ax.set_ylim(0, ylim_max)
            ax.set_yticks(yticks)
            if yticklabels is not None:
                ax.set_yticklabels(yticklabels)

        ax.tick_params(axis="y", labelsize=SPLIT_TICK_FS, labelleft=True)
        ax.set_title(dataset.capitalize(), pad=5, fontsize=FS + 4)
        if i == 0:
            ax.set_ylabel("Time (ms)", fontsize=FS + 3)

    # ── Legend ────────────────────────────────────────────────────────────────
    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p], hatch=PHASE_HATCH[p],
              edgecolor="black", linewidth=0.3, label=p)
        for p in PHASE_COLORS
    ]
    zone_handles = [
        Patch(facecolor=ZONE_COLOR_LIGHT, edgecolor="black", lw=0.3,
              label="Instantiated Cells"),
        Patch(facecolor=ZONE_COLOR_DARK,  edgecolor="black", lw=0.3,
              label="Mask Size"),
    ]
    fig.legend(
        handles=phase_handles + zone_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(phase_handles) + len(zone_handles),
        fontsize=SPLIT_LEG_FS,
        frameon=True,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
    )

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()

    for i, dataset in enumerate(DATASET_ORDER):
        ax  = axes[i]
        ax2 = axes2[i]
        if ax2 is None:
            continue

        bb_left  = ax.get_position()
        bb_right = ax2.get_position()

        pad_left  = PANEL_PAD_X_LEFT0 if i == 0 else PANEL_PAD_X_LEFT
        pad_right = PANEL_PAD_X_RIGHT_4 if i == len(DATASET_ORDER) - 1 else PANEL_PAD_X

        x0 = bb_left.x0  - pad_left
        y0 = bb_left.y0  - PANEL_PAD_Y
        x1 = bb_right.x1 + pad_right
        y1 = bb_left.y1  + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(dataset.lower(), "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 11 — Heatmap of mask-size reduction vs ε_m and L0 (paper, 3 datasets)
def heatmap_paper(df: pd.DataFrame) -> plt.Figure:
    """1-row × 6-col heatmap (airport / hospital / flight × Exp / Gum) with
    τ iso-contours and per-dataset cabinet-panel shading.

    Y-axis tick labels are suppressed on cols 1-5 (imshow cell lines are
    preserved because yticks are kept); only col 0 shows labels and ylabel.
    """
    PANEL_COLORS = [
        "#E1EBF8",  # col 0 — Airport
        "#E4F3E4",  # col 1 — Hospital
        "#EBE1F5",  # col 2 — Flight
        "#E1EBF8",  # col 3 — Airport  (Gum repeat)
        "#E4F3E4",  # col 4 — Hospital (Gum repeat)
        "#EBE1F5",  # col 5 — Flight   (Gum repeat)
    ]
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.0025   # thin horizontal gap between panels
    PANEL_PAD_Y       = 0.11     # extra room above title / below x-ticks
    PANEL_PAD_X_LEFT0 = 0.022    # col 0 extra left margin to cover y-tick numbers

    datasets = DATASETS_3
    eps_vals = sorted(df["epsilon_m"].unique())
    L0_vals  = sorted(df["L0"].unique())
    L0_plot  = L0_vals[:-1]
    df = df.copy()
    df["improvement"] = 100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
    agg = (df.groupby(["dataset", "mechanism", "epsilon_m", "L0"])["improvement"]
             .mean().reset_index())

    fig = plt.figure(figsize=(19.3, 3.5))
    gs  = GridSpec(1, 6, wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    norm    = Normalize(vmin=0, vmax=75)
    ordered = [(ds, "Exp") for ds in datasets] + [(ds, "Gum") for ds in datasets]

    for i, (ds, mech) in enumerate(ordered):
        ax = axes[i]
        ax.set_title(
            rf"{ds.capitalize()} ($\textsc{{{mech}}}$)",
            pad=6, fontsize=FS + 4,
        )

        sub   = agg[(agg.dataset == ds) & (agg.mechanism == mech)]
        pivot = sub.pivot_table(index="L0", columns="epsilon_m",
                                values="improvement").reindex(
                                    index=L0_plot, columns=eps_vals)
        ax.imshow(pivot.values, cmap=PASTEL_CMAP, norm=norm,
                  origin="lower", aspect="auto")

        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels(eps_vals, rotation=45)

        # Keep yticks on every axis so imshow cell lines are preserved;
        # only suppress the tick *labels* and ylabel on cols 1-5.
        ax.set_yticks(range(len(L0_plot)))
        if i == 0:
            ax.set_yticklabels(L0_plot)
            ax.set_ylabel(r"Dependency Leakage Threshold" + "\n" + r"$L_0$",
                          fontsize=FS + 4)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel(None)

        for tau in TAU_CONTOURS:
            x_coords, y_coords = [], []
            for ei, em in enumerate(eps_vals):
                l0 = tau / (np.exp(em) * (1 - tau) + tau)
                if L0_plot[0] <= l0 <= L0_plot[-1]:
                    yp = np.interp(l0, L0_plot, np.arange(len(L0_plot)))
                    x_coords.append(ei); y_coords.append(yp)
            if len(x_coords) >= 2:
                ax.plot(x_coords, y_coords, linestyle="--", linewidth=2.5,
                        color=TAU_COLORS[tau], zorder=5, clip_on=False)

    sm = ScalarMappable(norm=norm, cmap=PASTEL_CMAP); sm.set_array([])
    cax = fig.add_axes([0.26, 0.915, 0.28, 0.028])
    cb  = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=FS - 1)
    cb.set_label("")
    cax.text(-0.025, 0.5, r"Mask-Size Reduction M.R. (\%)",
             transform=cax.transAxes, va="center", ha="right", fontsize=FS + 3)

    tau_handles = [
        Line2D([0], [0], color=TAU_COLORS[t], linestyle="--",
               linewidth=2.0, label=rf"$\tau$={t}")
        for t in TAU_CONTOURS
    ]
    fig.legend(handles=tau_handles, loc="upper center",
               bbox_to_anchor=(0.7, 0.985), ncol=len(TAU_CONTOURS),
               frameon=True, fontsize=FS + 3, columnspacing=0.8,
               handlelength=1.8, handletextpad=0.4, borderpad=0.3)

    fig.supxlabel(r"Mask-Patten Privacy Budget $\varepsilon_m$", y=-0.04, fontsize=FS + 4)
    plt.subplots_adjust(top=0.75, bottom=0.18)

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()

    for col, ax in enumerate(axes):
        bb       = ax.get_position()
        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y - 0.03
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y - 0.03

        rect = matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS[col],
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 12 (Exp) / FIGURE 13 (Gum) — Sensitivity: mask-size reduction and
# achieved leakage vs L0 for a single mechanism (appendix, 5 datasets,
# 2-row × 5-col)
def sensitivity_appendix(df: pd.DataFrame, mech: str, legend: bool = True) -> plt.Figure:
    """Combined mask-size improvement (row 0) and achieved leakage (row 1)
    for all five datasets, for a single mechanism (exp or gum).
    """
    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "flight":   "#EBE1F5",
        "adult":    "#F8EAD2",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.006
    PANEL_PAD_Y       = 0.038
    PANEL_PAD_X_LEFT0 = 0.021   # col 0 extra left room for y-label

    subset_all = df[df["method"] == mech]

    fig, axes = plt.subplots(
        2, 5,
        figsize=(16.5, 7.1),
        sharey="row",
        sharex=False,
    )

    plt.subplots_adjust(
        top=0.87, bottom=0.10,
        left=0.06, right=0.98,
        wspace=0.30, hspace=0.18,
    )

    # ── Row 0: mask-size improvement ─────────────────────────────────────────
    for col, ds in enumerate(DATASETS_5):
        ax     = axes[0, col]
        subset = subset_all[subset_all["dataset"] == ds]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(curve["L0"], curve["improvement"],
                    marker="o", label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(
                curve["L0"],
                curve["improvement"] - curve["ci_improvement"],
                curve["improvement"] + curve["ci_improvement"],
                alpha=0.18,
            )

        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.set_title(
            rf"{ds.capitalize()} ($\textsc{{{mech_label}}}$)",
            pad=5, fontsize=FS + 4,
        )
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        if col == 0:
            ax.set_ylabel(r"Mask-Size Reduction" + "\n" + r"M.R. (\%)", fontsize=18)

    # ── Row 1: achieved leakage ───────────────────────────────────────────────
    for col, ds in enumerate(DATASETS_5):
        ax     = axes[1, col]
        subset = subset_all[subset_all["dataset"] == ds]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve     = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean_leak = 100 * curve["mean_leakage"]
            lower     = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper     = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean_leak, marker="o",
                    label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)

        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.tick_params(axis="y", pad=10)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])
        if col == 0:
            ax.set_ylabel(
                r"Achieved Dependency Leakage" + "\n" + r" $\mathcal{L}(M)$ (\%)",
                fontsize=18,
                labelpad=10,
            )

    # ── Shared legend ─────────────────────────────────────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if legend:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=len(handles),
            frameon=True,
            bbox_to_anchor=(0.5, 1.01),
            fontsize=SPLIT_LEG_FS,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.4,
        )

    # ── Shared x-axis label ───────────────────────────────────────────────────
    fig.supxlabel(
        r"Dependency Leakage Threshold $L_0$",
        y=0.01,
        fontsize=SPLIT_SUP_FS + 4,
    )

    # ── Cabinet-panel shading — one rect per column spanning both rows ────────
    fig.canvas.draw()

    for col, ds in enumerate(DATASETS_5):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        bb_top = ax_top.get_position()
        bb_bot = ax_bot.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X

        x0 = bb_top.x0 - pad_left
        y0 = bb_bot.y0 - PANEL_PAD_Y
        x1 = bb_top.x1 + PANEL_PAD_X
        y1 = bb_top.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(ds, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 14 — Pareto frontier: leakage vs mask size (appendix, 5 datasets)
def pareto_frontier_appendix(df: pd.DataFrame) -> plt.Figure:
    """Leakage–mask-size trade-off for all five datasets, with two ε_m values
    and two mechanisms (Exp / Gum), plus reference points for M_MIN and the
    empty mask.
    """
    EM_VALS   = [0.1, 1.0]
    EPS_COLOR = {1.0: "#d62728", 0.1: "#1f77b4"}

    TARGET_ONLY_LEAK = {
        "airport":  0.49830908053361767,
        "hospital": 0.6623992294064142,
        "adult":    0.49122554744404323,
        "flight":   0.9826408586840785,
        "tax":      0.5548418449270073,
    }
    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "flight":   "#EBE1F5",
        "adult":    "#F8EAD2",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.005
    PANEL_PAD_Y       = 0.072
    PANEL_PAD_X_LEFT0 = 0.020

    agg = (
        df[df["epsilon_m"].isin(EM_VALS)]
        .groupby(["dataset", "mechanism", "epsilon_m", "L0"])
        .agg(mean_mask=("mask_size", "mean"), mean_leak=("leakage", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False, sharex=False)

    plt.subplots_adjust(
        top=0.78, bottom=0.22,
        left=0.06, right=0.98,
        wspace=0.30,
    )

    for col, ds in enumerate(DATASETS_5):
        ax       = axes[col]
        baseline = MIN_MASK[ds]

        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_title(rf"{ds.capitalize()}", pad=5, fontsize=FS + 4)

        for mech in ["Exp", "Gum"]:
            for em in EM_VALS:
                sub = agg[
                    (agg["dataset"]   == ds)  &
                    (agg["mechanism"] == mech) &
                    (agg["epsilon_m"] == em)
                ].sort_values("L0")
                if sub.empty:
                    continue

                mask_pct     = 100 * sub["mean_mask"] / baseline
                marker_style = "o" if em == 1.0 else "x"
                marker_face  = "none" if em == 1.0 else None
                line_style   = "-"   if mech == "Exp" else ":"

                ax.plot(
                    sub["mean_leak"], mask_pct,
                    marker=marker_style,
                    linestyle=line_style,
                    color=EPS_COLOR[em],
                    markerfacecolor=marker_face,
                    markeredgewidth=1.8,
                    markersize=6,
                    linewidth=1.8,
                    label=rf"$\textsc{{{mech}}}$, $\varepsilon_m={em}$",
                )

        ax.scatter(0, 100, marker="*", s=260,
                   facecolor="black", edgecolor="black",
                   linewidth=1.2, zorder=7,
                   label=(r"$M_{\mathrm{MIN}}\,(M_{\mathrm{det}})$" if col == 0 else None))
        ax.scatter(TARGET_ONLY_LEAK[ds], 0, marker="D", s=120,
                   facecolor="black", edgecolor="black", zorder=6,
                   label=(r"$M = \emptyset$" if col == 0 else None))

        max_leak    = max(agg[agg["dataset"] == ds]["mean_leak"].max(),
                         TARGET_ONLY_LEAK[ds])
        rounded_max = np.ceil(max_leak / 0.2) * 0.2
        ax.set_xlim(-0.03, rounded_max + 0.04)
        ax.set_xticks(np.arange(0, rounded_max + 0.001, 0.2))
        ax.set_ylim(-8, 124)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        # Y-axis labels: left-most column only
        if col == 0:
            ax.set_ylabel(
                r"Relative Mask Size" + "\n" + r"(\% of $|M_{\text{det}}|$)",
                fontsize=FS + 3,
            )
            ax.tick_params(axis="y", left=True, labelleft=True)
        else:
            ax.tick_params(axis="y", left=True, labelleft=False)

    # ── Shared legend ─────────────────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        fontsize=SPLIT_LEG_FS,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
    )

    # ── Shared x-axis label ───────────────────────────────────────────────────
    fig.supxlabel(
        r"Expected Dependency Leakage $\mathbb{E}[\mathcal{L}(M)]$",
        y=0.01,
        fontsize=SPLIT_SUP_FS + 4,
    )

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()

    for col, ds in enumerate(DATASETS_5):
        ax = axes[col]
        bb = ax.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(ds, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 16 — Dependency leakage vs mask size for three leakage estimators
#             (appendix, 5 datasets)
def leakage_models_appendix(
    csv_path: str = "all_masks_all_methods.csv",
) -> plt.Figure:
    """Compares three leakage-estimation methods (Nor / This Work / Max) across
    all five datasets as mask size increases.
    """
    METHODS = [
        ("leakage_noisy_or",        r"\textsc{Nor}",       "#d62728", "-",  "o"),
        ("leakage_greedy_disjoint", r"\textsc{This Work}", "#1f77b4", "--", "o"),
        ("leakage_max",             r"\textsc{Max}",       "#2ca02c", ":",  "o"),
    ]
    BAND_ALPHA = 0.15
    LINE_W     = 1.8
    MARKER_SZ  = 6

    PANEL_COLORS = {
        "Airport":  "#E1EBF8",
        "Hospital": "#E4F3E4",
        "Adult":    "#F8EAD2",
        "Flight":   "#EBE1F5",
        "Tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.006
    PANEL_PAD_Y       = 0.072
    PANEL_PAD_X_LEFT0 = 0.020   # col 0 — covers y-tick numbers + ylabel
    PANEL_PAD_X_LEFT  = 0.010   # all other columns

    df = load_all_methods_data(csv_path)
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(
        wspace=0.30, left=0.07, right=0.97,
        top=0.78, bottom=0.22,
    )

    for col, dataset in enumerate(DATASET_ORDER):
        ax  = axes[col]
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            ax.set_axis_off()
            continue

        x_vals = sorted(sub["mask_size"].unique())
        xs_arr = np.array(x_vals, dtype=float)

        for col_name, label, color, ls, marker in METHODS:
            means = np.array([
                100.0 * sub[sub["mask_size"] == xs][col_name].dropna().mean()
                for xs in x_vals
            ])
            stds = np.array([
                100.0 * sub[sub["mask_size"] == xs][col_name].dropna().std()
                for xs in x_vals
            ])
            stds  = np.nan_to_num(stds,  nan=0.0)
            means = np.nan_to_num(means, nan=0.0)
            xs_dense = np.linspace(0, xs_arr[-1], 300)
            k = min(3, len(xs_arr) - 1)
            from scipy.interpolate import make_interp_spline
            spl_mean    = make_interp_spline(xs_arr, means, k=k)
            spl_std     = make_interp_spline(xs_arr, stds,  k=k)
            means_dense = spl_mean(xs_dense)
            stds_dense  = np.clip(spl_std(xs_dense), 0, None)
            ax.fill_between(xs_dense,
                            np.clip(means_dense - stds_dense, 0, 100),
                            np.clip(means_dense + stds_dense, 0, 100),
                            color=color, alpha=BAND_ALPHA, zorder=2)
            ax.plot(xs_arr, means,
                    color=color, linestyle=ls, linewidth=LINE_W,
                    marker=marker, markersize=MARKER_SZ,
                    markerfacecolor=color, markeredgewidth=1.5,
                    label=label, zorder=4)

        ax.set_title(rf"{dataset}", pad=5, fontsize=FS + 4)
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xlim(min(x_vals) - 0.1, max(x_vals) + 0.1)
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Y-axis: left-most column only
        if col == 0:
            ax.set_ylabel(r"Dependency Leakage (\%)", fontsize=18)
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.set_ylabel(None)
            ax.tick_params(axis="y", labelleft=False)

    # ── Shared legend ─────────────────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(METHODS),
        frameon=True,
        fontsize=SPLIT_LEG_FS,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
    )

    # ── Shared x-axis label ───────────────────────────────────────────────────
    fig.supxlabel(r"Mask Size $M$", y=0.01, fontsize=SPLIT_SUP_FS + 4)

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()

    for col, dataset in enumerate(DATASET_ORDER):
        ax = axes[col]
        bb = ax.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X_LEFT

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(dataset, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 17 (a) — Achieved leakage vs L0 for the Gum mechanism
#                 (appendix, 5 datasets, top row of a 3-figure stack)
def leakage_gum_appendix(df: pd.DataFrame, mech: str) -> plt.Figure:
    """1-row × 5-col achieved-leakage curves for the Gum mechanism.
    Carries the shared legend; the two ablation rows below it do not.
    """
    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "adult":    "#F8EAD2",
        "flight":   "#EBE1F5",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.005
    PANEL_PAD_Y       = 0.070
    PANEL_PAD_X_LEFT0 = 0.015
    PANEL_PAD_X_LEFT  = 0.005

    subset_all = df[df["method"] == mech]

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(
        top=0.78, bottom=0.22,
        left=0.07, right=0.97,
        wspace=0.30,
    )

    for col, ds in enumerate(DATASETS_5):
        ax     = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean  = 100 * curve["mean_leakage"]
            lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean, marker="o",
                    label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)

        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)

        ax.set_title(rf"{ds.capitalize()}", pad=5, fontsize=FS + 4)
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])

        if col == 0:
            ax.set_ylabel(
                r"Achieved Dependency Leakage" + "\n" + r"$\mathcal{L}(M)$ (\%)",
                fontsize=18,
            )
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # ── Shared legend (carried by this top row only) ──────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=SPLIT_LEG_FS,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
    )

    fig.supxlabel(
        r"Dependency Leakage Threshold $L_0$",
        fontsize=SPLIT_SUP_FS + 4, y=0.01,
    )

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()

    for col, ds in enumerate(DATASETS_5):
        ax = axes[col]
        bb = ax.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X_LEFT

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(ds, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 17 (b) / (c) — Achieved leakage vs L0 for one ablation variant
#                        (appendix, 5 datasets, no legend — shared with fig 17a)
def leakage_ablation_appendix(
    df: pd.DataFrame,
    variant: str,
    datasets: list = DATASETS_5,
) -> plt.Figure:
    """1-row × 5-col achieved-leakage curves for one Gum ablation variant
    (random or zero score initialisation).  No legend — the shared legend
    lives on leakage_gum_appendix stacked above.
    """
    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "adult":    "#F8EAD2",
        "flight":   "#EBE1F5",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.005
    PANEL_PAD_Y       = 0.070
    PANEL_PAD_X_LEFT0 = 0.015
    PANEL_PAD_X_LEFT  = 0.005

    subset_all = df[df["variant"] == variant]

    fig, axes = plt.subplots(1, len(datasets), figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(
        top=0.78, bottom=0.22,
        left=0.07, right=0.97,
        wspace=0.30,
    )

    for col, ds in enumerate(datasets):
        ax     = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean  = 100 * curve["mean_leakage"]
            lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean, marker="o",
                    label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)

        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)

        ax.set_title(rf"{ds.capitalize()}", pad=5, fontsize=FS + 4)
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])

        if col == 0:
            ax.set_ylabel(
                r"Achieved Dependency Leakage" + "\n" + r"$\mathcal{L}(M)$ (\%)",
                fontsize=18,
            )
            ax.yaxis.set_label_coords(-0.18, 0.38)   # shift label down; x controls distance from axis
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.supxlabel(
        r"Dependency Leakage Threshold $L_0$",
        fontsize=SPLIT_SUP_FS + 4, y=0.01,
    )

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()

    for col, ds in enumerate(datasets):
        ax = axes[col]
        bb = ax.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X_LEFT

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(ds, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 18 (a) — Mask-size reduction vs L0 for the Gum mechanism
#                 (appendix, 5 datasets, top row of a 3-figure stack)
def mask_reduction_appendix(df: pd.DataFrame, mech: str, legend: bool = True) -> plt.Figure:
    """1-row × 5-col mask-size improvement curves for a single mechanism.
    Carries the shared legend when legend=True; the two ablation rows below
    it do not.
    """
    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "adult":    "#F8EAD2",
        "flight":   "#EBE1F5",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.005
    PANEL_PAD_Y       = 0.070
    PANEL_PAD_X_LEFT0 = 0.02
    PANEL_PAD_X_LEFT  = 0.005

    subset_all = df[df["method"] == mech]

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(
        top=0.78, bottom=0.22,
        left=0.07, right=0.97,
        wspace=0.30,
    )

    for col, ds in enumerate(DATASETS_5):
        ax     = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(curve["L0"], curve["improvement"],
                    marker="o", label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(
                curve["L0"],
                curve["improvement"] - curve["ci_improvement"],
                curve["improvement"] + curve["ci_improvement"],
                alpha=0.18,
            )

        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.set_title(
            rf"{ds.capitalize()} ($\textsc{{{mech_label}}}$)",
            pad=5, fontsize=FS + 4,
        )
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        if col == 0:
            ax.set_ylabel(
                r"Mask-Size Reduction" + "\n" + r"M.R. (\%)",
                fontsize=18,
            )
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # ── Shared legend (carried by this top row only) ──────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    if legend:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=len(handles),
            frameon=True,
            bbox_to_anchor=(0.5, 1.01),
            fontsize=SPLIT_LEG_FS,
            columnspacing=0.8,
            handlelength=1.8,
            handletextpad=0.4,
        )

    fig.supxlabel(
        r"Dependency Leakage Threshold $L_0$",
        fontsize=SPLIT_SUP_FS + 4, y=0.01,
    )

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()

    for col, ds in enumerate(DATASETS_5):
        ax = axes[col]
        bb = ax.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X_LEFT

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(ds, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 18 (b) / (c) — Mask-size reduction vs L0 for one ablation variant
#                        (appendix, 5 datasets, no legend — shared with fig 18a)
def mask_ablation_appendix(
    df: pd.DataFrame,
    variant: str,
    datasets: list = DATASETS_5,
) -> plt.Figure:
    """1-row × 5-col mask-size improvement curves for one Gum ablation variant
    (random or zero score initialisation).  No legend — the shared legend
    lives on mask_reduction_appendix stacked above.
    """
    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "adult":    "#F8EAD2",
        "flight":   "#EBE1F5",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.005
    PANEL_PAD_Y       = 0.08
    PANEL_PAD_X_LEFT0 = 0.02
    PANEL_PAD_X_LEFT  = 0.005

    subset_all = df[df["variant"] == variant]

    fig, axes = plt.subplots(1, len(datasets), figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(
        top=0.78, bottom=0.22,
        left=0.07, right=0.97,
        wspace=0.30,
    )

    for col, ds in enumerate(datasets):
        ax     = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]

        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(curve["L0"], curve["improvement"], marker="o",
                    label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(
                curve["L0"],
                curve["improvement"] - curve["ci_improvement"],
                curve["improvement"] + curve["ci_improvement"],
                alpha=0.18,
            )

        ax.set_title(rf"{ds.capitalize()}", pad=5, fontsize=FS + 4)
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        if col == 0:
            ax.set_ylabel(
                r"Mask-Size Reduction" + "\n" + r"M.R. (\%)",
                fontsize=18,
            )
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.supxlabel(
        r"Dependency Leakage Threshold $L_0$",
        fontsize=SPLIT_SUP_FS + 4, y=0.01,
    )

    # ── Cabinet-panel shading ─────────────────────────────────────────────────
    fig.canvas.draw()

    for col, ds in enumerate(datasets):
        ax = axes[col]
        bb = ax.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X_LEFT

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(ds, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 19 — Heatmap of mask-size reduction vs ε_m and L0 (appendix, 5 datasets,
#             2-row × 5-col with shared vertical colorbar)
def heatmap_appendix(df: pd.DataFrame) -> plt.Figure:
    """2-row (Exp / Gum) × 5-col heatmap with τ iso-contours and per-dataset
    cabinet-panel shading spanning both rows.
    """
    df = df.copy()
    df["improvement"] = 100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
    eps_vals = sorted(df["epsilon_m"].unique())
    L0_vals  = sorted(df["L0"].unique())
    L0_plot  = L0_vals[:-1]
    agg = (df.groupby(["dataset", "mechanism", "epsilon_m", "L0"])["improvement"]
             .mean().reset_index())

    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "adult":    "#F8EAD2",
        "flight":   "#EBE1F5",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.005   # right edge + non-col-0 left edge
    PANEL_PAD_Y_TOP   = 0.04    # vertical margin above top row
    PANEL_PAD_Y_BOT   = 0.065   # vertical margin below bottom row (extra for x-tick labels)
    PANEL_PAD_X_LEFT0 = 0.028   # col 0 left — covers y-tick numbers + ylabel
    PANEL_PAD_X_LEFT  = 0.005   # cols 1-4 left

    fig = plt.figure(figsize=(16.5, 6))
    gs  = GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=0.27, hspace=0.3)
    norm = Normalize(vmin=0, vmax=75)

    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.09, right=0.96)

    axes = {}   # (row, col) -> ax, needed for panel bbox calculation

    for row, mech in enumerate(["Exp", "Gum"]):
        for col, ds in enumerate(DATASETS_5):
            ax = fig.add_subplot(gs[row, col])
            axes[(row, col)] = ax

            sub   = agg[(agg.dataset == ds) & (agg.mechanism == mech)]
            pivot = sub.pivot_table(index="L0", columns="epsilon_m",
                                    values="improvement").reindex(
                                        index=L0_plot, columns=eps_vals)
            ax.imshow(pivot.values, cmap=PASTEL_CMAP, norm=norm,
                      origin="lower", aspect="auto")

            ax.set_xticks(range(len(eps_vals)))
            ax.set_xticklabels(eps_vals, rotation=45, fontsize=SPLIT_TICK_FS - 2)
            ax.set_yticks(range(len(L0_plot)))
            if col == 0:
                ax.set_yticklabels(L0_plot, fontsize=SPLIT_TICK_FS - 2)
            else:
                ax.tick_params(axis="y", labelleft=False)

            if row == 0:
                ax.set_title(rf"{ds.capitalize()}", pad=5, fontsize=FS + 4)

            for tau in TAU_CONTOURS:
                xs, ys = [], []
                for ei, em in enumerate(eps_vals):
                    l0 = tau / (np.exp(em) * (1 - tau) + tau)
                    if L0_plot[0] <= l0 <= L0_plot[-1]:
                        yp = np.interp(l0, L0_plot, np.arange(len(L0_plot)))
                        xs.append(ei); ys.append(yp)
                if len(xs) >= 2:
                    ax.plot(xs, ys, linestyle="--", linewidth=2.5,
                            color=TAU_COLORS[tau], zorder=5, clip_on=False)

    # ── Row labels on left side ───────────────────────────────────────────────
    fig.text(0.05, 0.72, r"\textsc{Exp}", ha="center", va="center",
             fontsize=FS + 3, rotation=90)
    fig.text(0.05, 0.28, r"\textsc{Gum}", ha="center", va="center",
             fontsize=FS + 3, rotation=90)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = ScalarMappable(norm=norm, cmap=PASTEL_CMAP)
    sm.set_array([])
    cbar_ax = fig.add_subplot(gs[:, 5])
    cb = plt.colorbar(sm, cax=cbar_ax)
    cb.set_label(r"Mask-Size Reduction M.R. (\%)", fontsize=FS + 2)
    cb.ax.tick_params(labelsize=SPLIT_TICK_FS - 2)

    # ── Shared y/x labels ─────────────────────────────────────────────────────
    fig.supylabel(
        r"Dependency Leakage Threshold $L_0$",
        fontsize=SPLIT_SUP_FS + 2, x=0.02,
    )
    fig.supxlabel(
        r"Mask-Pattern Privacy Budget $\varepsilon_m$",
        fontsize=SPLIT_SUP_FS + 2, y=-0.02,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    tau_handles = [
        Line2D([0], [0], color=TAU_COLORS[t], linestyle="--", linewidth=2.0,
               label=rf"$\tau={t}$")
        for t in TAU_CONTOURS
    ]
    fig.legend(
        handles=tau_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(TAU_CONTOURS),
        frameon=True,
        fontsize=SPLIT_LEG_FS,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.3,
    )

    # ── Cabinet-panel shading — one panel per dataset spanning both rows ──────
    fig.canvas.draw()

    for col, ds in enumerate(DATASETS_5):
        ax_top = axes[(0, col)]
        ax_bot = axes[(1, col)]

        bb_top = ax_top.get_position()
        bb_bot = ax_bot.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X_LEFT

        # Rectangle spans from top of row-0 to bottom of row-1
        x0 = bb_top.x0 - pad_left
        y0 = bb_bot.y0 - PANEL_PAD_Y_BOT
        x1 = bb_top.x1 + PANEL_PAD_X
        y1 = bb_top.y1 + PANEL_PAD_Y_TOP

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(ds, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# FIGURE 20 — Best vs worst privacy-budget split at fixed τ values
#             (appendix, 5 datasets, 2-row × 5-col)
def budget_split_appendix(df: pd.DataFrame) -> plt.Figure:
    """For each dataset and mechanism (Exp / Gum), groups (ε_m, L0) pairs by
    their implied τ value and shows the best and worst mask-size reduction
    achievable at each τ level.
    """
    df = df.copy()
    df["improvement"] = 100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
    agg = (df.groupby(["dataset", "mechanism", "epsilon_m", "L0"])["improvement"]
             .mean().reset_index())
    agg["tau"] = agg.apply(lambda r: tau_fn(r["epsilon_m"], r["L0"]), axis=1)
    tol = 0.04

    PANEL_COLORS = {
        "airport":  "#E1EBF8",
        "hospital": "#E4F3E4",
        "adult":    "#F8EAD2",
        "flight":   "#EBE1F5",
        "tax":      "#F5E1E1",
    }
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.005
    PANEL_PAD_Y_TOP   = 0.04
    PANEL_PAD_Y_BOT   = 0.065
    PANEL_PAD_X_LEFT0 = 0.02
    PANEL_PAD_X_LEFT  = 0.005

    fig, axes = plt.subplots(2, 5, figsize=(16.5, 6.5), sharey="row")
    plt.subplots_adjust(
        hspace=0.30, wspace=0.18,
        left=0.09, right=0.97, top=0.88, bottom=0.10,
    )

    for row, mech in enumerate(["Exp", "Gum"]):
        for col, ds in enumerate(DATASETS_5):
            ax  = axes[row, col]
            sub = agg[(agg.dataset == ds) & (agg.mechanism == mech)]
            valid_taus, best_vals, worst_vals = [], [], []
            for t in TAU_CONTOURS:
                close = sub[np.abs(sub["tau"] - t) <= tol]
                if close.empty:
                    continue
                valid_taus.append(t)
                best_vals.append(close["improvement"].max())
                worst_vals.append(close["improvement"].min())
            if not valid_taus:
                ax.set_axis_off()
                continue
            x = np.arange(len(valid_taus))
            w = 0.35
            ax.bar(x,     worst_vals, w, color="#d62728", edgecolor="black",
                   linewidth=0.5, label="Worst split")
            ax.bar(x + w, best_vals,  w, color="#2ca02c", edgecolor="black",
                   linewidth=0.5, label="Best split")
            ax.set_xticks(x + w / 2)
            ax.set_xticklabels(
                [rf"$\tau\approx{t}$" for t in valid_taus],
                fontsize=SPLIT_TICK_FS - 4,
            )
            ax.tick_params(axis="y", labelsize=SPLIT_TICK_FS - 2)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            if col != 0:
                ax.tick_params(axis="y", labelleft=False)
            if row == 0:
                ax.set_title(rf"{ds.capitalize()}", pad=5, fontsize=FS + 4)

    # ── Row labels ────────────────────────────────────────────────────────────
    fig.text(0.05, 0.72, r"\textsc{Exp}", ha="center", va="center",
             fontsize=FS + 3, rotation=90)
    fig.text(0.05, 0.28, r"\textsc{Gum}", ha="center", va="center",
             fontsize=FS + 3, rotation=90)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        Patch(facecolor="#d62728", edgecolor="black", lw=0.5, label="Worst split"),
        Patch(facecolor="#2ca02c", edgecolor="black", lw=0.5, label="Best split"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=True,
        fontsize=SPLIT_LEG_FS,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
    )

    # ── Shared axis labels ────────────────────────────────────────────────────
    fig.supylabel(
        r"Mask-Size Reduction M.R. (\%)",
        fontsize=SPLIT_SUP_FS + 2, x=0.02,
    )
    fig.supxlabel(
        r"Re-inference Threshold $\tau$",
        fontsize=SPLIT_SUP_FS + 2, y=-0.02,
    )

    # ── Cabinet-panel shading — one panel per dataset spanning both rows ──────
    fig.canvas.draw()

    for col, ds in enumerate(DATASETS_5):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        bb_top = ax_top.get_position()
        bb_bot = ax_bot.get_position()

        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X_LEFT

        x0 = bb_top.x0 - pad_left
        y0 = bb_bot.y0 - PANEL_PAD_Y_BOT
        x1 = bb_top.x1 + PANEL_PAD_X
        y1 = bb_top.y1 + PANEL_PAD_Y_TOP

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS.get(ds, "#F0F0F0"),
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


# ════════════════════════════════════════════════════════════════════════════
#  graph_all_experiments
# ════════════════════════════════════════════════════════════════════════════

def graph_all_experiments():
    FIG_DIR = BASE_DIR / "fig"
    FIG_DIR.mkdir(exist_ok=True)

    df_heat     = load_heatmap_data()
    df_curves_3 = load_curves_data(DATASETS_3)
    df_curves_5 = load_curves_data(DATASETS_5)
    df_abl      = load_ablation_data()

    # FIGURE 8
    fig = sensitivity_paper(df_curves_3)
    fig.savefig(FIG_DIR / "fig8_sensitivity_paper.pdf", bbox_inches="tight")
    plt.close(fig)

    # FIGURE 9
    if not df_heat.empty:
        fig = pareto_frontier_paper(df_heat)
        fig.savefig(FIG_DIR / "fig9_pareto_frontier_paper.pdf", bbox_inches="tight")
        plt.close(fig)

    # FIGURE 10
    fig = runtime_paper()
    fig.savefig(FIG_DIR / "fig10_runtime_paper.pdf", bbox_inches="tight")
    plt.close(fig)

    # FIGURE 11
    if not df_heat.empty:
        fig = heatmap_paper(df_heat)
        fig.savefig(FIG_DIR / "fig11_heatmap_paper.pdf", bbox_inches="tight")
        plt.close(fig)

    # FIGURE 12 (Exp) and FIGURE 13 (Gum)
    for mech in ["exp", "gum"]:
        fig = sensitivity_appendix(df_curves_5, mech)
        fig_num = 12 if mech == "exp" else 13
        fig.savefig(FIG_DIR / f"fig{fig_num}_sensitivity_{mech}_appendix.pdf", bbox_inches="tight")
        plt.close(fig)

    # FIGURE 14
    if not df_heat.empty:
        fig = pareto_frontier_appendix(df_heat)
        fig.savefig(FIG_DIR / "fig14_pareto_frontier_appendix.pdf", bbox_inches="tight")
        plt.close(fig)

    # FIGURE 16
    fig = leakage_models_appendix("data/all_masks_all_methods.csv")
    fig.savefig(FIG_DIR / "fig16_leakage_models_appendix.pdf", bbox_inches="tight")
    plt.close(fig)

    # FIGURE 17 (a)
    fig = leakage_gum_appendix(df_curves_5, "gum")
    fig.savefig(FIG_DIR / "fig17a_leakage_gum_appendix.pdf", bbox_inches="tight")
    plt.close(fig)

    # FIGURE 17 (b), (c)
    for variant in ["random", "zero"]:
        fig = leakage_ablation_appendix(df_abl, variant)
        fig.savefig(FIG_DIR / f"fig17{'b' if variant == 'random' else 'c'}_leakage_{variant}_appendix.pdf",
                    bbox_inches="tight")
        plt.close(fig)

    # FIGURE 18 (a)
    fig = mask_reduction_appendix(df_curves_5, "gum")
    fig.savefig(FIG_DIR / "fig18a_mask_reduction_appendix.pdf", bbox_inches="tight")
    plt.close(fig)

    # FIGURE 18 (b), (c)
    for variant in ["random", "zero"]:
        fig = mask_ablation_appendix(df_abl, variant)
        fig.savefig(FIG_DIR / f"fig18{'b' if variant == 'random' else 'c'}_mask_{variant}_appendix.pdf",
                    bbox_inches="tight")
        plt.close(fig)

    # FIGURE 19
    if not df_heat.empty:
        fig = heatmap_appendix(df_heat)
        fig.savefig(FIG_DIR / "fig19_heatmap_appendix.pdf", bbox_inches="tight")
        plt.close(fig)

    # FIGURE 20
    if not df_heat.empty:
        fig = budget_split_appendix(df_heat)
        fig.savefig(FIG_DIR / "fig20_budget_split_appendix.pdf", bbox_inches="tight")
        plt.close(fig)



if __name__ == '__main__':
    graph_all_experiments()