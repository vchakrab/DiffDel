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
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/release_data"
print(DATA_DIR)

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

# Shared subplots_adjust for ALL 1-row split/ablation figures so panels are
# exactly the same size and position — the only visual difference between
# the legend figure and the no-legend figures is the legend band itself.
SPLIT_ADJUST = dict(wspace=0.30, left=0.07, right=0.97, top=0.82, bottom=0.15)

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


# ── Shared axis-style helper ──────────────────────────────────────────────────

def _apply_split_axis_style(ax, col: int, yticks, ylabel: str):
    """Apply the unified tick / label style to a single split-plot axis."""
    ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlim(0.0, 0.9)
    ax.set_yticks(yticks)
    if col == 0:
        ax.set_ylabel(ylabel, fontsize=SPLIT_FS)
    else:
        ax.set_ylabel(None)


# ── Heatmap figures ───────────────────────────────────────────────────────────

def plot_heatmap_3(df: pd.DataFrame):
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
        ax.set_title(rf"{ds.capitalize()} ($\mathbf{{{mech}}}$)", pad=8, fontsize=FS)
        sub   = agg[(agg.dataset == ds) & (agg.mechanism == mech)]
        pivot = sub.pivot_table(index="L0", columns="epsilon_m",
                                values="improvement").reindex(index=L0_plot, columns=eps_vals)
        ax.imshow(pivot.values, cmap=PASTEL_CMAP, norm=norm, origin="lower", aspect="auto")
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels(eps_vals, rotation=45)
        ax.set_yticks(range(len(L0_plot)))
        ax.set_yticklabels(L0_plot)
        if i == 0:
            ax.set_ylabel(r"Re-inference Leakage Threshold $L_0$", fontsize=FS + 5)
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
    cax.text(-0.025, 0.5, r"Mask Improvement (\%)",
             transform=cax.transAxes, va="center", ha="right", fontsize=FS + 3)
    tau_handles = [
        Line2D([0], [0], color=TAU_COLORS[t], linestyle="--",
               linewidth=2.0, label=rf"$\tau={t}$")
        for t in TAU_CONTOURS
    ]
    fig.legend(handles=tau_handles, loc="upper center",
               bbox_to_anchor=(0.7, 0.985), ncol=len(TAU_CONTOURS),
               frameon=True, fontsize=FS + 3, columnspacing=0.8,
               handlelength=1.8, handletextpad=0.4, borderpad=0.3)
    fig.supxlabel(r"Masking-Privacy Budget $\varepsilon_m$", y=-0.04, fontsize=FS + 5)
    plt.subplots_adjust(top=0.75, bottom=0.18)
    return fig

def plot_heatmap_3_with_panels(df):
    """Heatmap 3 with per-dataset coloured cabinet panels.

    Identical content to plot_heatmap_3. Changes:
      - Y-axis tick labels suppressed on cols 1-5 (ticks kept so imshow
        cell lines are preserved); only col 0 shows labels and ylabel.
      - A FancyBboxPatch panel in each dataset's colour sits behind every
        column (Exp col and Gum col share the same dataset colour).
      - Panel drawn after fig.canvas.draw() so positions are finalised.
    """
    PANEL_COLORS = [
        "#E1EBF8",  # col 0 — Airport  (rowA: 225,235,248)
        "#E4F3E4",  # col 1 — Hospital (rowB: 228,243,228)
        "#EBE1F5",  # col 2 — Flight   (rowD: 235,225,245)
        "#E1EBF8",  # col 3 — Airport
        "#E4F3E4",  # col 4 — Hospital
        "#EBE1F5",  # col 5 — Flight
    ]
    PANEL_ALPHA       = 0.75
    PANEL_PAD_X       = 0.0025   # thin horizontal gap between panels
    PANEL_PAD_Y       = 0.11   # extra room above title / below x-ticks
    PANEL_PAD_X_LEFT0 = 0.022   # col 0 extra left margin to cover y-tick numbers

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
            pad = 6, fontsize = FS+4
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
            ax.set_ylabel(r"Re-inference Leakage Threshold" + "\n" + r"$L_0$",
                          fontsize = FS + 4)
        else:
            ax.set_yticklabels([])   # no numbers, but ticks/lines stay
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

    # ── Cabinet panels ────────────────────────────────────────────────────────
    fig.canvas.draw()

    for col, ax in enumerate(axes):
        bb = ax.get_position()
        pad_left = PANEL_PAD_X_LEFT0 if col == 0 else PANEL_PAD_X

        x0 = bb.x0 - pad_left
        y0 = bb.y0 - PANEL_PAD_Y - 0.03
        x1 = bb.x1 + PANEL_PAD_X
        y1 = bb.y1 + PANEL_PAD_Y -0.03

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



# ════════════════════════════════════════════════════════════════════════════
#  MOCK DATA for the demo render
# ════════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)
eps_vals_demo = [0.1, 0.5, 1.0, 2.0, 5.0]
L0_vals_demo  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

rows = []
for ds in DATASETS_3:
    for mech in ["Exp", "Gum"]:
        for em in eps_vals_demo:
            for l0 in L0_vals_demo:
                imp = rng.uniform(0, 75)
                rows.append({"dataset": ds, "mechanism": mech,
                             "epsilon_m": em, "L0": l0,
                             "mask_size": MIN_MASK[ds] * (1 - imp / 100),
                             "min_mask": MIN_MASK[ds],
                             "improvement": imp})

def plot_pareto_3_datasets(df: pd.DataFrame) -> plt.Figure:
    """Pareto frontier for airport / hospital / flight only.
    Identical to plot_pareto in every way except:
      - Tax and Adult are removed.
      - Figure width is scaled from 16.5 (5 datasets) → 9.9 (3 datasets).
      - Subplots stretched to match legend width.
      - Y-ticks at 0, 20, 40, 60, 80, 100 with headroom above.
      - Per-dataset cabinet panels matching the other figures.
      - Y-axis tick labels only on leftmost panel.
      - Nothing else is modified.
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
    PANEL_PAD_X_LEFT0 = 0.036   # wider — covers y-tick numbers & ylabel
    PANEL_PAD_X_REST  = 0.015   # same as other panels

    agg = (
        df[df["epsilon_m"].isin(EM_VALS)]
        .groupby(["dataset", "mechanism", "epsilon_m", "L0"])
        .agg(mean_mask=("mask_size", "mean"), mean_leak=("leakage", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.9, 3))

    for col, ds in enumerate(DATASETS_3_PARETO):
        ax       = axes[col]
        baseline = MIN_MASK[ds]
        ax.tick_params(axis="y", left=True, labelleft=(col == 0))
        ax.set_title(rf"{ds.capitalize()}",
                     pad=5, fontsize=FS + 8)
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
                        label=rf"$\textsc{{{mech}}}$, $\varepsilon={em}$")
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
            ax.set_ylabel(r"Relative Mask Size" + "\n" + r"(\% of $|M_{\text{det}}|$)", fontsize=FS + 10)
        else:
            ax.set_ylabel(None)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", frameon=True,
                     bbox_to_anchor=(0.5, 1.12), ncol=6, fontsize=FS + 2)

    # ── Stretch subplots to match legend width ────────────────────────────────
    fig.canvas.draw()
    renderer  = fig.canvas.get_renderer()
    fig_width = fig.get_window_extent(renderer).width
    leg_bb    = leg.get_window_extent(renderer)
    left_frac  = leg_bb.x0 / fig_width
    right_frac = leg_bb.x1 / fig_width
    fig.subplots_adjust(left=left_frac, right=right_frac, top=0.85, bottom=0.2)

    fig.supxlabel(r"Expected Re-inference Leakage $\mathbb{E}[\mathcal{L}(M)]$",
                  y=-0.06, fontsize=FS + 10)

    # ── Cabinet panels ────────────────────────────────────────────────────────
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
def plot_heatmap_5(df: pd.DataFrame):
    df = df.copy()
    df["improvement"] = 100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
    eps_vals = sorted(df["epsilon_m"].unique())
    L0_vals  = sorted(df["L0"].unique())
    L0_plot  = L0_vals[:-1]
    agg = (df.groupby(["dataset", "mechanism", "epsilon_m", "L0"])["improvement"]
             .mean().reset_index())
    fig = plt.figure(figsize=(16.5, 6))
    gs  = GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=0.27, hspace=0.3)
    norm  = Normalize(vmin=0, vmax=75)
    FS_HM = 11
    for row, mech in enumerate(["Exp", "Gum"]):
        for col, ds in enumerate(DATASETS_5):
            ax  = fig.add_subplot(gs[row, col])
            sub = agg[(agg.dataset == ds) & (agg.mechanism == mech)]
            pivot = sub.pivot_table(index="L0", columns="epsilon_m",
                                    values="improvement").reindex(index=L0_plot, columns=eps_vals)
            ax.imshow(pivot.values, cmap=PASTEL_CMAP, norm=norm, origin="lower", aspect="auto")
            ax.set_xticks(range(len(eps_vals)))
            ax.set_xticklabels(eps_vals, rotation=45, fontsize=FS_HM)
            ax.set_yticks(range(len(L0_plot)))
            ax.set_yticklabels(L0_plot, fontsize=FS_HM)
            if row == 0:
                ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", pad=6, fontsize=FS)
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
    sm = ScalarMappable(norm=norm, cmap=PASTEL_CMAP); sm.set_array([])
    cbar_ax = fig.add_subplot(gs[:, 5])
    cb = plt.colorbar(sm, cax=cbar_ax)
    cb.set_label(r"Mask Improvement (\%)", fontsize=FS - 2)
    cb.ax.tick_params(labelsize=FS - 3)
    fig.text(0.07, 0.75, "Exp", ha="center", va="center",
             fontsize=FS, rotation=90, fontweight="bold")
    fig.text(0.07, 0.28, "Gum", ha="center", va="center",
             fontsize=FS, rotation=90, fontweight="bold")
    fig.supylabel(r"$L_0$", fontsize=FS, x=0.08)
    fig.supxlabel(r"Masking Budget $\varepsilon_m$", fontsize=FS - 1, y=0.01)
    tau_handles = [
        Line2D([0], [0], color=TAU_COLORS[t], linestyle="--", linewidth=2.0,
               label=rf"$\tau={t}$")
        for t in TAU_CONTOURS
    ]
    fig.legend(handles=tau_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.00), ncol=len(TAU_CONTOURS),
               frameon=True, fontsize=FS - 1, columnspacing=0.8,
               handlelength=1.8, handletextpad=0.4, borderpad=0.3)
    return fig


# ── 3-dataset curve figures ───────────────────────────────────────────────────

def plot_mask_curves_3(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 6, figsize=(19.8, 3.5), sharey=False)
    ordered = [(ds, "exp") for ds in DATASETS_3] + [(ds, "gum") for ds in DATASETS_3]
    for i, (dataset, mech) in enumerate(ordered):
        ax = axes[i]
        subset = df[(df["dataset"] == dataset) & (df["method"] == mech)]
        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(curve["L0"], curve["improvement"],
                    marker="o", label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"],
                            curve["improvement"] - curve["ci_improvement"],
                            curve["improvement"] + curve["ci_improvement"],
                            alpha=0.18)
        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.set_title(rf"$\mathbf{{{dataset.capitalize()}}}$ ({mech_label})",
                     pad=2, fontsize=SPLIT_FS)
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        if i == 0:
            ax.set_ylabel(r"Mask Size Improvement (\%)", fontsize=SPLIT_FS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 1.08), fontsize=SPLIT_LEG_FS,
               columnspacing=0.8, handlelength=1.8, handletextpad=0.4)
    fig.supxlabel(r"Re-inference Leakage Threshold $L_0$", y=-0.04,
                  fontsize=SPLIT_SUP_FS)
    plt.subplots_adjust(top=0.78, bottom=0.18, wspace=0.28)
    return fig


"""
Drop-in addition to the existing plotting script.
Paste this function anywhere after the existing imports and constants.
Call it the same way you call plot_mask_curves_3 / plot_leakage_curves_3.

Produces a 2-row × 6-col figure:
  Row 0  — mask-size improvement   (identical to plot_mask_curves_3 rows)
  Row 1  — achieved leakage        (identical to plot_leakage_curves_3 rows)

A light-grey shaded rectangle is drawn behind every vertical pair
(one rectangle per column, spanning both rows) to give the
"inner-cabinet" grouped appearance.

No existing function or constant is touched.
"""


"""
Drop-in addition to the existing plotting script.
Paste this function anywhere after the existing imports and constants.
Call it the same way you call plot_mask_curves_3 / plot_leakage_curves_3.

Produces a 2-row × 6-col figure:
  Row 0  — mask-size improvement   (identical to plot_mask_curves_3 rows)
  Row 1  — achieved leakage        (identical to plot_leakage_curves_3 rows)

A light-grey shaded rectangle is drawn behind every vertical pair
(one rectangle per column, spanning both rows) to give the
"inner-cabinet" grouped appearance.

No existing function or constant is touched.
"""


def plot_combined_3_with_panels(df: pd.DataFrame) -> plt.Figure:
    """Combined 2-row × 6-col figure for the 3-dataset results.

    Row 0: Mask-size improvement (replicates plot_mask_curves_3 content).
    Row 1: Achieved re-inference leakage (replicates plot_leakage_curves_3 content).

    A single light-grey rectangle spans both rows for each of the six
    column pairs, creating the "cabinet panel" grouping effect.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_curves_data(DATASETS_3).  Must contain the columns
        produced by that loader: dataset, method, epsilon_m, L0,
        improvement, ci_improvement, mean_leakage, ci_leakage.
    """
    # Per-dataset panel colours (RGB from LaTeX definitions)
    # Order matches `ordered`: [airport, hospital, flight, airport, hospital, flight]
    PANEL_COLORS = [
        "#E1EBF8",  # col 0 — Airport  (rowA: 225,235,248)
        "#E4F3E4",  # col 1 — Hospital (rowB: 228,243,228)
        "#EBE1F5",  # col 2 — Flight   (rowD: 235,225,245)
        "#E1EBF8",  # col 3 — Airport
        "#E4F3E4",  # col 4 — Hospital
        "#EBE1F5",  # col 5 — Flight
    ]
    PANEL_ALPHA = 0.75        # translucency — adjust freely without touching other code
    PANEL_PAD_X = 0.006       # horizontal margin — thin so columns have clear air between them
    PANEL_PAD_Y = 0.035       # vertical margin — extra room above title and below x-ticks

    ordered = [(ds, "exp") for ds in DATASETS_3] + [(ds, "gum") for ds in DATASETS_3]

    fig, axes = plt.subplots(
        2, 6,
        figsize=(19.8, 7.0),   # same natural size as the two single-row originals stacked
        sharey="row",
        sharex=False,
    )

    # ── Shared subplots_adjust ───────────────────────────────────────────────
    plt.subplots_adjust(
        top=0.87, bottom=0.10,
        left=0.06, right=0.98,
        wspace=0.30, hspace=0.48,
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
            pad = 5, fontsize = FS + 4
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
        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])
        if i == 0:
            ax.set_ylabel(r"Achieved Re-inference Leakage" + "\n" + r" $\mathcal{L}(M)$ (\%)", fontsize = 18, labelpad = 10)

    # ── Shared legend (above the figure, mirrors the single-row versions) ────
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
        r"Re-inference Leakage Threshold $L_0$",
        y=0.01,
        fontsize=SPLIT_SUP_FS + 4,
    )

    # ── Draw the cabinet-panel shading behind every column pair ──────────────
    # We must call this AFTER subplots_adjust so that axis positions are final.
    fig.canvas.draw()   # force layout computation

    # Extra left-side pad for col 0 only — covers y-tick numbers & ylabel
    PANEL_PAD_X_LEFT_COL0 = 0.016

    for col in range(6):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Bounding boxes in figure-fraction coordinates
        bb_top = ax_top.get_position()   # Bbox(x0, y0, x1, y1)
        bb_bot = ax_bot.get_position()

        # Left edge: col 0 gets extra room to cover y-tick numbers & ylabel
        pad_left = PANEL_PAD_X_LEFT_COL0 if col == 0 else PANEL_PAD_X

        # Rectangle that encloses both axes in the same column
        x0 = bb_top.x0 - pad_left
        y0 = bb_bot.y0 - PANEL_PAD_Y
        x1 = bb_top.x1 + PANEL_PAD_X
        y1 = bb_top.y1 + PANEL_PAD_Y

        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.005",
            linewidth=0.8,
            edgecolor="#BBBBBB",
            facecolor=PANEL_COLORS[col],
            alpha=PANEL_ALPHA,
            transform=fig.transFigure,
            zorder=0,           # behind everything
            clip_on=False,
        )
        fig.add_artist(rect)

    return fig


def plot_leakage_curves_3(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 6, figsize=(19.8, 3.5), sharey=False)
    ordered = [(ds, "exp") for ds in DATASETS_3] + [(ds, "gum") for ds in DATASETS_3]
    for i, (dataset, mech) in enumerate(ordered):
        ax = axes[i]
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
        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.set_title(rf"$\mathbf{{{dataset.capitalize()}}}$ ({mech_label})",
                     pad=2, fontsize=SPLIT_FS)
        ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])
        if i == 0:
            ax.set_ylabel(r"Achieved Re-inference Leakage (\%)", fontsize=SPLIT_FS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 1.08), fontsize=SPLIT_LEG_FS,
               columnspacing=0.8, handlelength=1.8, handletextpad=0.4)
    fig.supxlabel(r"Re-inference Leakage Threshold $L_0$", y=-0.04,
                  fontsize=SPLIT_SUP_FS)
    plt.subplots_adjust(top=0.78, bottom=0.18, wspace=0.28)
    return fig


# ── Pareto figures ────────────────────────────────────────────────────────────

def plot_pareto(df: pd.DataFrame):
    EM_VALS   = [0.1, 1.0]
    EPS_COLOR = {1.0: "#d62728", 0.1: "#1f77b4"}
    TARGET_ONLY_LEAK = {
        "airport":  0.49830908053361767,
        "hospital": 0.6623992294064142,
        "adult":    0.49122554744404323,
        "flight":   0.9826408586840785,
        "tax":      0.5548418449270073,
    }
    agg = (
        df[df["epsilon_m"].isin(EM_VALS)]
        .groupby(["dataset", "mechanism", "epsilon_m", "L0"])
        .agg(mean_mask=("mask_size", "mean"), mean_leak=("leakage", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 2.75))
    for col, ds in enumerate(DATASETS_5):
        ax       = axes[col]
        baseline = MIN_MASK[ds]
        ax.tick_params(axis="y", left=True, labelleft=True)
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$",
                     pad=2, fontweight="bold", fontsize=FS + 2)
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
                        label=rf"{mech}, $\varepsilon={em}$")
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
        ax.set_ylim(-8, 124)
        if col == 0:
            ax.set_ylabel(r"Mask Size (\% of Baseline)", fontsize=FS + 3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=True,
               bbox_to_anchor=(0.5, 1.12), ncol=6)
    fig.supxlabel(r"Expected Leakage ($\mathbb{E}[\mathcal{L}]$)",
                  y=-0.06, fontsize=FS + 2)
    return fig


def plot_pareto_3(df: pd.DataFrame):
    DATASETS_3_PARETO = ["airport", "hospital", "flight"]
    EM_VALS   = [0.1, 1.0]
    EPS_COLOR = {1.0: "#d62728", 0.1: "#1f77b4"}
    TARGET_ONLY_LEAK = {
        "airport":  0.49830908053361767,
        "hospital": 0.6623992294064142,
        "flight":   0.9826408586840785,
    }
    agg = (
        df[df["epsilon_m"].isin(EM_VALS)]
        .groupby(["dataset", "mechanism", "epsilon_m", "L0"])
        .agg(mean_mask=("mask_size", "mean"), mean_leak=("leakage", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 3))
    for col, ds in enumerate(DATASETS_3_PARETO):
        ax       = axes[col]
        baseline = MIN_MASK[ds]
        ax.tick_params(axis="y", left=True, labelleft=True)
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$",
                     pad=2, fontweight="bold", fontsize=FS + 2)
        for mech in ["Exp", "Gum"]:
            for em in EM_VALS:
                sub = agg[(agg.dataset == ds) & (agg.mechanism == mech) &
                          (agg.epsilon_m == em)].sort_values("L0")
                if sub.empty:
                    continue
                mask_pct     = 100 * sub["mean_mask"] / baseline
                marker_style = "o"  if em == 1.0 else "x"
                marker_face  = "none" if em == 1.0 else None
                line_style   = "-"  if mech == "Exp" else ":"
                ax.plot(sub["mean_leak"], mask_pct,
                        marker=marker_style, linestyle=line_style,
                        color=EPS_COLOR[em], markerfacecolor=marker_face,
                        markeredgewidth=1.8, markersize=6, linewidth=1.8,
                        label=rf"$\textbf{{{mech[0]}{{\footnotesize {mech[1:].upper()}}}}}$, $\varepsilon={em}$")
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
        ax.set_ylim(-8, 124)
        if col == 0:
            ax.set_ylabel(r"Mask Size (\% of Baseline)", fontsize=FS + 10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=True,
               bbox_to_anchor=(0.5, 1.12), ncol=6)
    fig.supxlabel(r"Expected Leakage ($\mathbb{E}[\mathcal{L}]$)",
                  y=-0.06, fontsize=FS + 10)
    plt.subplots_adjust(wspace=0.25)
    return fig


# ── All-methods comparison ────────────────────────────────────────────────────

def load_all_methods_data(csv_path: str = "all_masks_all_methods.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["dataset"] = df["dataset"].str.capitalize()
    return df


def plot_mask_vs_leakage_all_methods(
    csv_path: str = "all_masks_all_methods.csv",
) -> plt.Figure:
    METHODS = [
        ("leakage_noisy_or",        r"\textsc{Nor}",       "#d62728", "-",  "o"),
        ("leakage_greedy_disjoint", r"\textsc{This Work}", "#1f77b4", "--", "o"),
        ("leakage_max",             r"\textsc{Max}",       "#2ca02c", ":",  "o"),
    ]
    BAND_ALPHA = 0.15
    LINE_W     = 1.8
    MARKER_SZ  = 6
    FS_BIG     = FS + 6

    df = load_all_methods_data(csv_path)
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(wspace=0.18, left=0.07, right=0.97, top=0.82, bottom=0.15)

    for col, dataset in enumerate(DATASET_ORDER):
        ax  = axes[col]
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            ax.set_axis_off(); continue

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

        ax.set_title(rf"$\mathbf{{{dataset}}}$", pad=6, fontsize=FS_BIG)
        ax.tick_params(axis="both", labelsize=FS_BIG - 2)
        ax.set_xlim(min(x_vals) - 0.1, max(x_vals) + 0.1)
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if col == 0:
            ax.set_ylabel(r"Re-inference Leakage (\%)", fontsize=FS_BIG)
        else:
            ax.set_ylabel(None)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=len(METHODS), frameon=True, fontsize=FS_BIG,
               columnspacing=1.2, handlelength=2.2,
               handletextpad=0.5, borderpad=0.4)
    fig.supxlabel("Mask Size", y=-0.06, fontsize=FS_BIG)
    return fig


# ── Runtime breakdown ─────────────────────────────────────────────────────────

def build_runtime(df: pd.DataFrame):
    fig = plt.figure(figsize=(16.5, 3.5))
    subspec = fig.add_gridspec(1, 1)[0, 0].subgridspec(1, 5, wspace=0.35)
    axes = [fig.add_subplot(subspec[0, i]) for i in range(5)]
    for i, dataset in enumerate(DATASET_ORDER):
        ax  = axes[i]
        ddf = df[df["dataset"] == dataset]
        if ddf.empty:
            ax.set_axis_off(); continue
        s = (ddf.groupby("method").mean(numeric_only=True)
                .reindex(METHOD_ORDER).fillna(0.0))
        x      = np.arange(len(METHOD_ORDER))
        w      = 0.25
        bottom = np.zeros(len(METHOD_ORDER))
        for key, label in [("init_time_ms",   "Instantiation"),
                            ("model_time_ms",  "Modeling"),
                            ("update_time_ms", "Update Masks")]:
            vals = s[key].values
            ax.bar(x - w, vals, width=w, bottom=bottom,
                   color=PHASE_COLORS[label], hatch=PHASE_HATCH[label],
                   edgecolor="black", linewidth=0.3)
            bottom += vals
        ax.set_xticks(x - w)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
        ax2 = ax.twinx()
        deleted_pct      = s["deletion_ratio"] * 100.0
        instantiated_pct = 100.0 - deleted_pct
        ax2.bar(x + w, instantiated_pct, width=w,
                color=ZONE_COLOR_LIGHT, edgecolor="black", linewidth=0.3)
        ax2.bar(x + w, deleted_pct, width=w, bottom=instantiated_pct,
                color=ZONE_COLOR_DARK, edgecolor="black", linewidth=0.3)
        ax2.set_ylim(0, 100)
        ax.tick_params(axis="y", labelleft=True)
        ax2.tick_params(axis="y", labelright=True)
        if i == 0:
            ax.set_ylabel("Time (ms)")
        else:
            ax.set_ylabel(None)
        if i == len(DATASET_ORDER) - 1:
            ax2.set_ylabel(r"Deleted Cells (\%)")
        else:
            ax2.set_ylabel(None)
            ax2.spines["right"].set_visible(False)
        ax.set_title(dataset, pad=10)
    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p], hatch=PHASE_HATCH[p],
              edgecolor="black", linewidth=0.3, label=p)
        for p in PHASE_COLORS
    ]
    zone_handles = [
        Patch(facecolor=ZONE_COLOR_LIGHT, edgecolor="black", lw=0.3, label="Instantiated Cells"),
        Patch(facecolor=ZONE_COLOR_DARK,  edgecolor="black", lw=0.3, label="Mask Size"),
    ]
    fig.legend(handles=phase_handles + zone_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=5, fontsize=FS, frameon=True)
    fig.subplots_adjust(top=0.72)
    return fig


# ── Budget split ──────────────────────────────────────────────────────────────

def plot_budget_split_5(df: pd.DataFrame):
    df = df.copy()
    df["improvement"] = 100 * (df["min_mask"] - df["mask_size"]) / df["min_mask"]
    agg = (df.groupby(["dataset", "mechanism", "epsilon_m", "L0"])["improvement"]
             .mean().reset_index())
    agg["tau"] = agg.apply(lambda r: tau_fn(r["epsilon_m"], r["L0"]), axis=1)
    tol = 0.04
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 6.5), sharey="row")
    plt.subplots_adjust(hspace=0.30, wspace=0.18,
                        left=0.07, right=0.97, top=0.88, bottom=0.10)
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
                ax.set_axis_off(); continue
            x = np.arange(len(valid_taus))
            w = 0.35
            ax.bar(x,     worst_vals, w, color="#d62728", edgecolor="black",
                   linewidth=0.5, label="Worst split")
            ax.bar(x + w, best_vals,  w, color="#2ca02c", edgecolor="black",
                   linewidth=0.5, label="Best split")
            ax.set_xticks(x + w / 2)
            ax.set_xticklabels([rf"$\tau\approx{t}$" for t in valid_taus],
                               fontsize=FS - 2)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            if row == 0:
                ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", pad=6, fontsize=FS)
    handles = [
        Patch(facecolor="#d62728", edgecolor="black", lw=0.5, label="Worst split"),
        Patch(facecolor="#2ca02c", edgecolor="black", lw=0.5, label="Best split"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1), ncol=2, frameon=True)
    fig.text(0.02, 0.75, "Exp", ha="center", va="center",
             fontsize=FS, rotation=90, fontweight="bold")
    fig.text(0.02, 0.28, "Gum", ha="center", va="center",
             fontsize=FS, rotation=90, fontweight="bold")
    fig.supylabel(r"Mask Size Improvement (\%)", fontsize=FS, x=0.03)
    fig.supxlabel(r"Privacy Budget Allocation ($\tau$)", fontsize=FS, y=0.01)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  SPLIT & ABLATION PLOTS — all use SPLIT_FS / SPLIT_TICK_FS / SPLIT_LEG_FS
# ════════════════════════════════════════════════════════════════════════════

def plot_mask_split(df: pd.DataFrame, mech: str, legend: bool = True):
    """Mask-size improvement curves, one panel per dataset (single mechanism)."""
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(**SPLIT_ADJUST)
    subset_all = df[df["method"] == mech]
    for col, ds in enumerate(DATASETS_5):
        ax     = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]
        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            ax.plot(curve["L0"], curve["improvement"],
                    marker="o", label=rf"$\varepsilon_m = {eps}$")
            ax.fill_between(curve["L0"],
                            curve["improvement"] - curve["ci_improvement"],
                            curve["improvement"] + curve["ci_improvement"],
                            alpha=0.18)
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=SPLIT_FS)
        _apply_split_axis_style(ax, col, [0, 25, 50, 75, 100],
                                r"Mask Size Improvement (\%)")
        ax.set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   frameon=True, bbox_to_anchor=(0.5, 1.05), fontsize=SPLIT_LEG_FS)
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=SPLIT_SUP_FS)
    return fig


def plot_leakage_split(df: pd.DataFrame, mech: str):
    """Achieved leakage curves, one panel per dataset (single mechanism)."""
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(**SPLIT_ADJUST)
    subset_all = df[df["method"] == mech]
    for col, ds in enumerate(DATASETS_5):
        ax     = axes[col]
        subset = subset_all[subset_all["dataset"] == ds]
        for eps in sorted(subset["epsilon_m"].unique()):
            curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean  = 100 * curve["mean_leakage"]
            lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean, marker="o")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)
        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=SPLIT_FS)
        _apply_split_axis_style(ax, col, [0, 20, 40, 60, 80],
                                r"Achieved Re-inference Leakage (\%)")
        ax.set_ylim(0, 80)
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=SPLIT_SUP_FS)
    return fig


def plot_leakage_split_gum(df: pd.DataFrame, mech: str):
    """Leakage-split with legend — for the Gum mechanism appendix."""
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(**SPLIT_ADJUST)
    subset_all = df[df["method"] == mech]
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
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=SPLIT_FS)
        _apply_split_axis_style(ax, col, [0, 20, 40, 60, 80],
                                r"Achieved Re-inference Leakage (\%)")
        ax.set_ylim(0, 80)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 1.05), fontsize=SPLIT_LEG_FS)
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=SPLIT_SUP_FS)
    return fig


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


# ── Shared 2-row ablation grid ────────────────────────────────────────────────

def _ablation_axes_grid(datasets: list):
    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(19.8 / 5 * n, 6.5), sharey="row")
    plt.subplots_adjust(hspace=0.32, wspace=0.20,
                        left=0.09, right=0.97, top=0.85, bottom=0.12)
    return fig, axes


# ── 2-row combined ablation plots ─────────────────────────────────────────────

def plot_mask_split_ablation(
    df: pd.DataFrame,
    datasets: list = DATASETS_5,
):
    """2 rows (Random / Zero) x n cols — mask-size improvement."""
    VARIANT_LABEL = {"random": "Random", "zero": "Zero"}
    fig, axes = _ablation_axes_grid(datasets)
    for row, variant in enumerate(["random", "zero"]):
        subset_all = df[df["variant"] == variant]
        for col, ds in enumerate(datasets):
            ax     = axes[row, col]
            subset = subset_all[subset_all["dataset"] == ds]
            for eps in sorted(subset["epsilon_m"].unique()):
                curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
                ax.plot(curve["L0"], curve["improvement"],
                        marker="o", label=rf"$\varepsilon_m={eps}$")
                ax.fill_between(
                    curve["L0"],
                    curve["improvement"] - curve["ci_improvement"],
                    curve["improvement"] + curve["ci_improvement"],
                    alpha=0.18,
                )
            if row == 0:
                ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$",
                             pad=6, fontsize=SPLIT_FS)
            ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_xlim(0.0, 0.9)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            if col == 0:
                ax.set_ylabel(
                    f"{VARIANT_LABEL[variant]}\nMask Size Improvement (\\%)",
                    fontsize=SPLIT_FS,
                )
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="upper center", ncol=len(h),
                       frameon=True, bbox_to_anchor=(0.5, 1.02),
                       fontsize=SPLIT_LEG_FS)
            break
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=SPLIT_SUP_FS, y=0.01)
    return fig


def plot_leakage_split_ablation(
    df: pd.DataFrame,
    datasets: list = DATASETS_5,
):
    """2 rows (Random / Zero) x n cols — achieved leakage."""
    VARIANT_LABEL = {"random": "Random", "zero": "Zero"}
    fig, axes = _ablation_axes_grid(datasets)
    for row, variant in enumerate(["random", "zero"]):
        subset_all = df[df["variant"] == variant]
        for col, ds in enumerate(datasets):
            ax     = axes[row, col]
            subset = subset_all[subset_all["dataset"] == ds]
            for eps in sorted(subset["epsilon_m"].unique()):
                curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
                mean  = 100 * curve["mean_leakage"]
                lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
                upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
                ax.plot(curve["L0"], mean, marker="o",
                        label=rf"$\varepsilon_m={eps}$")
                ax.fill_between(curve["L0"], lower, upper, alpha=0.18)
            ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1,
                    color="black", alpha=0.4)
            if row == 0:
                ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$",
                             pad=6, fontsize=SPLIT_FS)
            ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_xlim(0.0, 0.9)
            ax.set_ylim(0, 80)
            ax.set_yticks([0, 20, 40, 60, 80])
            if col == 0:
                ax.set_ylabel(
                    f"{VARIANT_LABEL[variant]}\nAchieved Re-inference Leakage (\\%)",
                    fontsize=SPLIT_FS,
                )
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="upper center", ncol=len(h),
                       frameon=True, bbox_to_anchor=(0.5, 1.02),
                       fontsize=SPLIT_LEG_FS)
            break
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=SPLIT_SUP_FS, y=0.01)
    return fig


# ── Single-variant ablation plots ─────────────────────────────────────────────

def plot_mask_split_ablation_single(
    df: pd.DataFrame,
    variant: str,
    datasets: list = DATASETS_5,
) -> plt.Figure:
    """1 row x n cols — mask-size improvement for one ablation variant.
    No legend (legend lives only on the gum figure above it)."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(**SPLIT_ADJUST)
    subset_all = df[df["variant"] == variant]
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
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=SPLIT_FS)
        _apply_split_axis_style(ax, col, [0, 25, 50, 75, 100],
                                r"Mask Size Improvement (\%)")
        ax.set_ylim(0, 100)
    # no legend — shared with the gum figure stacked above
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=SPLIT_SUP_FS)
    return fig


def plot_leakage_split_ablation_single(
    df: pd.DataFrame,
    variant: str,
    datasets: list = DATASETS_5,
) -> plt.Figure:
    """1 row x n cols — achieved leakage for one ablation variant.
    No legend (legend lives only on the gum figure above it)."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(**SPLIT_ADJUST)
    subset_all = df[df["variant"] == variant]
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
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=SPLIT_FS)
        _apply_split_axis_style(ax, col, [0, 20, 40, 60, 80],
                                r"Achieved Re-inference Leakage (\%)")
        ax.set_ylim(0, 80)
    # no legend — shared with the gum figure stacked above
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=SPLIT_SUP_FS)
    return fig


# ── Combined 3-row figures (Gum / Random / Zero in one figure) ───────────────

ROW_LABELS = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}"]


def _make_combined_fig():
    """Return a 3-row × 5-col figure with shared layout constants."""
    fig, axes = plt.subplots(
        3, 5,
        figsize=(16.5, 9.0),
        sharey="row", sharex=False,
    )
    plt.subplots_adjust(
        hspace=0.45, wspace=0.30,
        left=0.07, right=0.97, top=0.86, bottom=0.06,
    )
    return fig, axes


def plot_mask_combined(df_curves: pd.DataFrame, df_abl: pd.DataFrame) -> plt.Figure:
    """3 rows x 5 cols — mask-size improvement.
       Row (a): Gumbel   Row (b): Random ablation   Row (c): Zero ablation
    """
    ROWS = [
        (df_curves, "method",  "gum"   ),
        (df_abl,    "variant", "random"),
        (df_abl,    "variant", "zero"  ),
    ]
    fig, axes = _make_combined_fig()

    for row, (df, key, val) in enumerate(ROWS):
        subset_all = df[df[key] == val]
        for col, ds in enumerate(DATASETS_5):
            ax     = axes[row, col]
            subset = subset_all[subset_all["dataset"] == ds]

            for eps in sorted(subset["epsilon_m"].unique()):
                curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
                ln,   = ax.plot(curve["L0"], curve["improvement"],
                                marker="o", label=rf"$\varepsilon_m = {eps}$")
                ax.fill_between(
                    curve["L0"],
                    curve["improvement"] - curve["ci_improvement"],
                    curve["improvement"] + curve["ci_improvement"],
                    alpha=0.18, color=ln.get_color(),
                )

            # Dataset name as title above the panel (row 0 only)
            if row == 0:
                ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$",
                             fontsize=SPLIT_FS, pad=6)

            ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_xlim(0.0, 0.9)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])

            # (a)/(b)/(c) label + ylabel on leftmost panel only
            if col == 0:
                ax.set_ylabel(r"Mask Size Improvement (\%)", fontsize=SPLIT_FS)
                ax.text(0.03, 0.97, ROW_LABELS[row], transform=ax.transAxes,
                        fontsize=SPLIT_FS, va="top", ha="left")

            # One x-axis label per row on the centre panel
            if col == 2:
                ax.set_xlabel(r"Re-inference Threshold $L_0$",
                              fontsize=SPLIT_SUP_FS)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 1.0), fontsize=SPLIT_LEG_FS,
               columnspacing=0.8, handlelength=1.8, handletextpad=0.4)
    return fig


def plot_leakage_combined(df_curves: pd.DataFrame, df_abl: pd.DataFrame) -> plt.Figure:
    """3 rows x 5 cols — achieved re-inference leakage.
       Row (a): Gumbel   Row (b): Random ablation   Row (c): Zero ablation
    """
    ROWS = [
        (df_curves, "method",  "gum"   ),
        (df_abl,    "variant", "random"),
        (df_abl,    "variant", "zero"  ),
    ]
    fig, axes = _make_combined_fig()

    for row, (df, key, val) in enumerate(ROWS):
        subset_all = df[df[key] == val]
        for col, ds in enumerate(DATASETS_5):
            ax     = axes[row, col]
            subset = subset_all[subset_all["dataset"] == ds]

            for eps in sorted(subset["epsilon_m"].unique()):
                curve = subset[subset["epsilon_m"] == eps].sort_values("L0")
                mean  = 100 * curve["mean_leakage"]
                lower = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
                upper = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
                ln,   = ax.plot(curve["L0"], mean, marker="o",
                                label=rf"$\varepsilon_m = {eps}$")
                ax.fill_between(curve["L0"], lower, upper,
                                alpha=0.18, color=ln.get_color())

            ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1,
                    color="black", alpha=0.4)

            # Dataset name as title above the panel (row 0 only)
            if row == 0:
                ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$",
                             fontsize=SPLIT_FS, pad=6)

            ax.tick_params(axis="both", labelsize=SPLIT_TICK_FS)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_xlim(0.0, 0.9)
            ax.set_ylim(0, 80)
            ax.set_yticks([0, 20, 40, 60, 80])

            if col == 0:
                ax.set_ylabel(r"Achieved Re-inference Leakage (\%)", fontsize=SPLIT_FS)
                ax.text(0.03, 0.97, ROW_LABELS[row], transform=ax.transAxes,
                        fontsize=SPLIT_FS, va="top", ha="left")

            if col == 2:
                ax.set_xlabel(r"Re-inference Threshold $L_0$",
                              fontsize=SPLIT_SUP_FS)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 1.0), fontsize=SPLIT_LEG_FS,
               columnspacing=0.8, handlelength=1.8, handletextpad=0.4)
    return fig


# ── graph_all_experiments ─────────────────────────────────────────────────────

def graph_all_experiments():
    df_heat     = load_heatmap_data()
    df_curves_3 = load_curves_data(DATASETS_3)
    df_curves_5 = load_curves_data(DATASETS_5)
    df_main     = load_main_data()

    FIG_DIR = BASE_DIR / "fig_v2"
    FIG_DIR.mkdir(exist_ok=True)

    if not df_heat.empty:
        fig = plot_heatmap_3(df_heat)
        fig.savefig(FIG_DIR / "heatmap_3.pdf", bbox_inches="tight"); plt.close(fig)

    if not df_curves_3.empty:
        fig = plot_mask_curves_3(df_curves_3)
        fig.savefig(FIG_DIR / "mask_curves_3.pdf", bbox_inches="tight"); plt.close(fig)
        fig = plot_leakage_curves_3(df_curves_3)
        fig.savefig(FIG_DIR / "leakage_curves_3.pdf", bbox_inches="tight"); plt.close(fig)

    if not df_heat.empty:
        fig = plot_pareto(df_heat)
        fig.savefig(FIG_DIR / "pareto_frontier.pdf", bbox_inches="tight"); plt.close(fig)

    if not df_main.empty:
        fig = build_runtime(df_main)
        fig.savefig(FIG_DIR / "runtime_breakdown.pdf", bbox_inches="tight"); plt.close(fig)

    if not df_heat.empty:
        fig = plot_heatmap_5(df_heat)
        fig.savefig(FIG_DIR / "heatmap_5.pdf", bbox_inches="tight"); plt.close(fig)
        fig = plot_budget_split_5(df_heat)
        fig.savefig(FIG_DIR / "budget_split_5.pdf", bbox_inches="tight"); plt.close(fig)

    if not df_curves_5.empty:
        for mech in ["exp", "gum"]:
            fig = plot_mask_split(df_curves_5, mech)
            fig.savefig(FIG_DIR / f"mask_split_{mech}.pdf", bbox_inches="tight"); plt.close(fig)
        for mech in ["exp", "gum"]:
            fig = plot_leakage_split(df_curves_5, mech)
            fig.savefig(FIG_DIR / f"leakage_split_{mech}.pdf", bbox_inches="tight"); plt.close(fig)

    print(f"\nAll figures saved to: {FIG_DIR.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

FIG_DIR = BASE_DIR / "fig_v2"
FIG_DIR.mkdir(exist_ok=True)

df_heat     = load_heatmap_data()
df_curves_3 = load_curves_data(DATASETS_3)
df_curves_5 = load_curves_data(DATASETS_5)
df_main     = load_main_data()

fig = plot_combined_3_with_panels(df_curves_3)
fig.savefig(FIG_DIR / "combined_3_with_panels.pdf", bbox_inches="tight")
plt.close(fig)
# fig = plot_pareto_3(df_heat)
# fig.savefig(FIG_DIR / "pareto_frontier_3.pdf", bbox_inches="tight")
# plt.close(fig)
#
# df_abl = load_ablation_data()
#
# for variant in ["random", "zero"]:
#     fig = plot_mask_split_ablation_single(df_abl, variant)
#     fig.savefig(FIG_DIR / f"mask_split_ablation_{variant}.pdf", bbox_inches="tight")
#     plt.close(fig)
#
#     fig = plot_leakage_split_ablation_single(df_abl, variant)
#     fig.savefig(FIG_DIR / f"leakage_split_ablation_{variant}.pdf", bbox_inches="tight")
#     plt.close(fig)
#
# fig = plot_leakage_split_gum(df_curves_5, "gum")
# fig.savefig(FIG_DIR / "leakage_split_gum_appendix.pdf", bbox_inches="tight")
# plt.close(fig)
#
# # ── Combined 3-row figures (Gum / Random / Zero in one figure) ───────────────
# fig = plot_mask_combined(df_curves_5, df_abl)
# fig.savefig(FIG_DIR / "mask_combined.pdf", bbox_inches="tight")
# plt.close(fig)
#
# fig = plot_leakage_combined(df_curves_5, df_abl)
# fig.savefig(FIG_DIR / "leakage_combined.pdf", bbox_inches="tight")
# plt.close(fig)
#
# fig = plot_mask_vs_leakage_all_methods("all_masks_all_methods.csv")
# fig.savefig(FIG_DIR / "mask_vs_leakage_all_methods.pdf", bbox_inches="tight")
# plt.close(fig)
#
# fig = plot_combined_3_with_panels(df_curves_3)
# fig.savefig(FIG_DIR / "combined_3_with_panels.pdf", bbox_inches="tight")
# plt.close(fig)
#
graph_all_experiments()

fig = plot_heatmap_3_with_panels(df_heat)
fig.savefig(FIG_DIR / "heatmap_3_with_panels.pdf", bbox_inches="tight")
plt.close(fig)

fig = plot_pareto_3_datasets(df_heat)
fig.savefig(FIG_DIR / "pareto_frontier_3_datasets.pdf", bbox_inches="tight")
plt.close(fig)