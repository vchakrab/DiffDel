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
DATA_DIR = BASE_DIR / "data"
print(DATA_DIR)

FS = 11

plt.rcParams.update({
    "font.family":           "STIXGeneral",
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
plt.rcParams["mathtext.fontset"] = "stix"

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
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$ ({mech})", pad=8, fontsize=FS)
        sub   = agg[(agg.dataset == ds) & (agg.mechanism == mech)]
        pivot = sub.pivot_table(index="L0", columns="epsilon_m",
                                values="improvement").reindex(index=L0_plot, columns=eps_vals)
        ax.imshow(pivot.values, cmap=PASTEL_CMAP, norm=norm, origin="lower", aspect="auto")
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels(eps_vals, rotation=45)
        ax.set_yticks(range(len(L0_plot)))
        ax.set_yticklabels(L0_plot)
        if i == 0:
            ax.set_ylabel(r"Re-inference Leakage Threshold $L_0$", fontsize=FS + 3)
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
    cax.text(-0.025, 0.5, "Mask Improvement (%)",
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
    fig.supxlabel(r"Masking-Privacy Budget $\epsilon_m$", y=-0.04, fontsize=FS + 3)
    plt.subplots_adjust(top=0.75, bottom=0.18)
    return fig


def plot_mask_curves_3(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 6, figsize=(19.8, 3), sharey=False)
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
                     pad=2, fontsize=FS + 2)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)
        if i == 0:
            ax.set_ylabel("Mask Size Improvement (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig.supxlabel(r"Re-inference Leakage Threshold $L_0$", y=-0.06)
    plt.subplots_adjust(top=0.82, wspace=0.25)
    return fig


def plot_leakage_curves_3(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 6, figsize=(19.8, 3), sharey=False)
    ordered = [(ds, "exp") for ds in DATASETS_3] + [(ds, "gum") for ds in DATASETS_3]
    for i, (dataset, mech) in enumerate(ordered):
        ax = axes[i]
        subset = df[(df["dataset"] == dataset) & (df["method"] == mech)]
        for eps in sorted(subset["epsilon_m"].unique()):
            curve     = subset[subset["epsilon_m"] == eps].sort_values("L0")
            mean_leak = 100 * curve["mean_leakage"]
            lower     = 100 * (curve["mean_leakage"] - curve["ci_leakage"])
            upper     = 100 * (curve["mean_leakage"] + curve["ci_leakage"])
            ax.plot(curve["L0"], mean_leak, marker="o")
            ax.fill_between(curve["L0"], lower, upper, alpha=0.18)
        ax.plot([0, 1], [0, 100], linestyle="--", linewidth=1)
        mech_label = "Gum" if mech == "gum" else mech.capitalize()
        ax.set_title(rf"$\mathbf{{{dataset.capitalize()}}}$ ({mech_label})",
                     pad=2, fontsize=FS + 2)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        if i == 0:
            ax.set_ylabel("Achieved Re-inference Leakage (%)")
    fig.supxlabel(r"Re-inference Leakage Threshold $L_0$", y=0.02)
    plt.subplots_adjust(top=0.85, bottom=0.18, wspace=0.25)
    return fig


def plot_pareto(df: pd.DataFrame):
    EM_VALS = [0.1, 1.0]
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
                        label=f"{mech}, ε={em}")
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
            ax.set_ylabel("Mask Size (% of Baseline)", fontsize=FS + 3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=True,
               bbox_to_anchor=(0.5, 1.12), ncol=6)
    fig.supxlabel(r"Expected Leakage ($\mathbb{E}[\mathcal{L}]$)",
                  y=-0.06, fontsize=FS + 2)
    return fig


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
        for key, label in [("init_time_ms", "Instantiation"),
                            ("model_time_ms", "Modeling"),
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
            ax2.set_ylabel("Deleted Cells (%)")
        else:
            ax2.set_ylabel(None)
            ax2.spines["right"].set_visible(False)
        ax.set_title(dataset, pad=10, fontweight="bold")
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
    cb.set_label("Mask Improvement (%)", fontsize=FS - 2)
    cb.ax.tick_params(labelsize=FS - 3)
    fig.text(0.07, 0.75, "Exp", ha="center", va="center",
             fontsize=FS, rotation=90, fontweight="bold")
    fig.text(0.07, 0.28, "Gum", ha="center", va="center",
             fontsize=FS, rotation=90, fontweight="bold")
    fig.supylabel(rf"$L_0$", fontsize=FS, x=0.08)
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
    fig.supylabel("Mask Size Improvement (%)", fontsize=FS, x=0.03)
    fig.supxlabel(r"Privacy Budget Allocation ($\tau$)", fontsize=FS, y=0.01)
    return fig


def plot_mask_split(df: pd.DataFrame, mech: str):
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(wspace=0.18, left=0.07, right=0.97, top=0.82, bottom=0.15)
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
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        if col == 0:
            ax.set_ylabel("Mask Size Improvement (%)")
        else:
            ax.set_ylabel(None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=FS)
    return fig


def plot_leakage_split(df: pd.DataFrame, mech: str):
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), sharey=False)
    plt.subplots_adjust(wspace=0.18, left=0.07, right=0.97, top=0.82, bottom=0.15)
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
        ax.set_title(rf"$\mathbf{{{ds.capitalize()}}}$", fontsize=FS)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0.0, 0.9)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 20, 40, 60, 80])
        if col == 0:
            ax.set_ylabel("Achieved Re-inference Leakage (%)", fontsize=FS - 1)
        else:
            ax.set_ylabel(None)
    fig.supxlabel(r"Re-inference Threshold $L_0$", fontsize=FS)
    return fig


def graph_all_experiments():
    df_heat     = load_heatmap_data()
    df_curves_3 = load_curves_data(DATASETS_3)
    df_curves_5 = load_curves_data(DATASETS_5)
    df_main     = load_main_data()

    with PdfPages("all_figures_combined.pdf") as pdf:

        if not df_heat.empty:
            fig = plot_heatmap_3(df_heat)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if not df_curves_3.empty:
            fig = plot_mask_curves_3(df_curves_3)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            fig = plot_leakage_curves_3(df_curves_3)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if not df_heat.empty:
            fig = plot_pareto(df_heat)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if not df_main.empty:
            fig = build_runtime(df_main)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if not df_heat.empty:
            fig = plot_heatmap_5(df_heat)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            fig = plot_budget_split_5(df_heat)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if not df_curves_5.empty:
            for mech in ["exp", "gum"]:
                fig = plot_mask_split(df_curves_5, mech)
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            for mech in ["exp", "gum"]:
                fig = plot_leakage_split(df_curves_5, mech)
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


