#!/usr/bin/env python3
"""
generate_combined.py (PRESS RUN — no CLI needed)

Layout:
LEFT column:
  (a) |M| vs Leakage tradeoff (TOP)
  (b) Deletion ratio          (BOTTOM)

RIGHT column:
  (c) Parameter sensitivity / ablation (spans both rows)

Robustness upgrades:
  - Does NOT assume any output filenames from generate_plots.py
  - Records actual savefig() calls to discover the produced PDFs/PNGs
  - Auto-saves PNG sibling for every PDF save

Folder scheme supported:
experiment_outputs/YYYY-MM-DD/HH-MM-SS/
  plot_ablation_3sweeps.py
  ablation/   (csvs here)

Output:
  fig_main_composed.pdf
Intermediates:
  _tmp_figs/
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# Settings
# =========================
ROOT = Path(__file__).resolve().parent  # DifferentialDeletionAlgorithms/
EXPERIMENT_OUTPUTS_DIRNAME = "experiment_outputs"

OUT_PDF_NAME = "fig_main_composed.pdf"
TMP_DIRNAME = "_tmp_figs"

CROP_BG_THRESH = 250
CROP_PAD_PX = 4
SAVEFIG_DPI = 300

PAD_OUTER = 0.006
GAP_COL = 0.008
GAP_ROW = 0.008


# =========================
# Import helpers
# =========================
def _import_module_from_path(module_name: str, file_path: Path):
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot import; file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for: {file_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_import_generate_plots(root: Path):
    try:
        import generate_plots  # type: ignore
        return generate_plots
    except Exception:
        gp_path = root / "generate_plots.py"
        if not gp_path.exists():
            raise FileNotFoundError(f"Couldn't find generate_plots.py at: {gp_path}")
        return _import_module_from_path("generate_plots", gp_path)


def _find_latest_ablation_script(root: Path) -> Path:
    base = (root / EXPERIMENT_OUTPUTS_DIRNAME).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Couldn't find {EXPERIMENT_OUTPUTS_DIRNAME}/ under: {root}")

    # Script is at run-folder level (NOT inside /ablation/)
    candidates = list(base.glob("**/plot_ablation_*sweeps.py"))
    if not candidates:
        raise FileNotFoundError(
            f"No plot_ablation_*sweeps.py found under {base}.\n"
            f"Expected: experiment_outputs/YYYY-MM-DD/HH-MM-SS/plot_ablation_3sweeps.py"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# =========================
# Savefig recorder (the fix)
# =========================
class SavefigRecorder:
    """
    Monkeypatch plt.savefig so we can:
      - record exactly what filenames were saved
      - automatically emit PNG sibling for any PDF
    """

    def __init__(self):
        self._orig = plt.savefig
        self.saved: List[Path] = []

    def install(self):
        def wrapped(fname, *args, **kwargs):
            p = Path(str(fname))
            self.saved.append(p)
            self._orig(fname, *args, **kwargs)

            # If saving PDF, also save PNG sibling
            s = str(fname)
            if s.lower().endswith(".pdf"):
                png_name = s[:-4] + ".png"
                self.saved.append(Path(png_name))
                self._orig(png_name, *args, **kwargs)

        plt.savefig = wrapped  # type: ignore[assignment]

    def reset(self):
        self.saved = []

    def uninstall(self):
        plt.savefig = self._orig  # type: ignore[assignment]

    def last_pdf(self) -> Optional[Path]:
        for p in reversed(self.saved):
            if str(p).lower().endswith(".pdf"):
                return p
        return None

    def last_png(self) -> Optional[Path]:
        for p in reversed(self.saved):
            if str(p).lower().endswith(".png"):
                return p
        return None


# =========================
# Image helpers
# =========================
def _crop_whitespace(img: Image.Image, bg_thresh: int = CROP_BG_THRESH, pad: int = CROP_PAD_PX) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    fg = np.any(arr < bg_thresh, axis=2)
    if not fg.any():
        return img

    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)

    return img.crop((x0, y0, x1 + 1, y1 + 1))


def _load_crop_png(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Expected PNG not found: {path}")
    return _crop_whitespace(Image.open(path))


def _require(p: Optional[Path], what: str) -> Path:
    if p is None:
        raise FileNotFoundError(f"Could not determine {what} from savefig() calls.")
    return p


# =========================
# Main
# =========================
def main():
    root = ROOT
    tmp = root / TMP_DIRNAME
    tmp.mkdir(parents=True, exist_ok=True)
    out_pdf = root / OUT_PDF_NAME

    print("🧠  Composing your main figure like a caffeinated grad student...")
    print("📍  ROOT =", root)

    plt.rcParams["savefig.dpi"] = SAVEFIG_DPI

    gp = _ensure_import_generate_plots(root)

    ablation_script = _find_latest_ablation_script(root)
    run_dir = ablation_script.parent
    ablation_dir = run_dir / "ablation"
    if not ablation_dir.exists():
        raise FileNotFoundError(f"Expected ablation folder at: {ablation_dir}")

    print("🧪  Using ablation script:", ablation_script)
    print("📦  Using ablation CSV dir:", ablation_dir)

    pa = _import_module_from_path("plot_ablation_3sweeps", ablation_script)

    rec = SavefigRecorder()
    rec.install()

    try:
        # -------------------------
        # (a) tradeoff
        # -------------------------
        df = gp.load_data(root)

        rec.reset()
        gp.fig_leakage_vs_mask_tradeoff(df, tmp)
        trade_png = _require(rec.last_png(), "tradeoff PNG")
        print("✅ tradeoff saved as:", trade_png)

        # -------------------------
        # (b) deletion ratio
        # -------------------------
        rec.reset()
        gp.fig_deletion_ratio_vs_constraints(df, tmp)
        ratio_png = _require(rec.last_png(), "deletion ratio PNG")
        print("✅ deletion ratio saved as:", ratio_png)

        # -------------------------
        # (c) ablation
        # -------------------------
        lam_raw = pa.load_sweep(ablation_dir, "lam", "lambda")
        eps_raw = pa.load_sweep(ablation_dir, "epsilon", "epsilon")
        l0_raw  = pa.load_sweep(ablation_dir, "L0", "L0")

        lam = {k: pa.compute_stats(v, "lambda") for k, v in lam_raw.items()}
        eps = {k: pa.compute_stats(v, "epsilon") for k, v in eps_raw.items()}
        l0  = {k: pa.compute_stats(v, "L0") for k, v in l0_raw.items()}

        rec.reset()
        ablation_pdf = tmp / "fig_ablation_tmp.pdf"
        pa.plot_figure(lam, eps, l0, str(ablation_pdf))
        abl_png = _require(rec.last_png(), "ablation PNG")
        print("✅ ablation saved as:", abl_png)

    finally:
        rec.uninstall()

    # ---- crop ----
    trade_img = _load_crop_png(trade_png)
    ratio_img = _load_crop_png(ratio_png)
    abl_img   = _load_crop_png(abl_png)

    trade_arr = np.array(trade_img)
    ratio_arr = np.array(ratio_img)
    abl_arr   = np.array(abl_img)

    # ---- compose ----
    left_w = max(trade_arr.shape[1], ratio_arr.shape[1])
    left_h = trade_arr.shape[0] + ratio_arr.shape[0]
    right_w = abl_arr.shape[1]
    right_h = abl_arr.shape[0]

    total_w = left_w + right_w
    total_h = max(left_h, right_h)

    fig_w_in = 12.0
    fig_h_in = fig_w_in * (total_h / total_w)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    usable_w = 1 - 2 * PAD_OUTER - GAP_COL
    usable_h = 1 - 2 * PAD_OUTER - GAP_ROW

    w_left = usable_w * (left_w / total_w)
    w_right = usable_w * (right_w / total_w)

    x_left = PAD_OUTER
    x_right = PAD_OUTER + w_left + GAP_COL

    h_top = usable_h * (trade_arr.shape[0] / left_h)
    h_bot = usable_h * (ratio_arr.shape[0] / left_h)

    y_bot = PAD_OUTER
    y_top = PAD_OUTER + h_bot + GAP_ROW

    ax_trade = fig.add_axes([x_left,  y_top,  w_left,  h_top])
    ax_ratio = fig.add_axes([x_left,  y_bot,  w_left,  h_bot])
    ax_abl   = fig.add_axes([x_right, PAD_OUTER, w_right, (1 - 2 * PAD_OUTER)])

    for ax, arr in [(ax_trade, trade_arr), (ax_ratio, ratio_arr), (ax_abl, abl_arr)]:
        ax.imshow(arr)
        ax.axis("off")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    print("\n✅ WROTE:", out_pdf)
    print("🗂️ intermediates in:", tmp)
    print("🎉 done. go bully LaTeX.")


if __name__ == "__main__":
    main()
