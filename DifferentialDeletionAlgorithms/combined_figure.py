#!/usr/bin/env python3
"""
Combine:
  - Radar plot from generate_plots.py (fig_radar_plots)
  - Faceted runtime plot from generate_runtime_plot.py (generate_runtime_figure)

into ONE stacked figure: radar on top, runtime facets below.

How it works:
  1) Monkeypatch plt.savefig so when either script saves a *.pdf, we ALSO save a *.png.
  2) Call the two plotting functions as-is (no editing needed).
  3) Read the two PNGs back and stack them in a final combined PDF/PNG.

Usage:
  Put this file in the SAME directory as:
    - generate_plots.py
    - generate_runtime_plot.py
    - the CSVs those scripts expect (same as when you run them normally)

  Then run:
    python3 combine_radar_and_runtime.py

Outputs:
  - fig_radar_plus_runtime_facets.pdf
  - fig_radar_plus_runtime_facets.png
"""

from __future__ import annotations

from pathlib import Path
import importlib

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _patch_savefig_to_also_save_png() -> None:
    """Whenever someone calls plt.savefig('*.pdf'), also save a PNG sibling."""
    orig_savefig = plt.savefig

    def wrapped_savefig(fname, *args, **kwargs):
        orig_savefig(fname, *args, **kwargs)

        s = str(fname)
        if s.lower().endswith(".pdf"):
            png_name = s[:-4] + ".png"
            orig_savefig(png_name, *args, **kwargs)

    plt.savefig = wrapped_savefig  # type: ignore[assignment]


def combine(output_dir: Path = Path(".")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    gp = importlib.import_module("generate_plots")
    gr = importlib.import_module("generate_runtime_plot")

    _patch_savefig_to_also_save_png()

    # --- Radar (generate_plots.py) ---
    df_radar = gp.load_data(Path("."))  # same behavior as gp.main()
    gp.fig_radar_plots(df_radar, output_dir)

    radar_png = output_dir / "fig_radar_plots.png"

    # --- Runtime facets (generate_runtime_plot.py) ---
    df_runtime = gr.load_data(
        gr.DATA_DIR / gr.DELMIN_CSV,
        gr.DATA_DIR / gr.DEL2PH_CSV,
        gr.DATA_DIR / gr.DELGUM_CSV,
    )
    gr.generate_runtime_figure(df_runtime, output_dir)

    runtime_png = output_dir / "fig_runtime_faceted.png"

    if not radar_png.exists():
        raise FileNotFoundError(f"Expected radar PNG at: {radar_png}")
    if not runtime_png.exists():
        raise FileNotFoundError(f"Expected runtime PNG at: {runtime_png}")

    radar_img = mpimg.imread(radar_png)
    runtime_img = mpimg.imread(runtime_png)

    fig = plt.figure(figsize = (10, 6))

    # Radar row
    ax_top = fig.add_axes([0.03, 0.38, 0.94, 0.44])

    # Runtime row (touches radar)
    ax_bot = fig.add_axes([0.03, 0.02, 0.94, 0.44])
    ax_top.imshow(radar_img)
    ax_top.axis("off")

    ax_bot.imshow(runtime_img)
    ax_bot.axis("off")

    out_pdf = output_dir / "fig_radar_plus_runtime_facets.pdf"
    out_png = output_dir / "fig_radar_plus_runtime_facets.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print("\nSaved combined figure:")
    print(f"  - {out_pdf}")
    print(f"  - {out_png}")


if __name__ == "__main__":
    combine(Path("."))
