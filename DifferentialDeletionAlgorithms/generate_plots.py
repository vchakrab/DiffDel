import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =============================================================================
# CONSISTENT COLOR SCHEME ACROSS ALL FIGURES
# =============================================================================

# Dataset colors (used in scatter plots)
DATASET_COLORS = {
    'Airport': '#1f77b4',  # blue
    'Hospital': '#ff7f0e',  # orange
    'Adult': '#2ca02c',  # green
    'Flight': '#d62728',  # red
    'Tax': '#9467bd',  # purple
}

# Method markers (used in scatter plots)
MARKERS = {
    'DelMin': 's',  # square
    'Del2Ph': '^',  # triangle
    'DelGum': 'o',  # circle
}

# Method line styles (used in radar plots)
LINESTYLES = {
    'DelMin': '-',
    'Del2Ph': '--',
    'DelGum': ':',
}

# Method colors (used in radar plots)
METHOD_COLORS = {
    'DelMin': '#2ecc71',  # green
    'Del2Ph': '#3498db',  # blue
    'DelGum': '#e74c3c',  # red
}

DATASET_ORDER = ['Airport', 'Hospital', 'Adult', 'Flight', 'Tax']

CONSTRAINTS_025 = {
    'Airport': 7, 'Hospital': 40, 'Adult': 57, 'Flight': 112, 'Tax': 31
}

ZONE_SIZES = {
    'Airport': 6, 'Hospital': 12, 'Adult': 12, 'Flight': 16, 'Tax': 7
}


def load_data(data_dir):
    delmin = pd.read_csv(data_dir / 'delmin_final_data.csv')
    del2ph = pd.read_csv(data_dir / 'del2ph_January11,202608:05:46PM_original.csv')
    delgum = pd.read_csv(data_dir / 'delgum_January11,202608:07:25PM_original.csv')

    delmin['method'] = 'DelMin'
    del2ph['method'] = 'Del2Ph'
    delgum['method'] = 'DelGum'

    df = pd.concat([delmin, del2ph, delgum], ignore_index=True)
    df['dataset'] = df['dataset'].str.capitalize()
    df['total_time_ms'] = df['total_time'] * 1000
    df['init_time_ms'] = df['init_time'] * 1000
    df['model_time_ms'] = df['model_time'] * 1000
    df['update_time_ms'] = df['update_time'] * 1000
    df['memory_kb'] = df['memory_overhead_bytes'] / 1024
    df['auxiliary_deletions'] = df['mask_size']

    return df


def fig_deletion_ratio_vs_constraints(df, output_path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    summary = df.groupby(['dataset', 'method']).agg({
        'auxiliary_deletions': 'mean',
    }).reset_index()

    for dataset in DATASET_ORDER:
        for method in ['DelMin', 'Del2Ph', 'DelGum']:
            row = summary[(summary['dataset'] == dataset) & (summary['method'] == method)]
            if len(row) == 0:
                continue

            constraints = CONSTRAINTS_025[dataset]
            zone_size = ZONE_SIZES[dataset]
            deletion_ratio = row['auxiliary_deletions'].values[0] / zone_size

            ax.scatter(constraints, deletion_ratio,
                       s=80,
                       c=DATASET_COLORS[dataset],
                       marker=MARKERS[method],
                       alpha=0.85,
                       edgecolor='black', linewidth=0.5)

    ax.set_xlabel(r'Number of Constraints ($\gamma=0.25$)')
    ax.set_ylabel(r'Deletion Ratio $|M|/|\mathcal{I}(c^*)|$')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-5, 125)

    dataset_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DATASET_COLORS[d],
               markeredgecolor='black', markersize=7, label=d, linestyle='None')
        for d in DATASET_ORDER]
    method_handles = [Line2D([0], [0], marker=MARKERS[m], color='w', markerfacecolor='gray',
                             markeredgecolor='black', markersize=7, label=m,
                             linestyle='None')
                      for m in ['DelMin', 'Del2Ph', 'DelGum']]

    all_handles = dataset_handles + [Line2D([0], [0], color='none', label='')] + method_handles
    all_labels = DATASET_ORDER + [''] + ['DelMin', 'Del2Ph', 'DelGum']

    ax.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=7, frameon=True, framealpha=0.9, borderpad=0.5,
              handletextpad=0.3, labelspacing=0.4)

    plt.tight_layout()
    plt.savefig(output_path / 'fig_deletion_ratio.pdf', bbox_inches='tight')
    plt.close()


def fig_leakage_vs_mask_tradeoff(df, output_path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    summary = df.groupby(['dataset', 'method']).agg({
        'auxiliary_deletions': 'mean',
        'leakage': 'mean',
    }).reset_index()

    for dataset in DATASET_ORDER:
        for method in ['DelMin', 'Del2Ph', 'DelGum']:
            row = summary[(summary['dataset'] == dataset) & (summary['method'] == method)]
            if len(row) == 0:
                continue

            ax.scatter(row['leakage'].values[0], row['auxiliary_deletions'].values[0],
                       s=80,
                       c=DATASET_COLORS[dataset],
                       marker=MARKERS[method],
                       alpha=0.85,
                       edgecolor='black', linewidth=0.5)

    ax.annotate('', xy=(0.38, 3), xytext=(0.02, 11),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.22, 8, 'Tradeoff', fontsize=7, color='gray', ha='center', style='italic')

    ax.set_xlabel(r'Leakage $\mathcal{L}$')
    ax.set_ylabel(r'Auxiliary Deletions $|M|$')
    ax.set_xlim(-0.03, 0.55)
    ax.set_ylim(0, 17)

    dataset_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DATASET_COLORS[d],
               markeredgecolor='black', markersize=7, label=d, linestyle='None')
        for d in DATASET_ORDER]
    method_handles = [Line2D([0], [0], marker=MARKERS[m], color='w', markerfacecolor='gray',
                             markeredgecolor='black', markersize=7, label=m,
                             linestyle='None')
                      for m in ['DelMin', 'Del2Ph', 'DelGum']]

    all_handles = dataset_handles + [Line2D([0], [0], color='none', label='')] + method_handles
    all_labels = DATASET_ORDER + [''] + ['DelMin', 'Del2Ph', 'DelGum']

    ax.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=7, frameon=True, framealpha=0.9, borderpad=0.5,
              handletextpad=0.3, labelspacing=0.4)

    plt.tight_layout()
    plt.savefig(output_path / 'fig_leakage_tradeoff.pdf', bbox_inches='tight')
    plt.close()


def fig_radar_plots(df, output_path):
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.8), subplot_kw=dict(polar=True))

    n_metrics = 5
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    for ax, dataset in zip(axes, DATASET_ORDER):
        ds_data = df[df['dataset'] == dataset]

        raw_values = {}
        for method in ['DelMin', 'Del2Ph', 'DelGum']:
            m_data = ds_data[ds_data['method'] == method]
            paths = m_data['total_paths'].mean()
            raw_values[method] = [
                m_data['mask_size'].mean(),
                m_data['leakage'].mean(),
                np.log10(m_data['memory_kb'].mean() + 1),
                np.log10(m_data['total_time_ms'].mean() + 1),
                np.log10(max(paths, 1) + 1),
            ]

        normalized = {m: [0] * 5 for m in ['DelMin', 'Del2Ph', 'DelGum']}
        for i in range(5):
            vals = [raw_values[m][i] for m in ['DelMin', 'Del2Ph', 'DelGum']]
            min_v, max_v = min(vals), max(vals)
            for method in ['DelMin', 'Del2Ph', 'DelGum']:
                if max_v - min_v > 1e-9:
                    normalized[method][i] = (raw_values[method][i] - min_v) / (max_v - min_v)
                else:
                    normalized[method][i] = 0.5

        for method in ['DelMin', 'Del2Ph', 'DelGum']:
            values = normalized[method] + [normalized[method][0]]
            ax.plot(angles, values, linestyle=LINESTYLES[method], linewidth=2,
                    color=METHOD_COLORS[method], marker=MARKERS[method], markersize=5)
            ax.fill(angles, values, alpha=0.12, color=METHOD_COLORS[method])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['|M|', r'$\mathcal{L}$', 'Mem', 'T', r'$|\Pi|$'], size=7)
        ax.set_title(dataset, size=9, fontweight='bold', pad=10)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color=METHOD_COLORS['DelMin'], linewidth=2, linestyle='-',
               marker='s', markersize=5, label='DelMin'),
        Line2D([0], [0], color=METHOD_COLORS['Del2Ph'], linewidth=2, linestyle='--',
               marker='^', markersize=5, label='Del2Ph'),
        Line2D([0], [0], color=METHOD_COLORS['DelGum'], linewidth=2, linestyle=':',
               marker='o', markersize=5, label='DelGum'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.01), frameon=False, columnspacing=3)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path / 'fig_radar_plots.pdf', bbox_inches='tight')
    plt.close()


def fig_runtime_decomposition(df, output_path):
    """
    Section 6.3: Runtime decomposition stacked bar chart (log y),
    PLUS: a 4th bar per dataset showing mean num_instantiated_cells (log y on right axis),
    with mask_size "shaded" inside it as a fraction of the inference zone size.

    - Left axis: time (ms), stacked by phase, grouped by method
    - Right axis: instantiated cells (count), one bar per dataset
      * shaded portion = cells * (mean_mask_size / zone_size)
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    # --- time summary (per dataset, per method) ---
    summary_time = df.groupby(['dataset', 'method']).agg({
        'init_time_ms': 'mean',
        'model_time_ms': 'mean',
        'update_time_ms': 'mean',
    }).reset_index()

    PHASE_COLORS = {
        'init': '#3498db',
        'model': '#e74c3c',
        'update': '#9b59b6',
    }

    n_datasets = len(DATASET_ORDER)
    x = np.arange(n_datasets)

    methods = ['DelMin', 'Del2Ph', 'DelGum']
    width = 0.22
    hatches = ['', '//', '\\\\']

    for i, method in enumerate(methods):
        method_data = summary_time[summary_time['method'] == method].set_index('dataset')
        method_data = method_data.reindex(DATASET_ORDER)

        init = method_data['init_time_ms'].values
        model = method_data['model_time_ms'].values
        update = method_data['update_time_ms'].values

        pos = x + (i - 1) * width

        ax.bar(pos, init, width, color=PHASE_COLORS['init'],
               edgecolor='black', linewidth=0.5, hatch=hatches[i])
        ax.bar(pos, model, width, bottom=init, color=PHASE_COLORS['model'],
               edgecolor='black', linewidth=0.5, hatch=hatches[i])
        ax.bar(pos, update, width, bottom=init + model, color=PHASE_COLORS['update'],
               edgecolor='black', linewidth=0.5, hatch=hatches[i])

    ax.set_yscale('log')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)

    # --- instantiated cells bar (per dataset, right axis) ---
    # We aggregate across methods to produce ONE bar per dataset (as you requested).
    if 'num_instantiated_cells' not in df.columns:
        raise KeyError("CSV is missing required column: num_instantiated_cells")
    if 'mask_size' not in df.columns:
        raise KeyError("CSV is missing required column: mask_size")

    summary_cells = df.groupby('dataset').agg({
        'num_instantiated_cells': 'mean',
        'mask_size': 'mean',
    }).reindex(DATASET_ORDER)

    cells = summary_cells['num_instantiated_cells'].values.astype(float)

    # fraction of zone masked (clipped to [0,1] so shading stays within bar)
    zone_sizes = np.array([float(ZONE_SIZES[d]) for d in DATASET_ORDER], dtype=float)
    mask_mean = summary_cells['mask_size'].values.astype(float)
    mask_frac = np.clip(mask_mean / np.maximum(zone_sizes, 1.0), 0.0, 1.0)

    shaded = cells * mask_frac

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('Instantiated cells (count)')

    # Place this "extra bar" to the right of the 3 method bars
    cells_width = 0.18
    cells_pos = x + 2.0 * width  # after DelGum bar

    # base cells bar
    ax2.bar(cells_pos, cells, cells_width,
            color='#dddddd', edgecolor='black', linewidth=0.6,
            label='Num instantiated cells (avg)')

    # shaded portion representing mask_size fraction of zone
    # (drawn as an overlay bar starting from 0)
    ax2.bar(cells_pos, shaded, cells_width,
            color='#ffffff', edgecolor='black', linewidth=0.6,
            hatch='....', alpha=0.65,
            label='Mask size shading (|M|/|I|)')

    # --- legends ---
    method_patches = [Patch(facecolor='white', edgecolor='black', hatch=h, label=m)
                      for h, m in zip(hatches, methods)]
    phase_patches = [Patch(color=PHASE_COLORS[p], label=p.capitalize())
                     for p in ['init', 'model', 'update']]

    # Right-axis legend items
    cells_patch = Patch(facecolor='#dddddd', edgecolor='black',
                        label='Num instantiated cells (avg)')
    shade_patch = Patch(facecolor='#ffffff', edgecolor='black', hatch='....',
                        label='Shaded: |M|/|I| of cells')

    leg1 = ax.legend(handles=method_patches, loc='upper left', fontsize=6,
                     title='Method', title_fontsize=6, framealpha=0.9)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=phase_patches, loc='upper center', fontsize=6,
                     title='Phase', title_fontsize=6, framealpha=0.9)

    # Put the cells legend on the right side
    ax2.legend(handles=[cells_patch, shade_patch], loc='upper right', fontsize=6,
               title='Cells', title_fontsize=6, framealpha=0.9)

    # Make sure everything fits
    ax.set_xlim(-0.6, (n_datasets - 1) + 0.6 + 2.0 * width)

    plt.tight_layout()
    plt.savefig(output_path / 'fig_runtime_decomposition.pdf')
    plt.close()

def fig_cells_masked_bars(df, output_path):
    """
    3 bars per dataset (DelMin/Del2Ph/DelGum) for mean num_instantiated_cells.
    Overlay a "masked cells" portion = cells * (mean_mask_size / zone_size).

    Hatch patterns:
      - Base bars: method-specific hatch
      - Shaded overlay: different method-specific hatch (so each method’s shading differs)
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    if 'num_instantiated_cells' not in df.columns:
        raise KeyError("CSV is missing required column: num_instantiated_cells")
    if 'mask_size' not in df.columns:
        raise KeyError("CSV is missing required column: mask_size")

    methods = ['DelMin', 'Del2Ph', 'DelGum']
    n_datasets = len(DATASET_ORDER)
    x = np.arange(n_datasets)

    # Summaries per (dataset, method)
    summary = df.groupby(['dataset', 'method']).agg({
        'num_instantiated_cells': 'mean',
        'mask_size': 'mean',
    }).reset_index()

    # Method-specific patterns for base + shaded overlay
    BASE_HATCH = {
        'DelMin': '',
        'Del2Ph': '///',
        'DelGum': 'xxx',
    }
    SHADE_HATCH = {
        'DelMin': '....',
        'Del2Ph': '\\\\\\\\',
        'DelGum': '++',
    }

    width = 0.25

    for i, method in enumerate(methods):
        m = summary[summary['method'] == method].set_index('dataset').reindex(DATASET_ORDER)

        cells = m['num_instantiated_cells'].values.astype(float)
        mask_mean = m['mask_size'].values.astype(float)

        zone_sizes = np.array([float(ZONE_SIZES[d]) for d in DATASET_ORDER], dtype=float)
        frac = np.clip(mask_mean / np.maximum(zone_sizes, 1.0), 0.0, 1.0)

        masked_cells = cells * frac

        pos = x + (i - 1) * width

        # Base instantiated cells bar
        ax.bar(
            pos, cells, width,
            color='#dddddd', edgecolor='black', linewidth=0.6,
            hatch=BASE_HATCH[method], alpha=0.95
        )

        # Shaded masked portion overlay (different hatch per method)
        ax.bar(
            pos, masked_cells, width,
            color='#ffffff', edgecolor='black', linewidth=0.6,
            hatch=SHADE_HATCH[method], alpha=0.80
        )

    ax.set_yscale('log')
    ax.set_ylabel('Instantiated cells (count)')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)

    # Legends
    method_handles = [
        Patch(facecolor='white', edgecolor='black', hatch=BASE_HATCH[m], label=m)
        for m in methods
    ]
    shaded_handles = [
        Patch(facecolor='white', edgecolor='black', hatch=SHADE_HATCH[m], label=f'{m} shaded')
        for m in methods
    ]

    leg1 = ax.legend(handles=method_handles, loc='upper left',
                     fontsize=6, title='Method (base hatch)', title_fontsize=6,
                     framealpha=0.9)
    ax.add_artist(leg1)

    ax.legend(handles=shaded_handles, loc='upper right',
              fontsize=6, title='Masked overlay hatch', title_fontsize=6,
              framealpha=0.9)

    ax.set_xlim(-0.6, (n_datasets - 1) + 0.6)

    plt.tight_layout()
    plt.savefig(output_path / 'fig_cells_masked_bars.pdf')
    plt.close()

def main():
    data_dir = Path('.')  # Update path as needed
    output_path = Path('.')

    print("Loading data...")
    df = load_data(data_dir)

    print("Generating figures...")
    fig_deletion_ratio_vs_constraints(df, output_path)
    fig_leakage_vs_mask_tradeoff(df, output_path)
    fig_radar_plots(df, output_path)
    fig_runtime_decomposition(df, output_path)
    fig_cells_masked_bars(df, output_path)

    print("Done!")


if __name__ == '__main__':
    main()
