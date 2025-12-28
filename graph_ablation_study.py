import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from io import StringIO

def surprisal(leak: float, cap=20.0) -> float:
    """Calculates surprisal: -log(1 - leak), capped for visualization."""
    leak = min(max(leak, 0.0), 1.0)
    if leak >= 1.0:
        return cap
    return -np.log(max(1e-18, 1.0 - leak))

def plot_all_smoothed_metrics(csv_path, output_dir='plots'):
    """
    Reads the CSV and generates two figures with smoothed curves per dataset:
    1. Epsilon vs. Leakage, Utility, Mask Size, and Surprisal (Original Scale).
    2. Normalized versions of the same plots.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        all_dfs = []
        current_dataset = None
        data_lines = []
        header = 'epsilon,leakage,utility,mask_size\n'

        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith('-----') and line_strip.endswith('-----'):
                if current_dataset and data_lines:
                    csv_string = header + "".join(data_lines)
                    df = pd.read_csv(StringIO(csv_string))
                    df['dataset'] = current_dataset
                    all_dfs.append(df)
                
                current_dataset = line_strip.replace('-', '')
                data_lines = []
            elif current_dataset and ',' in line:
                if 'epsilon,leakage' not in line:
                    data_lines.append(line)
        
        if current_dataset and data_lines:
            csv_string = header + "".join(data_lines)
            df = pd.read_csv(StringIO(csv_string))
            df['dataset'] = current_dataset
            all_dfs.append(df)

        if not all_dfs:
            print("No datasets found or processed in the CSV file.")
            return

        combined_df = pd.concat(all_dfs, ignore_index=True)

        for col in ['epsilon', 'leakage', 'utility', 'mask_size']:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        combined_df.dropna(subset=['epsilon', 'leakage', 'utility', 'mask_size'], inplace=True)
        
        combined_df['surprisal'] = combined_df['leakage'].apply(surprisal)

        metrics_to_normalize = ['leakage', 'utility', 'mask_size', 'surprisal']
        for metric in metrics_to_normalize:
            normalized_col_name = f'{metric}_normalized'
            norm_series = combined_df.groupby('dataset')[metric].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
            )
            combined_df[normalized_col_name] = norm_series

        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        datasets = combined_df['dataset'].unique()
        palette = sns.color_palette("tab10", len(datasets))
        color_map = {dataset: palette[i] for i, dataset in enumerate(datasets)}

        # --- Figure 1: Original Scale with Smoothing ---
        fig1, axes1 = plt.subplots(1, 4, figsize=(28, 6))
        fig1.suptitle('Ablation Study: Epsilon vs. Key Metrics by Dataset (Smoothed)', fontsize=16)
        
        metrics_original = [('leakage', 'Average Leakage'), ('utility', 'Average Utility'), ('mask_size', 'Average Mask Size'), ('surprisal', 'Average Surprisal')]

        for i, (metric, ylabel) in enumerate(metrics_original):
            ax = axes1[i]
            for dataset in datasets:
                subset = combined_df[combined_df['dataset'] == dataset]
                if not subset.empty:
                    color = color_map[dataset]
                    # Plot horizontal line for constant data, otherwise plot smoothed curve
                    if subset[metric].std() < 1e-6: # Use a small tolerance for floating point
                        ax.axhline(y=subset[metric].iloc[0], color=color, linestyle='--', label=f'{dataset} (constant)')
                    else:
                        sns.regplot(data=subset, x='epsilon', y=metric, ax=ax, lowess=True, 
                                    scatter=False, label=dataset, color=color)
            
            ax.set_title(f'Epsilon vs. {metric.replace("_", " ").capitalize()}')
            ax.set_xlabel('Epsilon (ε)')
            ax.set_ylabel(ylabel)
            ax.set_xlim(1, 300)
            ax.legend(title='Dataset')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot1_filename = os.path.join(output_dir, 'final_smoothed_metrics.png')
        plt.savefig(plot1_filename, dpi=300)
        plt.close(fig1)
        print(f"Saved final smoothed metrics plot to {plot1_filename}")

        # --- Figure 2: Normalized Scale with Smoothing ---
        fig2, axes2 = plt.subplots(1, 4, figsize=(28, 6))
        fig2.suptitle('Ablation Study: Epsilon vs. Key Metrics by Dataset (Normalized & Smoothed)', fontsize=16)

        for i, (metric, _) in enumerate(metrics_original):
            ax = axes2[i]
            normalized_metric = f'{metric}_normalized'
            
            for dataset in datasets:
                subset = combined_df[combined_df['dataset'] == dataset].copy() # Use .copy() to avoid SettingWithCopyWarning
                # Recalculate if there are NaN from normalization of surprisal
                if subset[normalized_metric].isnull().any():
                     subset[normalized_metric] = (subset[metric] - subset[metric].min()) / (subset[metric].max() - subset[metric].min()) if (subset[metric].max() - subset[metric].min()) > 0 else 0.5
                
                if not subset.empty:
                    color = color_map[dataset]
                    if subset[normalized_metric].std() < 1e-6:
                        ax.axhline(y=subset[normalized_metric].iloc[0], color=color, linestyle='--', label=f'{dataset} (constant)')
                    else:
                        sns.regplot(data=subset, x='epsilon', y=normalized_metric, ax=ax, lowess=True, 
                                    scatter=False, label=dataset, color=color)

            ax.set_title(f'Epsilon vs. Normalized {metric.replace("_", " ").capitalize()}')
            ax.set_xlabel('Epsilon (ε)')
            ax.set_ylabel('Normalized Value')
            ax.set_xlim(1, 300)
            ax.legend(title='Dataset')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot2_filename = os.path.join(output_dir, 'final_smoothed_normalized_metrics.png')
        plt.savefig(plot2_filename, dpi=300)
        plt.close(fig2)
        print(f"Saved final smoothed normalized metrics plot to {plot2_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    csv_file = 'delgum_data_epsilon_leakage_graph_v8_gpt_fixed_script.csv'
    plot_all_smoothed_metrics(csv_file)
