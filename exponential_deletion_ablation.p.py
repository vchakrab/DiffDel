"""
weight_threshold_ablation.py

Ablation study varying weight cutoff τ (tau) to justify threshold selection.

For each dataset:
- Fix ρ = 1.0
- Vary τ from 0.2 to 1.0 in steps of 0.1
- Filter RDRs by weight ≤ τ
- Compute leakage histogram for all candidate masks
- Track number of surviving RDRs

Outputs:
- CSV with leakage distributions per (dataset, tau)
- Summary statistics
"""

import os
import sys
import time
import csv
import numpy as np
from typing import Dict, List, Set, Tuple, Any
from collections import Counter

# Import from exponential_deletion.py
from exponential_deletion import (
    construct_hypergraph_actual,
    compute_leakage,
    compute_possible_mask_set_str,
    dc_to_hyperedges,
    get_dataset_weights,
    Hypergraph
)
from rtf_core import initialization_phase


def filter_rdrs_by_weight(
    rdrs: List[Tuple[str, ...]],
    weights: List[float],
    tau: float
) -> Tuple[List[Tuple[str, ...]], List[float]]:
    """
    Filter RDRs to only include those with weight ≤ tau.

    Args:
        rdrs: List of RDRs (tuples of attributes)
        weights: Corresponding weights
        tau: Weight threshold

    Returns:
        Filtered RDRs and weights
    """
    filtered_rdrs = []
    filtered_weights = []

    for rdr, weight in zip(rdrs, weights):
        if weight <= tau:
            filtered_rdrs.append(rdr)
            filtered_weights.append(weight)

    return filtered_rdrs, filtered_weights


def compute_leakage_histogram(
    target_cell: str,
    rdrs: List[Tuple[str, ...]],
    weights: List[float],
    rho: float
) -> Dict[str, Any]:
    """
    Compute leakage for all candidate masks and generate histogram data.

    Args:
        target_cell: Target attribute
        rdrs: List of RDRs
        weights: RDR weights
        rho: Safety threshold (fixed at 1.0)

    Returns:
        Dictionary with histogram data and statistics
    """
    # Construct hypergraph
    H = construct_hypergraph_actual(target_cell, rdrs, weights)

    # Generate all candidate masks
    candidates = compute_possible_mask_set_str(target_cell, H)

    if not candidates:
        return {
            'leakages': [],
            'mask_sizes': [],
            'num_candidates': 0,
            'num_vertices': 0,
            'num_edges': 0
        }

    # Compute leakage for each candidate
    leakages = []
    mask_sizes = []

    for mask in candidates:
        leakage = compute_leakage(mask, target_cell, H, rho, debug=False)
        leakages.append(float(leakage))
        mask_sizes.append(len(mask))

    return {
        'leakages': leakages,
        'mask_sizes': mask_sizes,
        'num_candidates': len(candidates),
        'num_vertices': len(H.vertices),
        'num_edges': len(H.edges)
    }


def run_ablation_for_dataset(
    dataset: str,
    key: int,
    target_cell: str,
    tau_values: List[float],
    rho_values: List[float]
) -> List[Dict[str, Any]]:
    """
    Run ablation study for one dataset across all (rho, tau) pairs.

    Args:
        dataset: Dataset name
        key: Record key
        target_cell: Target attribute
        tau_values: List of tau thresholds to test (weight cutoff)
        rho_values: List of rho thresholds to test (safety threshold)

    Returns:
        List of results, one per (rho, tau) pair
    """
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY: {dataset.upper()} - Target: {target_cell}")
    print(f"{'='*80}")

    # Initialize and get base RDRs
    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target_cell},
        dataset,
        0
    )

    base_rdrs, base_weights = dc_to_hyperedges(init_manager)

    print(f"\nBase statistics:")
    print(f"  Total RDRs: {len(base_rdrs)}")
    if base_weights:
        print(f"  Weight range: [{min(base_weights):.4f}, {max(base_weights):.4f}]")
        print(f"  Mean weight: {np.mean(base_weights):.4f}")

    results = []

    # Nested loop: for each rho, test all tau values
    for rho in rho_values:
        print(f"\n{'─'*60}")
        print(f"ρ = {rho:.2f}")
        print(f"{'─'*60}")

        for tau in tau_values:
            print(f"\n--- τ = {tau:.2f}, ρ = {rho:.2f} ---")

            # Filter RDRs by weight threshold
            filtered_rdrs, filtered_weights = filter_rdrs_by_weight(
                base_rdrs, base_weights, tau
            )

            num_surviving = len(filtered_rdrs)
            print(f"  RDRs surviving threshold: {num_surviving}/{len(base_rdrs)}")

            if num_surviving == 0:
                print(f"  ⚠️  No RDRs survive τ = {tau:.2f}, skipping...")
                results.append({
                    'dataset': dataset,
                    'target_cell': target_cell,
                    'tau': tau,
                    'rho': rho,
                    'num_base_rdrs': len(base_rdrs),
                    'num_surviving_rdrs': 0,
                    'num_candidates': 0,
                    'leakages': [],
                    'mask_sizes': [],
                    'error': 'no_rdrs_survived'
                })
                continue

            # Compute leakage histogram
            start_time = time.time()
            histogram_data = compute_leakage_histogram(
                target_cell, filtered_rdrs, filtered_weights, rho
            )
            compute_time = time.time() - start_time

            # Summary statistics
            leakages = histogram_data['leakages']
            if leakages:
                print(f"  Candidates evaluated: {len(leakages)}")
                print(f"  Leakage statistics:")
                print(f"    Min: {min(leakages):.6f}")
                print(f"    Max: {max(leakages):.6f}")
                print(f"    Mean: {np.mean(leakages):.6f}")
                print(f"    Median: {np.median(leakages):.6f}")
                print(f"    Leakage = 1.0: {sum(1 for l in leakages if l >= 0.9999)}")
                print(f"    Leakage < 0.5: {sum(1 for l in leakages if l < 0.5)}")
                print(f"  Computation time: {compute_time:.2f}s")

            results.append({
                'dataset': dataset,
                'target_cell': target_cell,
                'tau': tau,
                'rho': rho,
                'num_base_rdrs': len(base_rdrs),
                'num_surviving_rdrs': num_surviving,
                'num_candidates': histogram_data['num_candidates'],
                'num_vertices': histogram_data['num_vertices'],
                'num_edges': histogram_data['num_edges'],
                'leakages': leakages,
                'mask_sizes': histogram_data['mask_sizes'],
                'compute_time': compute_time
            })

    return results


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """
    Save ablation study results to CSV.

    Creates two CSV files:
    1. Summary statistics per (dataset, tau)
    2. Detailed leakage values for histogram generation
    """
    # Summary CSV
    summary_file = output_file.replace('.csv', '_summary.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'target_cell', 'tau', 'rho',
            'num_base_rdrs', 'num_surviving_rdrs', 'survival_rate',
            'num_candidates', 'num_vertices', 'num_edges',
            'min_leakage', 'max_leakage', 'mean_leakage', 'median_leakage',
            'std_leakage', 'leakage_1.0_count', 'leakage_<0.5_count',
            'compute_time'
        ])

        for result in results:
            if 'error' in result:
                writer.writerow([
                    result['dataset'], result['target_cell'],
                    result['tau'], result['rho'],
                    result['num_base_rdrs'], result['num_surviving_rdrs'],
                    0.0, 0, 0, 0,
                    None, None, None, None, None, None, None, None
                ])
                continue

            leakages = result['leakages']
            survival_rate = result['num_surviving_rdrs'] / result['num_base_rdrs']

            if leakages:
                writer.writerow([
                    result['dataset'], result['target_cell'],
                    result['tau'], result['rho'],
                    result['num_base_rdrs'], result['num_surviving_rdrs'],
                    f"{survival_rate:.4f}",
                    result['num_candidates'], result['num_vertices'],
                    result['num_edges'],
                    f"{min(leakages):.6f}",
                    f"{max(leakages):.6f}",
                    f"{np.mean(leakages):.6f}",
                    f"{np.median(leakages):.6f}",
                    f"{np.std(leakages):.6f}",
                    sum(1 for l in leakages if l >= 0.9999),
                    sum(1 for l in leakages if l < 0.5),
                    f"{result['compute_time']:.2f}"
                ])

    print(f"\n✓ Summary saved to: {summary_file}")

    # Detailed leakage CSV (for histograms)
    detail_file = output_file.replace('.csv', '_detailed.csv')
    with open(detail_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'target_cell', 'tau', 'rho',
            'mask_size', 'leakage'
        ])

        for result in results:
            if 'error' in result or not result['leakages']:
                continue

            for mask_size, leakage in zip(result['mask_sizes'], result['leakages']):
                writer.writerow([
                    result['dataset'], result['target_cell'],
                    result['tau'], result['rho'],
                    mask_size, f"{leakage:.6f}"
                ])

    print(f"✓ Detailed data saved to: {detail_file}")


def main():
    """Run complete ablation study across all datasets."""

    # Configuration
    datasets_config = [
        ('airport', 500, 'home_link'),
        ('adult', 500, 'education'),
        ('flight', 500, 'OriginCityMarketId'),
        ('ncvoter', 500, 'c90'),
        ('hospital', 500, 'EmergencyService')
    ]

    # Grid search over tau and rho
    tau_values = [round(x, 1) for x in np.arange(0.2, 1.1, 0.1)]  # 0.2 to 1.0
    rho_values = [round(x, 1) for x in np.arange(0.2, 1.1, 0.1)]  # 0.2 to 1.0

    print("="*80)
    print("WEIGHT THRESHOLD ABLATION STUDY")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  ρ values: {rho_values}")
    print(f"  τ values: {tau_values}")
    print(f"  Total (ρ, τ) pairs per dataset: {len(rho_values) * len(tau_values)}")
    print(f"  Datasets: {[d[0] for d in datasets_config]}")
    print(f"  Total experiments: {len(datasets_config) * len(rho_values) * len(tau_values)}")

    all_results = []

    # Run ablation for each dataset
    for dataset, key, target_cell in datasets_config:
        try:
            results = run_ablation_for_dataset(
                dataset, key, target_cell, tau_values, rho_values
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n❌ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"ablation_study_rho_tau_grid_{timestamp}.csv"
    save_results_to_csv(all_results, output_file)

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"\nTotal experiments run: {len(all_results)}")
    print(f"Results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = main()