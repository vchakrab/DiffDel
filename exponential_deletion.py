#!/usr/bin/env python3
"""
Differential Deletion Mechanisms
==================================
... (imports and placeholder classes remain the same)
"""

import sys
import os
from typing import Any, Dict, List, Set, Tuple, FrozenSet
from collections import defaultdict, deque, Counter
from itertools import chain, combinations
import time
from sys import getsizeof
import numpy as np
from importlib import import_module
import mysql.connector
from mysql.connector import Error


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    print("Warning: 'config.py' not found. Database operations will fail.")
    config = None

# The following imports are assumed to be available in the execution environment
try:
    from old_files.cell import Attribute, Cell, Hyperedge
    from fetch_row import RTFDatabaseManager
    from InferenceGraph.bulid_hyperedges import HyperedgeBuilder
    from InferenceGraph.build_hypergraph import build_hypergraph_tree, GraphNode
except ImportError as e:
    pass
    # Define placeholder classes/functions for running/inspection purposes


def powerset(iterable):
    """
    Computes the powerset of an iterable.
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_inference_paths_str(hyperedges: List[Tuple[str, ...]],
                             target_cell: str,
                             initial_known: Set[str] = None) -> List[List[int]]:
    # ... (find_inference_paths_str remains unchanged)
    if initial_known is None:
        raise ValueError("initial_known must be provided")

    all_paths = []
    seen_paths = set()

    def dfs(known_cells: Set[str], used_edges: Set[int], current_path: List[int],
            can_assume_known: Set[str]):

        for edge_idx, edge in enumerate(hyperedges):
            if edge_idx in used_edges:
                continue

            for inferred_cell in edge:
                if inferred_cell in known_cells:
                    continue

                other_cells = [c for c in edge if c != inferred_cell]

                if all(c in known_cells for c in other_cells):
                    new_known = known_cells | {inferred_cell}
                    new_used = used_edges | {edge_idx}
                    new_path = current_path + [edge_idx]

                    if inferred_cell == target_cell:
                        path_tuple = tuple(new_path)
                        if path_tuple not in seen_paths:
                            seen_paths.add(path_tuple)
                            all_paths.append(new_path)

                    dfs(new_known, new_used, new_path, can_assume_known)

                elif inferred_cell == target_cell:
                    unknown_cells = [c for c in other_cells if c not in known_cells]
                    if all(c in can_assume_known for c in unknown_cells):
                        new_path = current_path + [edge_idx]
                        path_tuple = tuple(new_path)
                        if path_tuple not in seen_paths:
                            seen_paths.add(path_tuple)
                            all_paths.append(new_path)

    all_cells = set()
    for edge in hyperedges:
        all_cells.update(edge)
    potentially_inferrable = all_cells - initial_known - {target_cell}

    dfs(initial_known, set(), [], potentially_inferrable)
    return all_paths


def filter_active_paths_str(hyperedges, paths, mask, initial_known, target_cell=None):
    active_paths = []

    # Use all attributes known initially for path calculation
    all_attributes = set(c for he in hyperedges for c in he)

    for path in paths:
        known_so_far = all_attributes - mask
        is_blocked = False

        for edge_idx in path:
            edge = set(hyperedges[edge_idx])
            unknown_in_edge = edge - known_so_far

            if len(unknown_in_edge) == 0:
                continue
            elif len(unknown_in_edge) == 1:
                known_so_far.update(unknown_in_edge)
            else:
                if target_cell is not None and target_cell in unknown_in_edge:
                    known_so_far.add(target_cell)
                else:
                    is_blocked = True
                    break

        if not is_blocked:
            active_paths.append(path)

    return active_paths


def compute_product_leakage_str(active_paths: List[List[int]],
                                hyperedges: List[Tuple[str, ...]],
                                edge_weights: Dict[int, float]) -> float:
    # ... (compute_product_leakage_str remains unchanged)
    if edge_weights is None:
        edge_weights = {i: 1.0 for i in range(len(hyperedges))}

    if len(active_paths) == 0:
        return 0.0

    product = 1.0
    for path in active_paths:
        path_weight = 1.0
        for edge_idx in path:
            path_weight *= edge_weights[edge_idx]

        product *= (1 - path_weight)

    leakage = 1 - product
    return leakage


def calculate_leakage_str(hyperedges: List[Tuple[str, ...]],
                          paths: List[List[int]],
                          mask: Set[str],
                          target_cell: str,
                          initial_known: Set[str],
                          edge_weights: Dict[int, float] = None) -> float:
    # ... (calculate_leakage_str remains unchanged)
    active_paths = filter_active_paths_str(
        hyperedges,
        paths,
        mask,
        initial_known
    )

    leakage = compute_product_leakage_str(
        active_paths,
        hyperedges,
        edge_weights
    )

    return leakage


def get_path_inference_zone_str(paths: List[List[int]], hyperedges: List[Tuple[str, ...]],
                                target_cell: str) -> Set[str]:
    # ... (get_path_inference_zone_str remains unchanged)
    all_cells_in_paths = set()

    for path in paths:
        for edge_idx in path:
            edge = hyperedges[edge_idx]
            all_cells_in_paths.update(edge)

    path_zone = all_cells_in_paths - {target_cell}

    return path_zone


def compute_possible_mask_set_str(inference_zone_str: Set[str]) -> List[Set[str]]:
    # ... (compute_possible_mask_set_str remains unchanged)
    all_subsets = [set(subset) for subset in powerset(inference_zone_str)]
    return all_subsets


def compute_utility_str(
        mask: Set[str],
        target_cell: str,
        paths: List[List[int]],
        hyperedges: List[Tuple[str, ...]],
        initial_known: Set[str],
        edge_weights: Dict[int, float],
        alpha: float,
        beta: float
) -> float:
    """
    Compute mask utility u(M, c_t) for the string-based approach.

    u(M, c_t) = -α · L(M, c_t) - β · |M|
    """
    leakage = calculate_leakage_str(
        hyperedges,
        paths,
        mask,
        target_cell,
        initial_known,
        edge_weights
    )
    # The term |M| is simply the size of the deletion mask (set of strings)
    utility = -alpha * leakage - beta * len(mask)
    return utility


def exponential_mechanism_sample_str(
        candidates: List[Set[str]],
        target_cell: str,
        paths: List[List[int]],
        hyperedges: List[Tuple[str, ...]],
        initial_known: Set[str],
        edge_weights: Dict[int, float],
        alpha: float,
        beta: float,
        epsilon: float
) -> Set[str]:
    """
    Sample a mask using the exponential mechanism for the string-based approach.

    Probability ∝ exp(ε · u(M, c_t) / (2α))
    """
    if not candidates:
        return set()

    # 1. Compute utilities for all candidates
    utilities = []
    for mask in candidates:
        u = compute_utility_str(
            mask,
            target_cell,
            paths,
            hyperedges,
            initial_known,
            edge_weights,
            alpha,
            beta
        )
        utilities.append(u)

    utilities = np.array(utilities)

    # 2. Compute probabilities using the exponential mechanism
    # Scores = ε * u(M, c_t) / (2α)
    scores = epsilon * utilities / (2 * alpha)

    # Use log-sum-exp trick for numerical stability
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    probabilities = exp_scores / np.sum(exp_scores)

    # 3. Sample according to probabilities
    selected_idx = np.random.choice(len(candidates), p = probabilities)
    return candidates[selected_idx]


def measure_memory_overhead_str(hyperedges, paths, candidate_masks):
    """
    Calculates the approximate memory overhead of the key data structures.
    This is a Python-specific way to estimate memory, summing the size of the
    main lists and their contained elements.
    """
    memory = getsizeof(hyperedges)
    for he in hyperedges:
        memory += getsizeof(he)
        for attr in he:
            memory += getsizeof(attr)

    memory += getsizeof(paths)
    for path in paths:
        memory += getsizeof(path)

    memory += getsizeof(candidate_masks)
    for mask in candidate_masks:
        memory += getsizeof(mask)
        for attr in mask:
            memory += getsizeof(attr)
    return memory


def clean_raw_dcs(raw_dcs: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
    """
    Cleans raw denial constraints into simple hyperedges (tuples of attribute strings).
    Example: [('t1.education', '!=', 't2.education'), ('t1.education_num', '==', 't2.education_num')]
             --> ('education', 'education_num')
    """
    cleaned_hyperedges = []
    for dc in raw_dcs:
        attributes = set()
        for pred in dc:
            # pred is like ('t1.education', '!=', 't2.education')
            # or ('t1.age', '==', '30')
            for item in [pred[0], pred[2]]:
                if '.' in item:  # It's an attribute like 't1.education'
                    attr_name = item.split('.')[-1]
                    attributes.add(attr_name)
        if attributes:
            cleaned_hyperedges.append(tuple(sorted(list(attributes))))
    return cleaned_hyperedges


def exponential_deletion_main(dataset: str, key: int, target_cell: str):
    """
    Main orchestrator for the string-based exponential deletion mechanism.
    Captures and returns a dictionary of performance and result metrics.
    Deletions are performed on the '{dataset}_copy_data' table.
    """
    print("\n" + "=" * 70)
    print(
        f"Running Main Orchestrator for Dataset: '{dataset}', Key: {key}, Target: '{target_cell}'")
    print("=" * 70)

    # --- Initialization Phase ---
    init_start = time.time()
    try:
        # Determine the correct casing for the dataset module name
        if dataset == 'ncvoter':
            dataset_module_name = 'NCVoter'
        else:
            dataset_module_name = dataset.capitalize()

        dc_module_path = f'DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed'
        dc_module = __import__(dc_module_path, fromlist = ['denial_constraints'])
        raw_dcs = dc_module.denial_constraints
        print(f"Successfully loaded {len(raw_dcs)} raw denial constraints for '{dataset}'.")
    except ImportError:
        print(f"Error: Could not find or load denial constraints for dataset '{dataset}'.")
        print(f"Attempted to load module: {dc_module_path}")  # Added for better debugging
        return None
    hyperedges = clean_raw_dcs(raw_dcs)
    print("Cleaned DCs into simple hyperedges.")
    init_time = time.time() - init_start

    # --- Modeling Phase ---
    model_start = time.time()
    all_attributes = set(attr for he in hyperedges for attr in he)
    attribute_counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_cells = {attr for attr, count in attribute_counts.items() if count > 1}
    initial_known = all_attributes - {target_cell} - intersecting_cells

    # --- DEBUGGING FOR NCVOTER ---
    if dataset == 'ncvoter':
        print("\n--- NCVOTER DEBUG ---")
        print(f"Target Cell: {target_cell}")
        print(f"All Attributes ({len(all_attributes)}): {all_attributes}")
        print(f"Intersecting Cells ({len(intersecting_cells)}): {intersecting_cells}")
        print(f"Final Initial Known Set: {initial_known}")
        print("---------------------\n")
    # --- END DEBUGGING ---

    paths = find_inference_paths_str(hyperedges, target_cell, initial_known)
    num_paths = len(paths)

    if dataset == 'ncvoter':
        print(f"\n--- NCVOTER DEBUG ---")
        print(f"Number of paths found: {num_paths}")
        print("---------------------\n")

    inference_zone = get_path_inference_zone_str(paths, hyperedges, target_cell)
    candidate_masks = compute_possible_mask_set_str(inference_zone)

    memory_overhead = measure_memory_overhead_str(hyperedges, paths, candidate_masks)

    # --- Execution/Sampling (Part of Modeling Time) ---
    edge_weights = {i: 0.8 for i in range(len(hyperedges))}
    alpha = 1.0
    beta = 0.5
    epsilon = 1.0

    final_mask = exponential_mechanism_sample_str(
        candidate_masks, target_cell, paths, hyperedges, initial_known,
        edge_weights, alpha, beta, epsilon
    )
    mask_size = len(final_mask)
    model_time = time.time() - model_start

    # --- Metrics Calculation for Chosen Mask ---
    leakage = calculate_leakage_str(
        hyperedges, paths, final_mask, target_cell, initial_known, edge_weights
    )
    utility = compute_utility_str(
        final_mask, target_cell, paths, hyperedges, initial_known,
        edge_weights, alpha, beta
    )

    # --- Database Update Phase ---
    del_start = time.time()
    if config:
        cells_to_delete = final_mask | {target_cell}
        conn = None
        cursor = None
        try:
            db_details = config.get_database_config(dataset)
            primary_table = f"{dataset}_copy_data"
            conn = mysql.connector.connect(
                host = db_details['host'], user = db_details['user'],
                password = db_details['password'],
                database = db_details['database'],
                ssl_disabled = db_details.get('ssl_disabled', True)
            )
            if conn.is_connected():
                cursor = conn.cursor()
                DELETION_QUERY = "UPDATE {table_name} SET `{column_name}` = NULL WHERE id = {key};"
                for cell_to_delete in cells_to_delete:
                    query = DELETION_QUERY.format(
                        table_name = primary_table, column_name = cell_to_delete, key = key
                    )
                    cursor.execute(query)
                conn.commit()
        except Error as e:
            print(f"Error during database update: {e}")
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
    del_time = time.time() - del_start

    # --- Compile and Return Results ---
    results = {
        'init_time': init_time,
        'model_time': model_time,
        'del_time': del_time,
        'leakage': leakage,
        'utility': utility,
        'mask_size': mask_size,
        'final_mask': final_mask,
        'num_paths': num_paths,
        'memory_overhead_bytes': memory_overhead,
        'num_instantiated_cells': len(inference_zone)
    }

    print("\n" + "=" * 70)
    print("Main Orchestrator Finished. Results:")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  - {key}: {val:.4f}")
        else:
            print(f"  - {key}: {val}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    # Run the main orchestrator for a single test case
    results = exponential_deletion_main(
        dataset = 'adult',
        key = 2,
        target_cell = 'education'
    )

