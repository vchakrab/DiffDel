"""
Implementation of the Greedy Gumbel-Max Deletion Mechanism (Algorithm 4)
from the paper "Differential Inference Deletion (DID)".
"""
import math
import random
import sys
import os
import time
from sys import getsizeof
from collections import Counter
from importlib import import_module
import mysql.connector
from mysql.connector import Error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    print("Warning: 'config.py' not found. Database operations will fail.")
    config = None

# Import helper functions from the the exact exponential mechanism implementation
# The original file's design for these implies they exist elsewhere or are part of this file.
# For this revert, we'll restore its original simplified versions if they were internal.
# Based on prior reading, it referred to 'exponential_mechanism' for these, so we'll re-add that import.

# --- Reverted: Re-import original exponential_mechanism functions ---
from exponential_deletion import clean_raw_dcs, find_inference_paths_str, calculate_leakage_str

def sample_gumbel(mu=0, beta=1):
    """Sample from Gumbel(mu, beta)"""
    u = random.uniform(0, 1)
    # Ensure u is not 0 or 1 to avoid log(0) issues
    if u == 0: u = 1e-10
    if u == 1: u = 1 - 1e-10
    return mu - beta * math.log(-math.log(u))

# --- Reverted: Original greedy_gumbel_max_deletion with adapted parameters ---
def greedy_gumbel_max_deletion(hyperedges, paths, inference_zone, initial_known, edge_weights, alpha, beta, epsilon, K, target_cell):
    """
    Implementation of Algorithm 4: Greedy Gumbel-Max Deletion.
    (Reverted to original logic, adapted for string-based metrics collection).
    """
    mask = set()
    if not inference_zone: return frozenset()
        
    # Allocate privacy budget per iteration
    epsilon_prime = epsilon / K if K > 0 else epsilon
    gumbel_beta_scale = 2 * alpha / epsilon_prime if epsilon_prime > 0 else float('inf')

    for _ in range(K): # Line 5: for k = 1 to K do
        candidate_cells = list(inference_zone - mask)
        if not candidate_cells: break

        scores = {}
        # Original logic implies these functions are available and use the full context
        leakage_current_mask = calculate_leakage_str(hyperedges, paths, mask, target_cell, initial_known, edge_weights)
        
        for cell_candidate in candidate_cells: # Line 6: for each c in I(c*) \ M do
            new_mask = mask | {cell_candidate}
            leakage_new_mask = calculate_leakage_str(hyperedges, paths, new_mask, target_cell, initial_known, edge_weights)
            marginal_utility_gain = alpha * (leakage_current_mask - leakage_new_mask) - beta
            gumbel_noise_for_cell = sample_gumbel(mu=0, beta=gumbel_beta_scale)
            scores[cell_candidate] = marginal_utility_gain + gumbel_noise_for_cell
            
        # Add a "stop" option (Lines 12-14 in paper)
        stop_gumbel_noise = sample_gumbel(mu=0, beta=gumbel_beta_scale)
        scores["__STOP__"] = 0 + stop_gumbel_noise # Marginal utility of stopping is 0

        best_candidate = max(scores, key=scores.get)

        if best_candidate == "__STOP__": # Check for termination (Line 13-14)
            break
        
        mask.add(best_candidate)

    return frozenset(mask)

# --- Helper Functions (Reverted and unified) ---

def measure_memory_overhead_gumbel(hyperedges, paths, inference_zone):
    memory = getsizeof(hyperedges) + getsizeof(paths) + getsizeof(inference_zone)
    for item in hyperedges: memory += getsizeof(item)
    for item in paths: memory += getsizeof(item)
    for item in inference_zone: memory += getsizeof(item)
    return memory
    
def get_path_inference_zone_str(paths, hyperedges, target_cell):
    all_cells_in_paths = set()
    for path in paths:
        for edge_idx in path:
            edge = hyperedges[edge_idx]
            all_cells_in_paths.update(edge)
    return all_cells_in_paths - {target_cell}


# --- Main Orchestrator Function ---

def gumbel_deletion_main(dataset: str, key: int, target_cell: str):
    init_start = time.time()
    try:
        if dataset == 'ncvoter':
            dataset_module_name = 'NCVoter'
        else:
            dataset_module_name = dataset.capitalize()
        dc_module_path = f'DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed'
        dc_module = __import__(dc_module_path, fromlist=['denial_constraints'])
        raw_dcs = dc_module.denial_constraints
    except ImportError:
        print(f"Error: Could not load constraints for {dataset}.")
        return None
    hyperedges = clean_raw_dcs(raw_dcs)
    init_time = time.time() - init_start

    model_start = time.time()
    all_attributes = set(attr for he in hyperedges for attr in he)
    attribute_counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_cells = {attr for attr, count in attribute_counts.items() if count > 1}
    initial_known = all_attributes - {target_cell} - intersecting_cells
    
    paths = find_inference_paths_str(hyperedges, target_cell, initial_known)
    num_paths = len(paths)
    inference_zone = get_path_inference_zone_str(paths, hyperedges, target_cell)
    
    K = len(inference_zone) if len(inference_zone) > 0 else 1 # Number of iterations for Gumbel
    alpha, beta, epsilon = 1.0, 0.5, 1.0
    edge_weights = {i: 0.8 for i in range(len(hyperedges))} # Uniform weights for now

    final_mask = greedy_gumbel_max_deletion(
        hyperedges, paths, inference_zone, initial_known, edge_weights, alpha, beta, epsilon, K, target_cell
    )
    mask_size = len(final_mask)
    
    memory_overhead = measure_memory_overhead_gumbel(hyperedges, paths, inference_zone)
    model_time = time.time() - model_start

    leakage = calculate_leakage_str(hyperedges, paths, final_mask, target_cell, initial_known, edge_weights)
    utility = -alpha * leakage - beta * mask_size
    
    del_start = time.time()
    if config:
        cells_to_delete = final_mask | {target_cell}
        conn = None
        cursor = None
        try:
            db_details = config.get_database_config(dataset)
            primary_table = f"{dataset}_copy_data"
            conn = mysql.connector.connect(
                host=db_details['host'], user=db_details['user'], password=db_details['password'],
                database=db_details['database'], ssl_disabled=db_details.get('ssl_disabled', True)
            )
            if conn.is_connected():
                cursor = conn.cursor()
                DELETION_QUERY = "UPDATE {table_name} SET `{column_name}` = NULL WHERE id = {key};"
                for cell_to_delete in cells_to_delete:
                    query = DELETION_QUERY.format(
                        table_name=primary_table, column_name=cell_to_delete, key=key
                    )
                    cursor.execute(query)
                conn.commit()
        except Error as e:
            print(f"Error during database update for gumbel: {e}")
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
    del_time = time.time() - del_start

    results = {
        'init_time': init_time, 'model_time': model_time, 'del_time': del_time,
        'leakage': leakage, 'utility': utility, 'mask_size': mask_size,
        'final_mask': final_mask, 'num_paths': num_paths, 'memory_overhead_bytes': memory_overhead,
        'num_instantiated_cells': len(inference_zone)
    }
    return results
