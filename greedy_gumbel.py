import numpy as np
import random
import time
from typing import Set, Dict, List, Tuple, Iterable, Optional, Optional
from itertools import chain, combinations
from sys import getsizeof
from collections import Counter, defaultdict
# NOTE: These imports are required for database interaction.
import mysql.connector
from mysql.connector import Error


# The 'config' module must be present in your environment
# and contain the get_database_config method.
import config


# =================================================================
# 1. CORE UTILITIES
# =================================================================

def powerset(iterable: Iterable) -> chain:
    """Computes the powerset of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def gumbel_noise(scale: float) -> float:
    """
    Samples Gumbel noise g ~ Gumbel(0, scale) using the formula:
    g = -scale * log(-log(u)) where u ~ Uniform(0, 1)
    """
    u = random.random()
    u = max(1e-10, min(1.0 - 1e-10, u))
    return -scale * np.log(-np.log(u))


def clean_raw_dcs(raw_dcs: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
    """
    Cleans raw denial constraints into simple hyperedges (tuples of attribute strings).
    """
    cleaned_hyperedges = []
    for dc in raw_dcs:
        attributes = set()
        for pred in dc:
            for item in [pred[0], pred[2]]:
                if '.' in item:
                    attr_name = item.split('.')[-1]
                    attributes.add(attr_name)
        for item in dc[0]:
            if not isinstance(item, str) or (
                    '.' not in item and item not in ('!=', '==', '<', '>', '<=', '>=')):
                attributes.add(str(item))

        if attributes:
            cleaned_hyperedges.append(tuple(sorted(list(attributes))))
    return cleaned_hyperedges


# =================================================================
# 2. STR HYPEREDGE METHODS (LOGIC RETAINED AS PROVIDED)
# =================================================================

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


def compute_product_leakage_str(active_paths: List[List[int]], hyperedges: List[Tuple[str, ...]],
                                edge_weights: Dict[int, float]) -> float:
    if edge_weights is None: edge_weights = {i: 1.0 for i in range(len(hyperedges))}
    if len(active_paths) == 0: return 0.0
    print(len(active_paths))
    product = 1.0
    for path in active_paths:
        path_weight = 1.0
        for edge_idx in path: path_weight *= edge_weights.get(edge_idx, 1.0)
        product *= (1 - path_weight)
    leakage = 1 - product
    return leakage

def calculate_inferential_leakage(active_paths: List[List[int]],
                                  hyperedges: List[Tuple[str, ...]],
                                  edge_weights: Optional[List[float] | Dict[int, float]]) -> float:
    """
    Implements the Paper's logic (Definition 5.1).
    Groups paths into Channels (Eq 3) then aggregates Channels (Eq 4).
    """
    if not active_paths:
        return 0.0

    # Ensure edge_weights is a dictionary for .get() access
    if isinstance(edge_weights, list):
        edge_weights_dict = {i: w for i, w in enumerate(edge_weights)}
    elif edge_weights is None:
        # Default to 1.0 if no weights are provided
        edge_weights_dict = {i: 1.0 for i in range(len(hyperedges))}
    else:
        edge_weights_dict = edge_weights

    # 1. Group paths by final hyperedge (Inference Channel e in E*)
    inference_channels = defaultdict(list)
    for path in active_paths:
        final_edge_idx = path[-1]
        inference_channels[final_edge_idx].append(path)

    # 2. Equation 3: Effective weight (w*) for each channel
    channel_effective_weights = []
    for edge_idx, paths in inference_channels.items():
        path_success_probs = []
        for path in paths:
            # w(pi): product of edge weights in path
            w_pi = np.prod([edge_weights_dict.get(e_idx, 1.0) for e_idx in path])
            path_success_probs.append(w_pi)

        # w*_e = 1 - PRODUCT(1 - w_pi)
        w_star_e = 1 - np.prod([1 - p for p in path_success_probs])
        channel_effective_weights.append(w_star_e)

    # 3. Equation 4: Total Inferential Leakage (L)
    # L = 1 - PRODUCT(1 - w*_e)
    leakage = 1 - np.prod([1 - w_star for w_star in channel_effective_weights])

    return float(leakage)
def calculate_leakage_str(hyperedges: List[Tuple[str, ...]], paths: List[List[int]], mask: Set[str],
                          target_cell: str, initial_known: Set[str],
                          edge_weights: Dict[int, float] = None) -> float:
    active_paths = filter_active_paths_str(hyperedges, paths, mask, initial_known)
    leakage = calculate_inferential_leakage(active_paths, hyperedges, edge_weights)
    return leakage


def get_path_inference_zone_str(paths: List[List[int]], hyperedges: List[Tuple[str, ...]],
                                target_cell: str) -> Set[str]:
    all_cells_in_paths = set()
    for path in paths:
        for edge_idx in path:
            edge = hyperedges[edge_idx]
            all_cells_in_paths.update(edge)
    path_zone = all_cells_in_paths - {target_cell}
    return path_zone


def compute_utility_str(mask: Set[str], target_cell: str, paths: List[List[int]],
                        hyperedges: List[Tuple[str, ...]], initial_known: Set[str],
                        edge_weights: Dict[int, float], alpha: float, beta: float) -> float:
    leakage = calculate_leakage_str(hyperedges, paths, mask, target_cell, initial_known,
                                    edge_weights)
    utility = -alpha * leakage - beta * len(mask)
    return utility


def compute_possible_mask_set_str(inference_zone_str: Set[str]) -> List[Set[str]]:
    all_subsets = [set(subset) for subset in powerset(inference_zone_str)]
    return all_subsets


def exponential_mechanism_sample_str(candidates: List[Set[str]], target_cell: str,
                                     paths: List[List[int]], hyperedges: List[Tuple[str, ...]],
                                     initial_known: Set[str], edge_weights: Dict[int, float],
                                     alpha: float, beta: float, epsilon: float) -> Set[str]:
    if not candidates: return set()
    utilities = []
    for mask in candidates:
        u = compute_utility_str(mask, target_cell, paths, hyperedges, initial_known, edge_weights,
                                alpha, beta)
        utilities.append(u)
    utilities = np.array(utilities)
    scores = epsilon * utilities / (2 * alpha)
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    probabilities = exp_scores / np.sum(exp_scores)
    selected_idx = np.random.choice(len(candidates), p = probabilities)
    return candidates[selected_idx]


def measure_memory_overhead_str(hyperedges, paths, candidate_masks):
    memory = getsizeof(hyperedges)
    for he in hyperedges:
        memory += getsizeof(he)
        for attr in he: memory += getsizeof(attr)
    memory += getsizeof(paths)
    for path in paths: memory += getsizeof(path)
    memory += getsizeof(candidate_masks)
    for mask in candidate_masks:
        memory += getsizeof(mask)
        for attr in mask: memory += getsizeof(attr)
    return memory


# =================================================================
# 3. ALGORITHM 4: GREEDY GUMBEL-MAX DELETION
# =================================================================

def calculate_marginal_utility_gain(
        c: str, M_curr: Set[str], paths: List[List[int]], hyperedges: List[Tuple[str, ...]],
        target_cell: str, initial_known: Set[str], edge_weights: Dict[int, float],
        alpha: float, beta: float
) -> Tuple[float, float, float]:
    """Calculates the marginal utility gain Delta_u(c) for adding cell c to mask M_curr."""
    L_curr = calculate_leakage_str(hyperedges, paths, M_curr, target_cell, initial_known,
                                   edge_weights)
    M_new = M_curr.union({c})
    L_new = calculate_leakage_str(hyperedges, paths, M_new, target_cell, initial_known,
                                  edge_weights)
    delta_u_c = alpha * (L_curr - L_new) - beta
    return delta_u_c, L_curr, L_new


def greedy_gumbel_max_deletion_str(
        hyperedges: List[Tuple[str, ...]], target_cell: str, initial_known: Set[str],
        edge_weights: Dict[int, float], alpha: float, beta: float, epsilon: float,
        K: int, tau: float = 0.0, paths: Optional[List[List[int]]] = None
) -> Tuple[Set[str], float]:
    """ Implements Algorithm 4: Greedy Gumbel-Max Deletion Mechanism (String-based). """
    start_time = time.time()

    if paths is None:
        paths_all: List[List[int]] = find_inference_paths_str(hyperedges, target_cell,
                                                              initial_known)
    else:
        paths_all = paths

    # Filter paths based on weight threshold tau
    Pi: List[List[int]] = []
    for path in paths_all:
        path_weight = 1.0
        for edge_idx in path:
            path_weight *= edge_weights.get(edge_idx, 1.0)
        if path_weight >= tau:
            Pi.append(path)

    I_c_star: Set[str] = get_path_inference_zone_str(Pi, hyperedges, target_cell)

    M: Set[str] = set()
    if K <= 0 or epsilon <= 0: return M, (time.time() - start_time)
    epsilon_prime: float = epsilon / K

    b_cell: float = 4.0 * alpha / epsilon_prime
    b_stop: float = 2.0 * alpha / epsilon_prime

    for k in range(1, K + 1):
        candidate_cells: Set[str] = I_c_star - M
        if not candidate_cells: break

        scores: Dict[str, float] = {}
        for c in candidate_cells:
            delta_u_c, _, _ = calculate_marginal_utility_gain(
                c, M, Pi, hyperedges, target_cell, initial_known, edge_weights, alpha, beta
            )
            g_c: float = gumbel_noise(b_cell)
            scores[c] = delta_u_c + g_c

        s_stop: float = gumbel_noise(b_stop)
        max_s_c: float = max(scores.values())
        best_c: str = max(scores, key = scores.get)

        if s_stop > max_s_c:
            break
        else:
            M.add(best_c)

    final_leakage = calculate_leakage_str(hyperedges, Pi, M, target_cell, initial_known,
                                          edge_weights)
    end_time = time.time()
    return M, (end_time - start_time)


# =================================================================
# 4. MAIN ORCHESTRATOR
# =================================================================
def gumbel_deletion_main(dataset: str, key: int, target_cell: str, method: str = 'gumbel', epsilon: float = 1.0, alpha: float = 1, beta: float = 0.5, K : int = 10, edge_weights = None):
    """
    Main orchestrator for the string-based deletion mechanism (Exp or Gumbel).
    """
    print("\n" + "=" * 70)
    print(
        f"Running Orchestrator for Dataset: '{dataset}', Key: {key}, Target: '{target_cell}' (Method: {method.upper()})")
    print("=" * 70)

    # --- Initialization Phase (Loading Actual DCs) ---
    init_start = time.time()
    try:
        if dataset == 'ncvoter':
            dataset_module_name = 'NCVoter'
        else:
            dataset_module_name = dataset.capitalize()

        dc_module_path = f'DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed'
        dc_module = __import__(dc_module_path, fromlist = ['denial_constraints'])
        raw_dcs = dc_module.denial_constraints
        print(f"Successfully loaded {len(raw_dcs)} raw denial constraints for '{dataset}'.")
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not find or load denial constraints for dataset '{dataset}'.")
        print(f"Attempted to load module: {dc_module_path}")
        print(f"Python Error: {e}")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to process raw denial constraints: {e}")
        return None

    hyperedges = clean_raw_dcs(raw_dcs)
    print(f"Cleaned DCs into {len(hyperedges)} simple hyperedges.")
    init_time = time.time() - init_start

    # --- Modeling Phase ---
    model_start = time.time()
    all_attributes = set(attr for he in hyperedges for attr in he)
    attribute_counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_cells = {attr for attr, count in attribute_counts.items() if count > 1}

    # Core logic for Initial Known Set
    initial_known = all_attributes - {target_cell} - intersecting_cells

    paths = find_inference_paths_str(hyperedges, target_cell, initial_known)
    num_paths = len(paths)
    inference_zone = get_path_inference_zone_str(paths, hyperedges, target_cell)

    # --- Algorithm Parameters ---
    if edge_weights is None:
        edge_weights = {i: 0.8 for i in range(len(hyperedges))}
    memory_overhead = 0.0

    # --- Execution/Sampling ---
    if method == 'exp':
        candidate_masks = compute_possible_mask_set_str(inference_zone)
        memory_overhead = measure_memory_overhead_str(hyperedges, paths, candidate_masks)

        final_mask = exponential_mechanism_sample_str(
            candidate_masks, target_cell, paths, hyperedges, initial_known,
            edge_weights, alpha, beta, epsilon
        )
        model_time = time.time() - model_start

    elif method == 'gumbel':
        final_mask, gumbel_duration = greedy_gumbel_max_deletion_str(
            hyperedges = hyperedges, target_cell = target_cell, initial_known = initial_known,
            edge_weights = edge_weights, alpha = alpha, beta = beta, epsilon = epsilon,
            K = K, paths = paths
        )
        model_time = gumbel_duration
        candidate_masks = []
        memory_overhead = measure_memory_overhead_str(hyperedges, paths, candidate_masks)

    else:
        print(f"Error: Unknown method '{method}'.")
        return None

    mask_size = len(final_mask)

    # --- Metrics Calculation ---
    leakage = calculate_leakage_str(hyperedges, paths, final_mask, target_cell, initial_known,
                                    edge_weights)
    utility = compute_utility_str(final_mask, target_cell, paths, hyperedges, initial_known,
                                  edge_weights, alpha, beta)

    # --- Database Update Phase (UNCOMMENTED) ---
    del_start = time.time()

    cells_to_delete = final_mask | {target_cell}
    conn = None
    cursor = None

    try:
        # NOTE: 'config' must be available in your environment!
        # Assuming config.get_database_config(dataset) returns the necessary details
        db_details = config.get_database_config(dataset)
        primary_table = f"{dataset}_copy_data"
        conn = mysql.connector.connect(
            host = db_details['host'], user = db_details['user'],
            password = db_details['password'],
            database = db_details['database'],
            ssl_disabled = db_details.get('ssl_disabled', True)
        )
        if conn and conn.is_connected():
            cursor = conn.cursor()
            DELETION_QUERY = "UPDATE {table_name} SET `{column_name}` = NULL WHERE id = {key};"
            for cell_to_delete in cells_to_delete:
                query = DELETION_QUERY.format(
                    table_name = primary_table, column_name = cell_to_delete, key = key
                )
                cursor.execute(query)
            conn.commit()
            print(f"Successfully deleted {len(cells_to_delete)} cells for key {key}.")
        else:
            print("Warning: Could not connect to the database. Deletion skipped.")

    except NameError:
        print("Warning: 'config' module not found. Skipping database deletion.")
    except ImportError:
        print("Warning: 'mysql.connector' not installed. Skipping database deletion.")
    except Error as e:
        print(f"Error during database update: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during database update: {e}")
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
    }

    print("\n" + "=" * 70)
    print(f"Main Orchestrator Finished. Results for {method.upper()}:")
    for k, val in results.items():
        if isinstance(val, float):
            print(f"  - {k}: {val:.4f}")
        else:
            print(f"  - {k}: {val}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    # Example run using the Greedy Gumbel-Max Deletion method
    # NOTE: Execution requires 'mysql.connector' and a 'config' module setup.
    results = gumbel_deletion_main(
        dataset = 'adult',
        key = 2,
        target_cell = 'education',
        method = 'gumbel'  # Change to 'exp' to run the exponential mechanism
    )