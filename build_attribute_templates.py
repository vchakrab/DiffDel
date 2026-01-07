import csv

from DifferentialDeletionAlgorithms.exponential_deletion import *


def build_template(dataset, attribute):
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

    # --- Modeling Phase ---
    all_attributes = set(attr for he in hyperedges for attr in he)
    attribute_counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_cells = {attr for attr, count in attribute_counts.items() if count > 1}
    initial_known = all_attributes - {attribute} - intersecting_cells

    paths = find_inference_paths_str(hyperedges, attribute, initial_known)
    num_paths = len(paths)


    inference_zone = get_path_inference_zone_str(paths, hyperedges, attribute)
    candidate_masks = compute_possible_mask_set_str(inference_zone)
    edge_weights = {i: 0.8 for i in range(len(hyperedges))}
    alpha = 1.0
    beta = 0.5
    epsilon = 1.0
    if not candidate_masks:
        return set()

    # 1. Compute utilities for all candidates
    utilities = []
    for mask in candidate_masks:
        u = compute_utility_str(
            mask,
            attribute,
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
    end_time = time.time() - init_start
    return end_time

def collect_data_put_in_csv():
    datasets_and_attributes = {'flights': 'FlightNum'}
    for dataset,attr in datasets_and_attributes.items():
        for i in range(100):
            time = build_template(dataset, attr)
            with open('experiment_2_data.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([attr + ',' + str(time)])


collect_data_put_in_csv()
"""

import pickle
import os

def build_template(dataset, attribute, save_dir="templates"):
    init_start = time.time()
    try:
        if dataset == 'ncvoter':
            dataset_module_name = 'NCVoter'
        else:
            dataset_module_name = dataset.capitalize()

        dc_module_path = f'DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed'
        dc_module = __import__(dc_module_path, fromlist=['denial_constraints'])
        raw_dcs = dc_module.denial_constraints
        print(f"Successfully loaded {len(raw_dcs)} raw denial constraints for '{dataset}'.")
    except ImportError:
        print(f"Error: Could not find or load denial constraints for dataset '{dataset}'.")
        print(f"Attempted to load module: {dc_module_path}")
        return None

    hyperedges = clean_raw_dcs(raw_dcs)
    print("Cleaned DCs into simple hyperedges.")

    # --- Modeling Phase ---
    all_attributes = set(attr for he in hyperedges for attr in he)
    attribute_counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_cells = {attr for attr, count in attribute_counts.items() if count > 1}
    initial_known = all_attributes - {attribute} - intersecting_cells

    paths = find_inference_paths_str(hyperedges, attribute, initial_known)
    inference_zone = get_path_inference_zone_str(paths, hyperedges, attribute)
    candidate_masks = compute_possible_mask_set_str(inference_zone)

    # --- Compute blocked/unblocked paths per mask ---
    Blocked = {}
    Unblocked = {}
    for mask in candidate_masks:
        active_paths = filter_active_paths_str(hyperedges, paths, mask, initial_known)
        blocked_paths = [p for p in paths if p not in active_paths]

        Blocked[frozenset(mask)] = blocked_paths
        Unblocked[frozenset(mask)] = active_paths

    # --- Build template dictionary ---
    T_attr = {
        'I_intra': initial_known,
        'Π_intra': paths,
        'R_intra': candidate_masks,
        'Blocked': Blocked,
        'Unblocked': Unblocked,
        'Σ_cross': []  # cross-tuple constraints, optional
    }

    # --- Save template to file ---
    os.makedirs(save_dir, exist_ok=True)
    template_filename = os.path.join(save_dir, f"{dataset}_{attribute}.pkl")
    with open(template_filename, "wb") as f:
        pickle.dump(T_attr, f)
    print(f"Template saved to: {template_filename}")

    end_time = time.time() - init_start
    print(f"Template built for attribute '{attribute}' in {end_time:.4f} seconds.")
    return T_attr
"""