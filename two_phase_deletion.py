#!/usr/bin/env python3

import pickle
import os
import time
from collections import Counter
import numpy as np

from exponential_deletion import (
    clean_raw_dcs,
    find_inference_paths_str,
    get_path_inference_zone_str,
    compute_possible_mask_set_str,
    filter_active_paths_str,
    calculate_leakage_str,
    compute_utility_str
)

# -----------------------------
# Helper: readable serialization
# -----------------------------

def template_to_string(T_attr, attribute_name=None):
    lines = []

    title = f"TEMPLATE FOR ATTRIBUTE: {attribute_name}" if attribute_name else "TEMPLATE"
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)
    lines.append("")

    # ---- I_intra ----
    lines.append("[I_INTRA]")
    for a in sorted(T_attr["I_intra"]):
        lines.append(f"  {a}")
    lines.append("\n" + "-" * 60 + "\n")

    # ---- Π_intra ----
    lines.append("[Π_INTRA]  (Inference Paths)")
    for i, p in enumerate(T_attr["Π_intra"], 1):
        lines.append(f"  Path {i}: {p}")
    lines.append("\n" + "-" * 60 + "\n")

    # ---- R_intra ----
    lines.append("[R_INTRA]  (Candidate Masks)")
    for i, m in enumerate(T_attr["R_intra"], 1):
        mask_str = ", ".join(sorted(m)) if m else ""
        lines.append(f"  Mask {i}: {{{mask_str}}}")
    lines.append("\n" + "-" * 60 + "\n")

    # This is now deferred to the online phase
    pass

    # ---- Leakage / Utility / Probability ----
    lines.append("[MASK METRICS]")
    for mask in sorted(T_attr["R_intra"], key=lambda x: (len(x), sorted(x))):
        mask_key = frozenset(mask)
        mask_str = ", ".join(sorted(mask)) if mask else ""
        lines.append(f"Mask: {{{mask_str}}}")
        lines.append(f"  Leakage: {T_attr['Leakage'][mask_key]:.4f}")
        lines.append(f"  Utility: {T_attr['Utility'][mask_key]:.4f}")
        lines.append(f"  Probability: {T_attr['Probability'][mask_key]:.6f}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# -----------------------------
# Main template builder
# -----------------------------
def build_template(dataset, attribute, save_dir="templates"):
    start_time = time.time()

    # ---- Load DCs ----
    try:
        dataset_module_name = "NCVoter" if dataset == "ncvoter" else dataset.capitalize()
        dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        raw_dcs = dc_module.denial_constraints
        print(f"[INFO] Loaded {len(raw_dcs)} denial constraints")
    except ImportError as e:
        print(f"[ERROR] Failed to load DCs: {e}")
        return None

    # ---- Clean DCs into hyperedges ----
    hyperedges = clean_raw_dcs(raw_dcs)
    print(f"[INFO] Cleaned into {len(hyperedges)} hyperedges")

    # ---- Compute intra-zone ----
    all_attributes = set(attr for he in hyperedges for attr in he)
    counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_cells = {a for a, c in counts.items() if c > 1}
    initial_known = all_attributes - {attribute} - intersecting_cells

    # ---- Paths & masks ----
    paths = find_inference_paths_str(hyperedges, attribute, initial_known)
    inference_zone = get_path_inference_zone_str(paths, hyperedges, attribute)
    candidate_masks = compute_possible_mask_set_str(inference_zone)

    # ---- Blocked / Unblocked calculation is deferred to online phase ----

    # ---- Compute Leakage / Utility / Probability ----
    alpha, beta, epsilon = 1.0, 0.5, 1.0
    Leakage, Utility, Probability = {}, {}, {}
    utilities = []

    for mask in candidate_masks:
        mask_frozen = frozenset(mask)
        leakage = calculate_leakage_str(hyperedges, paths, mask, attribute, initial_known)
        utility = compute_utility_str(mask, attribute, paths, hyperedges, initial_known,
                                      edge_weights=None, alpha=alpha, beta=beta)
        Leakage[mask_frozen] = leakage
        Utility[mask_frozen] = utility
        utilities.append(utility)

    # Exponential mechanism probabilities
    scores = epsilon * np.array(utilities) / (2 * alpha)
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / np.sum(exp_scores)
    for mask, p in zip(candidate_masks, probs):
        Probability[frozenset(mask)] = p

    # ---- Build template ----
    T_attr = {
        "hyperedges": hyperedges,
        "I_intra": initial_known,
        "Π_intra": paths,
        "R_intra": candidate_masks,
        "Σ_cross": [],
        "Leakage": Leakage,
        "Utility": Utility,
        "Probability": Probability
    }

    # ---- Save files ----
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f"{dataset}_{attribute}.pkl")
    txt_path = os.path.join(save_dir, f"{dataset}_{attribute}.txt")
    with open(pkl_path, "wb") as f:
        pickle.dump(T_attr, f)
    with open(txt_path, "w") as f:
        f.write(template_to_string(T_attr))

    elapsed = time.time() - start_time
    print(f"[DONE] Template built for ({dataset}, {attribute}) in {elapsed:.3f}s")
    print(f"       Pickle: {pkl_path}")
    print(f"       Text:   {txt_path}")

    return T_attr

import sys

def online_mask_selection(dataset: str, attribute: str, seed=None):
    """
    Selects a mask for a specific attribute from a precomputed template
    and returns timing info for init and model phases.
    """
    timings = {"init_time": 0.0, "model_time": 0.0}

    t0 = time.time()
    if seed is not None:
        np.random.seed(seed)

    # ---- Load precomputed template ----
    pkl_path = f"templates/{dataset}_{attribute}.pkl"
    try:
        with open(pkl_path, "rb") as f:
            T_attr = pickle.load(f)
    except FileNotFoundError:
        raise ValueError(f"Template not found at {pkl_path}. Run build_template() first.")
    t1 = time.time()
    timings["init_time"] = t1 - t0

    # ---- Step 3: Sample according to probabilities ----
    t2 = time.time()
    candidate_masks = T_attr["R_intra"]
    probs_dict = T_attr["Probability"]
    mask_list = candidate_masks
    probabilities = [probs_dict[frozenset(m)] for m in mask_list]
    selected_idx = np.random.choice(len(mask_list), p=probabilities)
    selected_mask = mask_list[selected_idx]
    t3 = time.time()
    timings["model_time"] = t3 - t2

    return set(selected_mask), timings, T_attr  # return template for memory calc

def memory_overhead(T_attr):
    """
    Compute memory footprint of online deletion structures only.
    Excludes the offline template building memory.
    """
    # Only include candidate masks, probabilities, and paths needed for online deletion
    overhead = sys.getsizeof(T_attr["R_intra"])
    overhead += sys.getsizeof(T_attr["Probability"])
    overhead += sys.getsizeof(T_attr["Π_intra"])
    overhead += sum(sys.getsizeof(frozenset(m)) for m in T_attr["R_intra"])
    return overhead

def offline_precomputation(dataset: str, attributes: list, force_rebuild=False):
    """
    Builds templates for a given dataset and list of attributes if they don't exist.
    """
    print(f"--- Running Offline Precomputation for {dataset} ---")
    all_templates = {}
    for attr in attributes:
        template_path = f"templates/{dataset}_{attr}.pkl"
        if not os.path.exists(template_path) or force_rebuild:
            print(f"Template not found for '{attr}', building...")
            build_template(dataset, attr)
        
        # Load the template
        with open(template_path, "rb") as f:
            all_templates[attr] = pickle.load(f)
    print("--- Offline Precomputation Finished ---")
    return all_templates


def two_phase_deletion_main(dataset: str, key: int, target_cell: str, templates: dict):
    """
    Main orchestrator for the 2-phase deletion mechanism.
    """
    import mysql.connector
    from mysql.connector import Error
    import config
    
    # --- Online Phase ---
    t0 = time.time()
    
    # Retrieve the specific template for the target attribute
    if target_cell not in templates:
        raise ValueError(f"No precomputed template for attribute '{target_cell}' in dataset '{dataset}'")
    T_attr = templates[target_cell]
    
    # Timings for init are effectively zero in the online phase as templates are pre-loaded
    init_time = 0.0

    # Model time is for sampling the mask
    model_start = time.time()
    
    candidate_masks = T_attr["R_intra"]
    probs_dict = T_attr["Probability"]
    mask_list = list(candidate_masks)
    probabilities = [probs_dict[frozenset(m)] for m in mask_list]
    
    selected_idx = np.random.choice(len(mask_list), p=probabilities)
    final_mask = mask_list[selected_idx]
    
    model_time = time.time() - model_start

    # --- Metrics Calculation ---
    mask_frozen = frozenset(final_mask)
    leakage = T_attr['Leakage'][mask_frozen]
    utility = T_attr['Utility'][mask_frozen]
    mask_size = len(final_mask)
    
    # Online calculation of paths_blocked
    paths = T_attr['Π_intra']
    hyperedges = T_attr['hyperedges']
    initial_known = T_attr['I_intra']
    num_paths = len(paths)
    
    active_paths = filter_active_paths_str(hyperedges, paths, final_mask, initial_known)
    paths_blocked = num_paths - len(active_paths)
    
    # --- Database Update Phase ---
    del_start = time.time()
    cells_to_delete = set(final_mask) | {target_cell}
    conn, cursor = None, None
    try:
        db_details = config.get_database_config(dataset)
        table_name = f"{dataset}_copy_data"
        conn = mysql.connector.connect(**db_details)
        if conn.is_connected():
            cursor = conn.cursor()
            for col in cells_to_delete:
                query = f"UPDATE {table_name} SET `{col}` = NULL WHERE id = {key};"
                cursor.execute(query)
            conn.commit()
    except Error as e:
        print(f"Database update failed: {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
    del_time = time.time() - del_start

    # --- Compile and Return Results ---
    mem_overhead = memory_overhead(T_attr)
    num_instantiated = len(T_attr['I_intra']) + len(T_attr['Π_intra']) # Approximation

    results = {
        'init_time': init_time,
        'model_time': model_time,
        'del_time': del_time,
        'leakage': leakage,
        'utility': utility,
        'mask_size': mask_size,
        'final_mask': final_mask,
        'num_paths': num_paths,
        'paths_blocked': paths_blocked,
        'memory_overhead_bytes': mem_overhead,
        'num_instantiated_cells': num_instantiated
    }
    return results




# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    build_template("tax", "marital_status")
