import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DifferentialDeletionAlgorithms.exponential_deletion import find_inference_paths_str, clean_raw_dcs

def find_heavy_paths():
    """
    This function replicates the path-finding process that results in a large
    number of paths for the tax dataset and marital_status attribute.
    """
    dataset = 'tax'
    target_attribute = 'marital_status'

    print(f"--- Finding paths for: dataset='{dataset}', attribute='{target_attribute}' ---")

    # 1. Load the same constraints used in the exponential deletion process.
    try:
        from DCandDelset.dc_configs import topTaxDCs_parsed
        raw_dcs = topTaxDCs_parsed.denial_constraints
        print(f"Successfully loaded {len(raw_dcs)} denial constraints for '{dataset}'.")
    except ImportError:
        print(f"Error: Could not load constraints for '{dataset}'.")
        print("Please ensure 'DCandDelset/dc_configs/topTaxDCs_parsed.py' exists.")
        return

    # 2. Process constraints into hyperedges
    hyperedges = clean_raw_dcs(raw_dcs)
    print(f"Processed into {len(hyperedges)} hyperedges.")

    all_paths = []
    seen_paths = set()
    
    print("\nStarting iterative path finding...")
    # Iterate through each hyperedge to create a set of precursors
    for i, precursor_edge in enumerate(hyperedges):
        initial_known = set(precursor_edge) - {target_attribute}
        
        # Find paths starting with this precursor set
        paths = find_inference_paths_str(hyperedges, target_attribute, initial_known)
        
        for path in paths:
            path_tuple = tuple(sorted(path))
            if path_tuple not in seen_paths:
                all_paths.append(path)
                seen_paths.add(path_tuple)
        
        if (i + 1) % 5 == 0:
            print(f"  ... processed {i + 1}/{len(hyperedges)} hyperedges, found {len(all_paths)} unique paths so far.")

    print("\n--- Results ---")
    print(f"Found {len(all_paths)} total unique inference paths for '{target_attribute}'.")
    
    if all_paths:
        print("\nExample paths (first 5 of final set):")
        for p in all_paths[:5]:
            print(p)

if __name__ == '__main__':
    find_heavy_paths()
