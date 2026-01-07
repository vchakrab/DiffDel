import sys
import os
import time
import csv
from collections import Counter

# Add project root to path to allow importing our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from build_attribute_templates import build_template_for_attribute, TEMPLATE_FILE
    from DifferentialDeletionAlgorithms.exponential_deletion import clean_raw_dcs
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    sys.exit(1)

# --- Benchmark Configuration ---
DATASETS_TO_BENCHMARK = [
    'tax'
    #'airport',
    #'hospital',
    #'ncvoter',
    #'adult',
    #'flights'
]

# Define specific iteration counts per dataset
DATASET_ITERATIONS = {

    'tax': 100,
}

OUTPUT_CSV_FILE = 'template_generation_benchmark.csv'

def find_most_constrained_attribute(dataset: str) -> str:
    """Loads DCs for a dataset and finds the most frequently occurring attribute."""
    print(f"  Determining most constrained attribute for '{dataset}'...")
    return "marital_status"
    try:
        # Dynamically import the correct DC module
        if dataset == 'ncvoter':
            dataset_module_name = 'NCVoter'
        else:
            dataset_module_name = dataset.capitalize()
            
        # Add specific handling for 'flights' as its module is directly under dc_configs
        if dataset == 'flights':
            dc_module_path = f'DCandDelset.dc_configs.topFlightsDCs_parsed'
        else:
            dc_module_path = f'DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed'

        dc_module = __import__(dc_module_path, fromlist=['denial_constraints'])
        raw_dcs = dc_module.denial_constraints
        
        hyperedges = clean_raw_dcs(raw_dcs)
        
        # Count all attribute occurrences
        attribute_counts = Counter(cell for he in hyperedges for cell in he)
        
        if not attribute_counts:
            return None
            
        # Find the most common attribute
        most_common_attr = attribute_counts.most_common(1)[0][0]
        print(f"  Found most constrained attribute: '{most_common_attr}'")
        return most_common_attr

    except ImportError:
        print(f"  Error: Could not load constraints for '{dataset}'. Skipping.")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred while analyzing {dataset}: {e}")
        return None

def main():
    """Main benchmark orchestration function."""
    print("--- Starting Template Generation Benchmark ---")
    all_results = []
    
    # CSV Headers
    csv_headers = [
        'dataset', 'attribute', 'iteration', 'generation_time', 
        'num_paths', 'inference_zone_size', 'num_masks'
    ]

    for dataset in DATASETS_TO_BENCHMARK:
        target_attribute = find_most_constrained_attribute(dataset)
        
        if not target_attribute:
            continue

        num_iterations_for_dataset = DATASET_ITERATIONS.get(dataset, 100) # Default to 100

        print(f"\n--- Benchmarking Dataset: '{dataset}' (Attribute: '{target_attribute}', Iterations: {num_iterations_for_dataset}) ---")
        
        for i in range(1, num_iterations_for_dataset + 1):
            print(f"  Running iteration {i}/{num_iterations_for_dataset}...", end=' ', flush=True)
            
            start_time = time.time()
            template = build_template_for_attribute(dataset, target_attribute)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"Done in {duration:.4f}s.")
            
            if template:
                all_results.append({
                    'dataset': dataset,
                    'attribute': target_attribute,
                    'iteration': i,
                    'generation_time': duration,
                    'num_paths': len(template.get('paths', [])),
                    'inference_zone_size': len(template.get('inference_zone', set())),
                    'num_masks': len(template.get('mask_probabilities', []))
                })
            else:
                print(f"  WARNING: Template generation failed for iteration {i}. Skipping record.")

            # ONLY delete the file if it's NOT the last iteration for this dataset
            if i < num_iterations_for_dataset and os.path.exists(TEMPLATE_FILE):
                os.remove(TEMPLATE_FILE)
                
    # After all benchmarks, ensure the last generated template files exist
    # This is implicitly handled by not deleting the last iteration's file.

    # Write results to CSV
    if all_results:
        print(f"\n--- Writing {len(all_results)} results to {OUTPUT_CSV_FILE} ---")
        try:
            file_exists = os.path.exists(OUTPUT_CSV_FILE)
            with open(OUTPUT_CSV_FILE, 'a' if file_exists else 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                if not file_exists: # Only write header if file is new
                    writer.writeheader()
                writer.writerows(all_results)
            print("Successfully saved benchmark results.")
        except IOError as e:
            print(f"Error writing to CSV file: {e}")
    else:
        print("No benchmark results were collected.")

if __name__ == '__main__':
    main()
