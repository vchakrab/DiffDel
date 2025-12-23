import sys
import os
import time
import mysql.connector
from mysql.connector import Error

# Add project root to path to allow importing our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    from build_attribute_templates import build_template_for_attribute, TEMPLATE_FILE
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    sys.exit(1)

import csv

# --- Benchmark Configuration ---
DATASETS_TO_BENCHMARK = {
    'tax': 'marital_status',
}
NUM_ITERATIONS = 5

def save_results_to_csv(results):
    """Saves a list of benchmark results to a CSV file."""
    output_file = 'template_generation_benchmark.csv'
    
    if not results:
        return
        
    # Check if file exists to write header
    write_header = not os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['dataset', 'attribute', 'iteration', 'generation_time', 'num_paths', 'inference_zone_size', 'num_masks']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()
        
        for r in results:
            writer.writerow({
                'dataset': r['dataset'],
                'attribute': r['attribute'],
                'iteration': r['iteration'],
                'generation_time': r['time'],
                'num_paths': r['num_paths'],
                'inference_zone_size': r['inference_zone_size'],
                'num_masks': r['num_masks']
            })
            
    print(f"  - Saved {len(results)} record(s) to {output_file}.")


def main():
    """Main benchmark orchestration function."""
    print("--- Starting Template Generation Benchmark ---")

    for dataset, attribute in DATASETS_TO_BENCHMARK.items():
        print(f"\n--- Benchmarking Dataset: '{dataset}' (Attribute: '{attribute}') ---")
        
        for i in range(1, NUM_ITERATIONS + 1):
            print(f"  Running iteration {i}/{NUM_ITERATIONS}...", end=' ', flush=True)
            
            start_time = time.time()
            
            template = build_template_for_attribute(dataset, attribute)
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"Done in {duration:.4f}s.")
            
            if template:
                num_paths = len(template['paths'])
                inference_zone_size = len(template['inference_zone'])
                num_masks = len(template['mask_probabilities'])
                
                result = {
                    'dataset': dataset,
                    'attribute': attribute,
                    'iteration': i,
                    'time': duration,
                    'num_paths': num_paths,
                    'inference_zone_size': inference_zone_size,
                    'num_masks': num_masks
                }
                save_results_to_csv([result])
            else:
                print(f"  WARNING: Template generation failed for iteration {i}. Not saving record.")

            if os.path.exists(TEMPLATE_FILE):
                os.remove(TEMPLATE_FILE)

if __name__ == '__main__':
    main()
