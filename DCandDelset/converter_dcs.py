import json
import os

def convert_constraints(input_file, output_file_py):
    # Map Metanome operators to logic operators
    op_map = {
        "EQUAL": "==",
        "UNEQUAL": "!=",  # Changed from <> to != for Python consistency
        "LESS": "<",
        "GREATER": ">",
        "LESS_EQUAL": "<=",
        "GREATER_EQUAL": ">="
    }

    all_denial_constraints = []

    try:
        with open(input_file, 'r') as f_in:
            for line in f_in:
                if not line.strip():
                    continue

                # Parse the JSON line
                data = json.loads(line)
                predicates = data.get("predicates", [])

                current_dc_predicates = []
                for pred in predicates:
                    # Extract column name and remove type info (e.g., 'FlightNum int' -> 'FlightNum')
                    col1_full = pred["column1"]["columnIdentifier"]
                    col1_name = col1_full.split(' ')[0]
                    col2_full = pred["column2"]["columnIdentifier"]
                    col2_name = col2_full.split(' ')[0]

                    # Determine tuple identifiers (t1, t2)
                    idx1 = f"t{pred['index1'] + 1}"
                    idx2 = f"t{pred['index2'] + 1}"
                    
                    operator = op_map.get(pred["op"], pred["op"])

                    # Create the predicate tuple: ('t1.Column', 'OPERATOR', 't2.Column')
                    current_dc_predicates.append((f"{idx1}.{col1_name}", operator, f"{idx2}.{col2_name}"))
                
                if current_dc_predicates:
                    all_denial_constraints.append(current_dc_predicates)

        # Write the Python module output file
        with open(output_file_py, 'w') as f_out:
            f_out.write(f"# Parsed denial constraints for {os.path.basename(input_file)}\n")
            f_out.write("# Generated automatically from raw constraints\n\n")
            f_out.write("denial_constraints = [\n")
            for dc in all_denial_constraints:
                f_out.write(f"    {dc},\n")
            f_out.write("]\n")

        print(f"Success! Converted constraints saved to {output_file_py}")

    except FileNotFoundError:
        print("Error: Input file not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Ensure each line is a valid JSON object.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage (update as needed for specific dataset conversion)
# This will now create a .py file that can be imported
# Ensure the output directory exists
os.makedirs('dc_configs', exist_ok=True)

# Convert flights DC file
convert_constraints('DCandDelset/dc_configs/raw_constraints/flights_0.0_dcs', 'DCandDelset/dc_configs/topFlightsDCs_parsed.py')