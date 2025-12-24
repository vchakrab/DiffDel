#!/usr/bin/env python3
import csv
import sys


def main(input_file, dataset):
    input_csv = input_file
    dataset_name = dataset

    weights = []

    with open(input_csv, newline = "", encoding = "utf-8") as f:
        reader = csv.DictReader(f)

        if "weight" not in reader.fieldnames:
            print("Error: CSV must contain a 'weight' column")
            sys.exit(1)

        for row in reader:
            raw = row.get("weight", "").strip()

            if raw == "":
                # Default weight when missing
                weights.append(1.0)
            else:
                try:
                    weights.append(float(raw))
                except ValueError:
                    # If malformed, still default to 1.0
                    weights.append(1.0)

    output_file = f"{dataset_name}_weights.py"

    with open(output_file, "w", encoding = "utf-8") as f:
        f.write("# Auto-generated weights file\n")
        f.write(f"# Dataset: {dataset_name}\n\n")
        f.write("WEIGHTS = [\n")
        for w in weights:
            f.write(f"    {w},\n")
        f.write("]\n")

    print(f"Wrote {len(weights)} weights to {output_file}")


if __name__ == "__main__":
    main("onlineretail_ALL_DCs_weights.csv", 'onlineretail_weights.py')
    main("hospital_ALL_DCs_weights_CORRECTED.csv", 'hospital_weights.py')

