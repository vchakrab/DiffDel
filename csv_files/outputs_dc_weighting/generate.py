#!/usr/bin/env python3
"""
Generate Python DC+weight files from wpos gamma CSVs
(keeping ONLY non-zero weights).

Handles DC format like:

t0.class str EQUAL t1.class str AND t0.Hours-per-week float LESS t1.Hours-per-week float
"""

import pandas as pd
from pathlib import Path


DATASETS = ["adult", "airport", "hospital", "tax", "flights"]

OPERATOR_MAP = {
    # word forms (adult, airport, hospital, flights)
    "EQUAL":          "==",
    "UNEQUAL":        "!=",
    "NOT_EQUAL":      "!=",
    "LESS":           "<",
    "GREATER":        ">",
    "LESS_EQUAL":     "<=",
    "GREATER_EQUAL":  ">=",
    # symbol forms (tax — already converted by dc_to_str)
    "==":  "==",
    "!=":  "!=",
    "<":   "<",
    ">":   ">",
    "<=":  "<=",
    ">=":  ">=",
}


def normalize_attr(attr):
    """Convert attribute names to Python-friendly format."""
    return attr.lower().replace("-", "_")


def parse_dc_string(dc_string):
    parts = dc_string.split(" AND ")
    predicates = []

    for part in parts:
        tokens = part.strip().split()

        if len(tokens) == 5:
            # Format: t0.col TYPE EQUAL t1.col
            left, op_word, right = tokens[0], tokens[2], tokens[3]
        elif len(tokens) == 3:
            # Format: t0.col EQUAL t1.col  (tax dataset)
            left, op_word, right = tokens[0], tokens[1], tokens[2]
        else:
            raise ValueError(f"Unexpected predicate token count ({len(tokens)}): {part!r}")

        left  = left.replace("t0.", "t1.").replace("t1.", "t2.")
        right = right.replace("t0.", "t1.").replace("t1.", "t2.")

        l_prefix, l_attr = left.split(".")
        r_prefix, r_attr = right.split(".")

        left  = f"{l_prefix}.{normalize_attr(l_attr)}"
        right = f"{r_prefix}.{normalize_attr(r_attr)}"

        op = OPERATOR_MAP.get(op_word)
        if op is None:
            raise ValueError(f"Unknown operator: {op_word}")

        predicates.append((left, op, right))

    return predicates


def detect_columns(df):
    dc_col = None
    weight_col = None

    for col in df.columns:
        lower = col.lower()
        if "weight" in lower:
            weight_col = col
        if "dc" in lower or "constraint" in lower:
            dc_col = col

    if dc_col is None:
        dc_col = df.columns[0]

    if weight_col is None:
        weight_col = df.columns[-1]

    return dc_col, weight_col


def process_dataset(dataset, base_dir):
    csv_name = f"{dataset}_dc_weights_wpos_gamma0p25.csv"
    output_name = f"{dataset}_dc_weights_wpos_gamma0p25.py"

    csv_path = base_dir / csv_name
    output_path = base_dir / output_name

    if not csv_path.exists():
        # print(f"⚠ Skipping {dataset}: CSV not found.")
        return

    # print(f"\nProcessing {dataset}...")

    df = pd.read_csv(csv_path)
    dc_col, weight_col = detect_columns(df)

    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df[df[weight_col] > 0].reset_index(drop=True)

    # print(f"  Keeping {len(df)} DCs with weight > 0")

    denial_constraints = []
    weights = []

    for _, row in df.iterrows():
        dc = parse_dc_string(row[dc_col])
        weight = float(row[weight_col])

        denial_constraints.append(dc)
        weights.append(weight)

    with open(output_path, "w") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("denial_constraints = [\n")
        for dc in denial_constraints:
            f.write(f"    {repr(dc)},\n")
        f.write("]\n\n")

        f.write("WEIGHTS = [\n")
        for w in weights:
            f.write(f"    {w},\n")
        f.write("]\n")

    # print(f"  ✔ Wrote {output_name}")


def main():
    base_dir = Path(__file__).resolve().parent
    # print("Generating DC weight Python files...")

    for dataset in DATASETS:
        process_dataset(dataset, base_dir)

    # print("\nDone.")


if __name__ == "__main__":
    main()