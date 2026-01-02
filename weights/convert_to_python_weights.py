#!/usr/bin/env python3
import csv

# =========================
# USER SETTINGS
# =========================
INPUT_CSV = "flights_semantic_dcs_w_norm_20251231_063113.csv"   # path to your CSV
OUTPUT_PY = "flight_weights.py"       # output python file
LIST_NAME = "WEIGHTS"                # name of the Python list
WEIGHT_COL = "w_norm"                # column holding the weight
DEDUP = False                        # usually FALSE for weights


def main():
    weights = []

    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)

        if WEIGHT_COL not in reader.fieldnames:
            raise RuntimeError(
                f"Weight column '{WEIGHT_COL}' not found. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            w = row.get(WEIGHT_COL)

            if w is None or w == "":
                continue

            try:
                w = float(w)
            except ValueError:
                continue

            weights.append(w)

    if DEDUP:
        # preserve order
        seen = set()
        deduped = []
        for w in weights:
            if w not in seen:
                seen.add(w)
                deduped.append(w)
        weights = deduped

    # -------------------------
    # Write Python file
    # -------------------------
    with open(OUTPUT_PY, "w") as out:
        out.write("# Auto-generated weights\n")
        out.write("# Order matches generated constraints\n\n")
        out.write(f"{LIST_NAME} = [\n")
        for w in weights:
            out.write(f"    {w},\n")
        out.write("]\n")

    print(f"Wrote {len(weights)} weights to {OUTPUT_PY}")


if __name__ == "__main__":
    main()
