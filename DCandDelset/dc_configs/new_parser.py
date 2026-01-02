#!/usr/bin/env python3
import csv

# =========================
# USER SETTINGS
# =========================
INPUT_CSV = "ncvoter_semantic_dcs_w_norm_20251231_063113.csv"   # path to your CSV
OUTPUT_TXT = "topNCVoterDCs.txt"          # output TEXT file
DEDUP = False                                    # remove duplicates


def canon_attr(tok: str) -> str:
    """
    Canonicalize attribute names:
      Education-num  -> education_num
      Hours-per-week -> hours_per_week
      Marital-status -> marital_status
    """
    s = tok.strip()
    s = s.replace("-", "_").replace(" ", "_")
    s = s.lower()
    while "__" in s:
        s = s.replace("__", "_")
    return s


def parse_premise(premise_cell: str):
    """
    premise column example:
      "age,fnlwgt,education"
    """
    if not premise_cell:
        return []
    parts = [p.strip() for p in premise_cell.split(",") if p.strip()]
    return [canon_attr(p) for p in parts]


def build_constraint(prem_attrs, target_attr):
    """
    Builds:
      not (t1.a==t2.a and ... and t1.target<>t2.target)
    """
    clauses = [f"t1.{a}==t2.{a}" for a in prem_attrs]
    clauses.append(f"t1.{target_attr}<>t2.{target_attr}")
    return f"not ({' and '.join(clauses)})"


def main():
    constraints = []

    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)

        required = {"premise", "target"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing CSV columns: {missing}")

        for row in reader:
            prem = parse_premise(row.get("premise", ""))
            target = canon_attr(row.get("target", ""))

            if not prem or not target:
                continue

            c = build_constraint(prem, target)
            constraints.append(c)

    if DEDUP:
        seen = set()
        deduped = []
        for c in constraints:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        constraints = deduped

    # -------------------------
    # Write TEXT file
    # -------------------------
    with open(OUTPUT_TXT, "w") as out:
        for c in constraints:
            out.write(c + "\n")

    print(f"Wrote {len(constraints)} constraints to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
