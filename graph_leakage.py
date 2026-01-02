#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
INPUT_FILE = "leakage_graphs.txt"   # <-- put your pasted output here
OUTPUT_FIG = "all_leakages_hist.png"
BINS = 50

# =========================
# PARSE LEAKAGES
# =========================
leakages = []

# matches lines like:
# 2047   11           0.000000     -0.001773    {...}
line_re = re.compile(
    r"^\s*\d+\s+\d+\s+([0-9]*\.?[0-9]+)\s+[-0-9\.]+"
)

with open(INPUT_FILE, "r") as f:
    for line in f:
        m = line_re.match(line)
        if m:
            leakages.append(float(m.group(1)))

if not leakages:
    raise RuntimeError("No leakage values parsed — check input format")

print(f"Parsed {len(leakages)} leakage values")
print(f"Min leakage: {min(leakages):.6f}")
print(f"Max leakage: {max(leakages):.6f}")

# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.hist(leakages, bins=BINS, edgecolor="black")
plt.xlabel("Leakage")
plt.ylabel("Count")
plt.title("Distribution of Leakage Over All Masks")
plt.tight_layout()

plt.savefig(OUTPUT_FIG, dpi=200)
plt.show()

print(f"Saved figure to {OUTPUT_FIG}")
