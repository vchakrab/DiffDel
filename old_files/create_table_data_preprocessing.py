from collections import defaultdict

# Replace with your file path
file_path = "../experiment_2_data.csv"

sums = defaultdict(float)
counts = defaultdict(int)

with open(file_path, "r") as f:
    for line in f:
        # Remove quotes and whitespace
        line = line.strip().strip('"')
        # Split on comma
        if not line:
            continue
        attr, value = line.split(",")
        value = float(value)
        sums[attr] += value
        counts[attr] += 1

# Compute averages
averages = {attr: sums[attr] / counts[attr] for attr in sums}

# Print results
for attr, avg in averages.items():
    print(f"{attr}: {avg}")
