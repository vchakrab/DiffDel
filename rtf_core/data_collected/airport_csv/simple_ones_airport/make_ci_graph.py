import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load CSV
# -----------------------------
csv_path = "/rtf_core/data_collected/voter_csv/simple_ones_voter/ncvoter_100_random_cell_data_simple.csv"  # <-- replace with your CSV path
data = pd.read_csv(csv_path)

# -----------------------------
# 2. Compute 95% confidence intervals
# -----------------------------
confidence = 0.95
n = len(data)

ci_dict = {}
for col in data.columns:
    mean_val = data[col].mean()
    std_val = data[col].std(ddof=1)
    t_score = stats.t.ppf((1 + confidence)/2, df=n-1)
    margin = t_score * (std_val / np.sqrt(n))
    ci_dict[col] = (mean_val - margin, mean_val + margin)
    print(f"{col} mean: {mean_val:.5f}, 95% CI: ({mean_val - margin:.5f}, {mean_val + margin:.5f})")

# -----------------------------
# 3. Plot mean and shaded CI
# -----------------------------
plt.figure(figsize=(8,5))
x = range(n)

for col in data.columns:
    mean_line = [data[col].mean()] * n
    ci_lower, ci_upper = ci_dict[col]
    plt.plot(x, mean_line, label=f"{col} mean")
    plt.fill_between(x, ci_lower, ci_upper, alpha=0.2, label=f"{col} 95% CI")

plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("Columns with 95% Confidence Interval")
plt.legend()
plt.grid(True)
plt.show()
