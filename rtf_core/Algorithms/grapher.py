import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
csv_path = "/Users/adhariya/src/RTF25/rtf_core/airport_longitude_deg_dataset_size_100_cell_data.csv"
x_col = "DBSize"
y_col = "Time"

# === LOAD DATA ===
df = pd.read_csv(csv_path)
df = df.sort_values(by=x_col)

# === GROUP BY ===
groups = df.groupby(x_col)
totals = groups[y_col].sum()

# === PLOT TOTALS ===
x = totals.index.to_numpy()
y = totals.values

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'o-', color='green', linewidth=2, markersize=6, label='Total Time')

plt.xlabel("Dataset Size", fontsize=11)
plt.ylabel("Total Runtime (seconds)", fontsize=11)
plt.title("Total Runtime vs Dataset Size", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# === OPTIONAL: Save summary ===
summary = pd.DataFrame({
    x_col: x,
    "Total_Time": y,
    "Count": groups[y_col].count().values
})
summary.to_csv("summary_total_time.csv", index=False)


#CODE TO GRAPH MEAN + CI
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
#
# # === CONFIG ===
# csv_path = "/Users/adhariya/src/RTF25/rtf_core/airport_elevation_ft_dataset_size_100_cell_data.csv"  # <-- change to your actual CSV filename
# x_col = "DBSize"   # column in CSV for number of denial constraints
# y_col = "Time"          # column in CSV for time measurements
# confidence = 0.95
#
# # === LOAD DATA ===
# df = pd.read_csv(csv_path)
#
# # Ensure numeric sorting
# df = df.sort_values(by=x_col)
#
# # === GROUP BY number of denial constraints ===
# groups = df.groupby(x_col)
# means = groups[y_col].sum()
# counts = groups[y_col].count()
# stds = groups[y_col].std()
#
# # === 95% CONFIDENCE INTERVALS ===
# sem = stds / np.sqrt(counts)
# t_val = stats.t.ppf((1 + confidence) / 2., counts - 1)
# ci_range = sem * t_val
# print(ci_range)
#
# # === PLOT ===
# x = means.index.to_numpy()
# y = means.values
# lower = y - ci_range.values
# upper = y + ci_range.values
#
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, 'o-', color='blue', linewidth=2, markersize=6, label='Mean Time')
# plt.fill_between(x, lower, upper, color='blue', alpha=0.2, label='95% Confidence Interval')
#
# plt.xlabel("Dataset Size", fontsize=11)
# plt.ylabel("Runtime (seconds)", fontsize=11)
# plt.title("Runtime vs Data Set Size (with 95% CI)", fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # === OPTIONAL: Save summary data with confidence intervals ===
# summary = pd.DataFrame({
#     x_col: x,
#     "Mean": y,
#     "Lower_CI": lower,
#     "Upper_CI": upper,
#     "Count": counts.values
# })
# summary.to_csv("summary_with_CI.csv", index=False)
