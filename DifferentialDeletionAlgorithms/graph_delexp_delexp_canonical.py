import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Load data
# =========================
df_full = pd.read_csv("/Users/adhariya/src/DiffDel/DifferentialDeletionAlgorithms/delexp_data_v2026_2.csv")
df_canon = pd.read_csv("/Users/adhariya/src/DiffDel/DifferentialDeletionAlgorithms/delexp_canonical_data_v2026_2.csv")

df_full["variant"] = "full"
df_canon["variant"] = "canonical"

df = pd.concat([df_full, df_canon], ignore_index=True)

# =========================
# Columns
# =========================
time_cols = [
    "init_time",
    "model_time",
    "del_time",
    "delete_db_time",
    "update_time",
    "greedy_time",
]

time_cols = [c for c in time_cols if c in df.columns]

# =========================
# Average per dataset + variant
# =========================
time_avg = (
    df.groupby(["dataset", "variant"])[time_cols]
    .mean()
    .reset_index()
)

mem_avg = (
    df.groupby(["dataset", "variant"])["memory_overhead_bytes"]
    .mean()
    .reset_index()
)

leak_avg = (
    df.groupby(["dataset", "variant"])["leakage"]
    .mean()
    .reset_index()
)

datasets = sorted(df["dataset"].unique())
x = np.arange(len(datasets))
width = 0.35

# =========================
# 1. Stacked runtime plot
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

bottom_full = np.zeros(len(datasets))
bottom_canon = np.zeros(len(datasets))

for col in time_cols:
    full_vals = (
        time_avg[time_avg["variant"] == "full"]
        .set_index("dataset")
        .loc[datasets][col]
        .values
    )

    canon_vals = (
        time_avg[time_avg["variant"] == "canonical"]
        .set_index("dataset")
        .loc[datasets][col]
        .values
    )

    ax.bar(x - width / 2, full_vals, width, bottom=bottom_full, label=f"{col} (full)")
    ax.bar(x + width / 2, canon_vals, width, bottom=bottom_canon, label=f"{col} (canonical)")

    bottom_full += full_vals
    bottom_canon += canon_vals

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Average Time (seconds)")
ax.set_title("Average Runtime Breakdown per Dataset")
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# =========================
# 2. Memory usage bar plot
# =========================
mem_pivot = mem_avg.pivot(index="dataset", columns="variant", values="memory_overhead_bytes")

mem_pivot.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Average Memory Overhead (bytes)")
plt.title("Average Memory Usage per Dataset")
plt.tight_layout()
plt.show()

# =========================
# 3. Leakage bar plot
# =========================
leak_pivot = leak_avg.pivot(index="dataset", columns="variant", values="leakage")

leak_pivot.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Average Leakage")
plt.title("Average Leakage per Dataset")
plt.tight_layout()
plt.show()
