import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import zipfile


# Re-extract archive (state was reset)
zip_path = "/mnt/data/Archive.zip"
extract_path = "/mnt/data/extracted"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# ---- epsilon_min values provided ----
epsilon_min_values = {
    "airport": 4.0,
    "hospital_data": 2.0,
    "tax_data": 10.3,
    "flight_data": 5.5,
    "adult_data": 0.1
}

# ---- P_min values ----
p_min_values = {
    "hospital_data": 2 / 114919,
    "tax_data": 1 / 99904,
    "adult_data": 49 / 32561,
    "flight_data": 137 / 499308,
    "airport": 1 / 55100
}

# ---- Load CSVs ----
all_dfs = []

for file in os.listdir(extract_path):
    if file.endswith(".csv") and not file.startswith("._"):
        df = pd.read_csv(os.path.join(extract_path, file))
        eps_m = float(file.split("_")[-1].replace(".csv", ""))
        df["epsilon_m"] = eps_m
        all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index = True)

# Clean dataset names
full_df["dataset"] = full_df["dataset"].replace({
    "hospital": "hospital_data",
    "tax": "tax_data",
    "adult": "adult_data",
    "flight": "flight_data"
})

# Ensure numeric
full_df["leakage"] = pd.to_numeric(full_df["leakage"], errors = "coerce")
full_df["mask_size"] = pd.to_numeric(full_df["mask_size"], errors = "coerce")
full_df = full_df.dropna(subset = ["leakage", "mask_size"])


def epsilon_r(L, p_min, eps = 1e-12):
    L = np.clip(L, eps, 1 - eps)
    p_min = np.clip(p_min, eps, 1 - eps)
    odds_L = L / (1 - L)
    odds_p = p_min / (1 - p_min)
    return np.maximum(0, np.log(odds_L / odds_p))


# ================================
# FILE 1: mask_size vs total_epsilon_achieved
# ================================
pdf1_path = "/mnt/data/mask_vs_total_epsilon.pdf"

with PdfPages(pdf1_path) as pdf:
    for dataset in epsilon_min_values.keys():

        df_d = full_df[full_df["dataset"] == dataset]
        if df_d.empty:
            continue

        p_min = p_min_values[dataset]

        df_d = df_d.copy()
        df_d["epsilon_r"] = epsilon_r(df_d["leakage"].values, p_min)
        df_d["total_epsilon"] = df_d["epsilon_r"] + df_d["epsilon_m"]

        grouped = df_d.groupby("mask_size")["total_epsilon"]
        mean_eps = grouped.mean()
        std_eps = grouped.std()
        count = grouped.count()
        ci = 1.96 * std_eps / np.sqrt(count)

        mask_sizes = mean_eps.index.values
        eps_center = mean_eps.values
        eps_lower = (mean_eps - ci).values
        eps_upper = (mean_eps + ci).values

        order = np.argsort(eps_center)
        mask_sizes = mask_sizes[order]
        eps_center = eps_center[order]
        eps_lower = eps_lower[order]
        eps_upper = eps_upper[order]

        plt.figure()
        plt.plot(eps_center, mask_sizes)
        plt.fill_betweenx(mask_sizes, eps_lower, eps_upper, alpha = 0.2)
        plt.xlabel("total_epsilon_achieved (epsilon_m + epsilon_r)")
        plt.ylabel("mask_size")
        plt.title(dataset)
        pdf.savefig()
        plt.close()

# ================================================
# FILE 2: mask_size vs total_epsilon / epsilon_min
# ================================================
pdf2_path = "/mnt/data/mask_vs_total_epsilon_ratio.pdf"

with PdfPages(pdf2_path) as pdf:
    for dataset in epsilon_min_values.keys():

        df_d = full_df[full_df["dataset"] == dataset]
        if df_d.empty:
            continue

        p_min = p_min_values[dataset]
        epsilon_min = epsilon_min_values[dataset]

        df_d = df_d.copy()
        df_d["epsilon_r"] = epsilon_r(df_d["leakage"].values, p_min)
        df_d["total_epsilon"] = df_d["epsilon_r"] + df_d["epsilon_m"]
        df_d["epsilon_ratio"] = df_d["total_epsilon"] / epsilon_min

        grouped = df_d.groupby("mask_size")["epsilon_ratio"]
        mean_eps = grouped.mean()
        std_eps = grouped.std()
        count = grouped.count()
        ci = 1.96 * std_eps / np.sqrt(count)

        mask_sizes = mean_eps.index.values
        eps_center = mean_eps.values
        eps_lower = (mean_eps - ci).values
        eps_upper = (mean_eps + ci).values

        order = np.argsort(eps_center)
        mask_sizes = mask_sizes[order]
        eps_center = eps_center[order]
        eps_lower = eps_lower[order]
        eps_upper = eps_upper[order]

        plt.figure()
        plt.plot(eps_center, mask_sizes)
        plt.fill_betweenx(mask_sizes, eps_lower, eps_upper, alpha = 0.2)
        plt.xlabel("total_epsilon_achieved / epsilon_min")
        plt.ylabel("mask_size")
        plt.title(dataset)
        pdf.savefig()
        plt.close()

print("Generated files:")
print(pdf1_path)
print(pdf2_path)
