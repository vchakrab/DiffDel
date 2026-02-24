import os
import re
import zipfile
import shutil
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ZIP_PATH = "tmp_extract 2.zip"   # <-- this is the new zip
EXTRACT_DIR = "exp_extracted_v2"
OUTPUT_DIR = "processed_output_v2"

KNOWN_DATASETS = {"airport", "hospital", "tax", "ncvoter", "adult"}

# ============================================================
# 1️⃣ UNZIP
# ============================================================

if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)

# ============================================================
# 2️⃣ PREP OUTPUT FILES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

master_path = os.path.join(OUTPUT_DIR, "combined_exp_data.csv")

if os.path.exists(master_path):
    os.remove(master_path)

dataset_paths = {
    ds: os.path.join(OUTPUT_DIR, f"{ds}_exp_data.csv")
    for ds in KNOWN_DATASETS
}

for path in dataset_paths.values():
    if os.path.exists(path):
        os.remove(path)

header_written_master = False
header_written_ds = {ds: False for ds in KNOWN_DATASETS}

# ============================================================
# 3️⃣ WALK DIRECTORY + STREAM PROCESS
# ============================================================

for root, _, files in os.walk(EXTRACT_DIR):
    for file in files:

        if not file.endswith(".csv"):
            continue

        # Extract epsilon_m and L0 from filename
        match = re.search(
            r"em[_\-]?([0-9eE\.\-]+).*L0[_\-]?([0-9eE\.\-]+)",
            file
        )

        if not match:
            continue

        epsilon_m = float(match.group(1).rstrip("."))
        L0 = float(match.group(2).rstrip("."))

        full_path = os.path.join(root, file)
        folder = os.path.basename(root)

        try:
            reader = pd.read_csv(full_path, chunksize=100000, encoding="latin1")
        except:
            continue

        for chunk in reader:

            # Determine dataset
            if "dataset" in chunk.columns:
                pass
            elif folder in KNOWN_DATASETS:
                chunk["dataset"] = folder
            else:
                continue

            chunk["epsilon_m"] = epsilon_m
            chunk["L0"] = L0

            # Write master file
            chunk.to_csv(
                master_path,
                mode="a",
                index=False,
                header=not header_written_master
            )
            header_written_master = True

            # Write per-dataset files
            for ds in KNOWN_DATASETS:
                ds_chunk = chunk[chunk["dataset"] == ds]
                if not ds_chunk.empty:
                    ds_chunk.to_csv(
                        dataset_paths[ds],
                        mode="a",
                        index=False,
                        header=not header_written_ds[ds]
                    )
                    header_written_ds[ds] = True

print("=====================================")
print("Finished processing new dataset.")
print("Master file:", master_path)
print("Dataset files:")
for ds, path in dataset_paths.items():
    print(" -", path)
print("=====================================")