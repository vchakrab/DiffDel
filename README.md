**The full paper is available at:**
**`DiffDel/DiffDel-Final.pdf`**

---

# Quick Start for Reviewers

1. **Install MySQL** and set the root password to the one provided by the authors
2. **Run**: `python main.py`

That's it. `main.py` will load all datasets, run all experiments, and generate all figures automatically.

---

# Requirements

- **Python 3.10+**
- **MySQL** (e.g., MySQL v9.4.0) — see install instructions below
- **Gurobi** installed and licensed (e.g., Gurobi v13.0) — required by `min.py`
- Python packages: `mysql-connector-python`

---

# Installing MySQL

### Mac
```bash
brew install mysql
brew services start mysql
mysql_secure_installation   # set root password when prompted
```

### Linux
```bash
sudo apt install mysql-server
sudo systemctl start mysql
sudo mysql_secure_installation  # set root password when prompted
```

Set the root password to the one provided by the authors. It is already configured in `config.py` — no other changes needed.

---

# Running

```bash
python main.py
```

`main.py` does the following in order:

1. Verifies MySQL is reachable (prints a helpful error if not)
2. Creates and populates all five databases from the CSVs in `csv_files/`
3. Runs all experiments (~3–4 hours with default parameters)
4. Generates all figures into `fig/`

### Default experiment parameters (in `collect_data.py`):
```python
EM_VALUES  = [0, 0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 10]
L0_VALUES  = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LAMBDA     = 1000
```

---

# Reproducibility Artifacts

Pre-collected data and pre-generated figures are already included if you want to skip the full run:

- **`data/release_data/`** — raw experiment results used in the paper
- **`fig/`** — all figures as they appear in the paper

To regenerate figures only:
```bash
python graph.py
```

Figure 7 (dependency weight assignment) is documented in **`csv_files/compute_dc_weights.py`**.

---

# Project Structure

```
DiffDel/
├── config.py                      # MySQL credentials (pre-configured)
├── main.py                        # Entry point — run this
├── collect_data.py                # Experiment orchestration
├── exp.py / gum.py / min.py       # Core mechanisms
├── graph.py                       # Figure generation
│
├── DataGeneration/
│   ├── populate_all_datasets.py   # Called automatically by main.py
│   ├── insert_data_into_adult.py
│   ├── insert_data_into_airport.py
│   ├── insert_data_into_flight.py
│   ├── insert_data_into_hospital.py
│   └── insert_data_into_tax.py
│
├── csv_files/                     # Source CSVs (included)
├── DCandDelset/dc_configs/        # Denial constraint definitions per dataset
├── weights/weights_corrected/     # Pre-computed DC weights per dataset
└── data/release_data/             # Pre-collected experiment results
```
