**The main paper (12 pages + references) is available at:**
**The appendix is available at:**
**`DiffDel/DiffDel-Final.pdf`**

The baseline `min.py` used in this project is available at [click here] (https://www.vldb.org/pvldb/vol18/p3435-chakraborty.pdf). 
The codebase for the same project is available at [click here] (https://github.com/HPI-Information-Systems/P2E2-Erasure). 

# Figures and Plots Used in the Paper are Available in the Repository Folders Listed Below:
- **`data/release_data/`** — raw experiment results used in the paper
- **`fig/release_figures/`** — all figures as they appear in the paper
---

# Quick Start Guide

1. Clone the repository to your machine.

Our implementation depends on the following requirements, which `install.sh` will install for you automatically.

# Requirements

- **Python 3.10+**
- **MySQL** (e.g., MySQL v9.4.0) — set up automatically by `install.sh`
- **Gurobi** installed and licensed (e.g., Gurobi v13.0) — required by `min.py`, activated automatically by `install.sh`
- **LaTeX** — required by `graph.py` for figure rendering (`text.usetex = True`); needs `latex`, `dvipng`, and `ghostscript` on your PATH — installed automatically by `install.sh`
- **Python packages** (installed automatically into a virtual environment via `requirements.txt`):
  - `mysql-connector-python` — MySQL driver
  - `numpy`, `pandas`, `scipy` — data processing
  - `matplotlib` — figure generation
  - `gurobipy` — Gurobi Python bindings

2. **Run the setup script:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```
   This single script works on both macOS and Linux, starting from a clean machine with nothing pre-installed. It will:
   - Install any missing prerequisites (`git`, `curl`, Python 3, build tools)
   - Install and start MySQL, then prompt you to set the root password (press Enter to accept the default `my_password`, which matches what's already configured in `config.py`)
   - Walk you through getting a free Gurobi license and prompt you to paste your license key, then activate it automatically (leave blank to skip and use the default trial license instead)
   - Install LaTeX (`dvipng`, `ghostscript`) for figure rendering
   - Create a Python virtual environment (`venv/`) and install all dependencies from `requirements.txt`
   - Finish with a pass/fail check confirming git, Python, pip, `gurobipy`, MySQL, and LaTeX are all working

   It's safe to re-run `./install.sh` at any time — every step skips automatically if it's already done.

3. **Activate the virtual environment** in any new terminal session before running the project:
   ```bash
   source venv/bin/activate
   ```

4. **Run**: `python3 main.py`
That's it. `main.py` will load all datasets, run all experiments, and generate all figures automatically. 
To see results from the paper as they appear, look at `data/release_data` and `fig/release_figures`. 
---

# Running

`main.py` does the following in order:

1. Verifies MySQL is reachable (prints a helpful error if not)
2. Creates and populates all five databases from the CSVs in `csv_files/`
3. Runs all experiments (~3–4 hours with default parameters)
4. Generates all figures into `fig/`

### Default experiment parameters (in `collect_data.py`):
```python
EM_VALUES  = [0, 0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 10]
L0_VALUES  = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

```bash
python main.py
```

---

# Troubleshooting
 
**`main.py` reports that MySQL is running but credentials are wrong**
 
This usually means the root password wasn't set correctly during `install.sh` — for example, if `mysql_native_password` isn't available on your MySQL version. Reset it explicitly:
 
- **Linux:**
```bash
  sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY 'my_password'; FLUSH PRIVILEGES;"
```
- **macOS:**
```bash
  mysql -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY 'my_password'; FLUSH PRIVILEGES;"
```
 
Then confirm it worked:
```bash
mysql -u root -pmy_password -e "SELECT 1;"
```

---

# Reproducibility Artifacts

Pre-collected data and pre-generated figures are already included if you want to skip the full run:

- **`data/release_data/`** — raw experiment results used in the paper
- **`fig/release_figures/`** — all figures as they appear in the paper

# Dependency Weight Artifacts

Figure 7 (dependency weight assignment) is documented in **`csv_files/compute_dc_weights.py`**.
The `csv_files` folder also contains pre-computed results for gamma_frac = 0.25 and gamma_frac = 0.5.

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

# Acknowledgements

Experiment scripts were generated with the help of Claude.
