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

Our implementation depends on the following requirements, which we will guide you on how to install. 
# Requirements

- **Python 3.10+**
- **MySQL** (e.g., MySQL v9.4.0) — see install instructions below
- **Gurobi** installed and licensed (e.g., Gurobi v13.0) — required by `min.py`.
- **LaTeX** — required by `graph.py` for figure rendering (`text.usetex = True`); needs `latex`, `dvipng`, and `ghostscript` on your PATH
- **Python packages** (install via `pip install -r requirements.txt`):
  - `mysql-connector-python` — MySQL driver
  - `numpy`, `pandas`, `scipy` — data processing
  - `matplotlib` — figure generation
  - `gurobipy` — Gurobi Python bindings (requires Gurobi to be installed first)
  
3. **Install MySQL** and set the root password to "my_password" as it is already configured in `config.py`.

### Mac
```bash
brew install mysql
brew services start mysql
mysql_secure_installation   # set root password when prompted
```

3. **Install Gurobi** and activate your license (see [Requirements](#requirements) below). 
For more instructions on setting up an Academic License with Gurobi.
For a tutorial [click here](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer). 
For more information on how to obtain a free academic license [click here](https://www.gurobi.com/academia/academic-program-and-licenses/).
4. **Install LaTeX** (required for figure rendering — see [Requirements](#requirements) below)
5. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
6. **Run**: `python main.py`

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

