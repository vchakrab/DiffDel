# Requirements

- **Python** (e.g., Python 3.10+)
- **MySQL** (e.g., MySQL v9.4.0)
- `config.py` updated with your MySQL credentials
- `min.py` with **Gurobi** installed and licensed (e.g., Gurobi v13.0)

---

# Datasets Setup

To simplify testing, we provide all required CSV files in the `csv_files/` directory.

Use the scripts in the `DataGeneration/` directory to populate the MySQL databases.

For each dataset:

1. Open the corresponding `insert_data_into_dataset.py` file.
2. Update:
   - The CSV file paths
   - Your MySQL credentials.
3. Run the script:
   
```bash
python insert_data_into_dataset.py
```

## Run Experiments
1. All Weights are provided in `weights/weights_corrected`
2. All Dependencies are provided in 'DCandDelset/dc_configs`
  - Each dataset has depenendcies in the format `topDatasetDCs_parsed.py`
3. `collect_data.py` has the data collection scripts, in the  `run_all_experiments` method verify the parameters you want to collect data for
  - By default, these values will be run, taking approximately 3-4 hours for the data collection process
  - `EM_VALUES = [0, 0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 10]`
  - `L0_VALUES = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
  - `LAMBDA = 1000`
4. To run the experiments and graph the collected data, run
   ```bash
      python main.py
   ```
  - **`data/min/`**  
    Contains results produced by the baseline minimization mechanism (e.g., mask sizes, leakage values, runtime logs).
  - **`data/exp/`**  
    Contains experiment outputs for the Exponential Mechanism variant.
  - **`data/gum/`**  
    Contains experiment outputs for the Gumbel-based mechanism.
  - **`fig`**  
    Stores all generated plots and final figures used in evaluation (e.g., heatmaps, Pareto frontiers, runtime plots).

## Reproducibility Artifacts
  - Data used to graph figures is contained withing `data/release_data` and `fig/release_figures`
