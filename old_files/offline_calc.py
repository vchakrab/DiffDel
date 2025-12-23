import os
import time
from collections import Counter
from exponential_deletion import *
# assumes these are already imported in your environment
# from your_str_module import (
#     clean_raw_dcs,
#     build_exponential_template_str
# )

def offline_build_template(
    dataset: str,
    target_cell: str,
    output_root: str = "offline_templates",
    alpha: float = 1.0,
    beta: float = 0.5,
    epsilon: float = 1.0
):
    """
    OFFLINE ONLY:
    Builds and saves exponential deletion templates for a dataset + target.
    """

    print("\n" + "=" * 80)
    print(f"OFFLINE TEMPLATE BUILD | Dataset: {dataset} | Target: {target_cell}")
    print("=" * 80)

    # --------------------------------------------------
    # Load denial constraints
    # --------------------------------------------------
    try:
        if dataset == "ncvoter":
            dataset_module_name = "NCVoter"
        else:
            dataset_module_name = dataset.capitalize()

        dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        raw_dcs = dc_module.denial_constraints

        print(f"Loaded {len(raw_dcs)} denial constraints")
    except ImportError as e:
        print(f"[ERROR] Failed to load DCs for dataset '{dataset}'")
        raise e

    # --------------------------------------------------
    # Clean DCs → hyperedges
    # --------------------------------------------------
    hyperedges = clean_raw_dcs(raw_dcs)
    print(f"Cleaned into {len(hyperedges)} hyperedges")

    # --------------------------------------------------
    # Compute initial_known (schema-level)
    # --------------------------------------------------
    all_attributes = set(attr for he in hyperedges for attr in he)
    attr_counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_attrs = {a for a, c in attr_counts.items() if c > 1}

    initial_known = all_attributes - {target_cell} - intersecting_attrs

    print(f"Total attributes: {len(all_attributes)}")
    print(f"Intersecting attributes: {len(intersecting_attrs)}")
    print(f"Initial known set size: {len(initial_known)}")

    # --------------------------------------------------
    # Edge weights (uniform, schema-level)
    # --------------------------------------------------
    edge_weights = {i: 0.8 for i in range(len(hyperedges))}

    # --------------------------------------------------
    # Output path
    # --------------------------------------------------
    dataset_dir = os.path.join(output_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    output_path = os.path.join(dataset_dir, f"{target_cell}.pkl")

    # --------------------------------------------------
    # Build & save template
    # --------------------------------------------------
    start = time.time()

    build_exponential_template_str(
        hyperedges=hyperedges,
        target_cell=target_cell,
        initial_known=initial_known,
        edge_weights=edge_weights,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        save_path=output_path
    )

    elapsed = time.time() - start

    print(f"[SAVED] {output_path}")
    print(f"Build time: {elapsed:.2f}s")

def main():
    """
    Builds offline exponential deletion templates for:
      - adult       → education
      - ncvoter     → voter_reg_num
      - airport     → latitude_deg
      - tax         → marital_status
      - hospital    → ProviderNum
    """

    template_plan = {
        "adult": "education",
        "ncvoter": "voter_reg_num",
        "airport": "latitude_deg",
        "tax": "marital_status",
        "hospital": "ProviderNum"
    }

    for dataset, target in template_plan.items():
        offline_build_template(
            dataset=dataset,
            target_cell=target,
            output_root="offline_templates",
            alpha=1.0,
            beta=0.5,
            epsilon=1.0
        )


if __name__ == "__main__":
    main()
