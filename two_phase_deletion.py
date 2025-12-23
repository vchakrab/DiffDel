#!/usr/bin/env python3

import pickle
import os
import time
from collections import Counter
from pprint import pformat

from exponential_deletion import (
    clean_raw_dcs,
    find_inference_paths_str,
    get_path_inference_zone_str,
    compute_possible_mask_set_str,
    filter_active_paths_str
)

# -----------------------------
# Helper: readable serialization
# -----------------------------
def template_to_string(T_attr, attribute_name=None):
    lines = []

    title = f"TEMPLATE FOR ATTRIBUTE: {attribute_name}" if attribute_name else "TEMPLATE"
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)
    lines.append("")

    # ---- I_intra ----
    lines.append("[I_INTRA]")
    for a in sorted(T_attr["I_intra"]):
        lines.append(f"  {a}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("")

    # ---- Π_intra ----
    lines.append("[Π_INTRA]  (Inference Paths)")
    for i, p in enumerate(T_attr["Π_intra"], 1):
        lines.append(f"  Path {i}: {p}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("")

    # ---- R_intra ----
    lines.append("[R_INTRA]  (Candidate Masks)")
    for i, m in enumerate(T_attr["R_intra"], 1):
        mask_str = ", ".join(sorted(m)) if m else ""
        lines.append(f"  Mask {i}: {{{mask_str}}}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("")

    # ---- Blocked / Unblocked ----
    lines.append("[BLOCKED / UNBLOCKED PATHS PER MASK]")
    lines.append("")

    for mask in sorted(T_attr["R_intra"], key=lambda x: (len(x), sorted(x))):
        mask_key = frozenset(mask)
        mask_str = ", ".join(sorted(mask)) if mask else ""
        lines.append(f"Mask: {{{mask_str}}}")

        blocked = T_attr["Blocked"].get(mask_key, [])
        unblocked = T_attr["Unblocked"].get(mask_key, [])

        lines.append("  Blocked Paths:")
        if blocked:
            for p in blocked:
                lines.append(f"    {p}")
        else:
            lines.append("    (none)")

        lines.append("  Unblocked Paths:")
        if unblocked:
            for p in unblocked:
                lines.append(f"    {p}")
        else:
            lines.append("    (none)")

        lines.append("")

    lines.append("-" * 60)
    lines.append("")

    # ---- Σ_cross ----
    lines.append("[Σ_CROSS]")
    if T_attr["Σ_cross"]:
        for c in T_attr["Σ_cross"]:
            lines.append(f"  {c}")
    else:
        lines.append("  (empty)")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# -----------------------------
# Main template builder
# -----------------------------
def build_template(dataset, attribute, save_dir="templates"):
    start_time = time.time()

    # ---- Load DCs ----
    try:
        if dataset == "ncvoter":
            dataset_module_name = "NCVoter"
        else:
            dataset_module_name = dataset.capitalize()

        dc_module_path = f"DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed"
        dc_module = __import__(dc_module_path, fromlist=["denial_constraints"])
        raw_dcs = dc_module.denial_constraints

        print(f"[INFO] Loaded {len(raw_dcs)} denial constraints")
    except ImportError as e:
        print(f"[ERROR] Failed to load DCs: {e}")
        return None

    # ---- Clean DCs into hyperedges ----
    hyperedges = clean_raw_dcs(raw_dcs)
    print(f"[INFO] Cleaned into {len(hyperedges)} hyperedges")

    # ---- Compute intra-zone ----
    all_attributes = set(attr for he in hyperedges for attr in he)
    counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_cells = {a for a, c in counts.items() if c > 1}

    initial_known = all_attributes - {attribute} - intersecting_cells

    # ---- Paths & masks ----
    paths = find_inference_paths_str(hyperedges, attribute, initial_known)
    inference_zone = get_path_inference_zone_str(paths, hyperedges, attribute)
    candidate_masks = compute_possible_mask_set_str(inference_zone)

    # ---- Blocked / Unblocked ----
    Blocked = {}
    Unblocked = {}

    for mask in candidate_masks:
        active_paths = filter_active_paths_str(
            hyperedges, paths, mask, initial_known
        )
        blocked_paths = [p for p in paths if p not in active_paths]

        Blocked[frozenset(mask)] = blocked_paths
        Unblocked[frozenset(mask)] = active_paths

    # ---- Build template ----
    T_attr = {
        "I_intra": initial_known,
        "Π_intra": paths,
        "R_intra": candidate_masks,
        "Blocked": Blocked,
        "Unblocked": Unblocked,
        "Σ_cross": []
    }

    # ---- Save files ----
    os.makedirs(save_dir, exist_ok=True)

    pkl_path = os.path.join(save_dir, f"{dataset}_{attribute}.pkl")
    txt_path = os.path.join(save_dir, f"{dataset}_{attribute}.txt")

    with open(pkl_path, "wb") as f:
        pickle.dump(T_attr, f)

    with open(txt_path, "w") as f:
        f.write(template_to_string(T_attr))

    elapsed = time.time() - start_time

    print(f"[DONE] Template built for ({dataset}, {attribute})")
    print(f"       Pickle: {pkl_path}")
    print(f"       Text:   {txt_path}")
    print(f"       Time:   {elapsed:.3f}s")

    return T_attr


# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    build_template("airport", "type")
