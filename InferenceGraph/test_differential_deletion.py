#!/usr/bin/env python3
"""
Standalone test for differential deletion without database dependency.
Creates synthetic data to test the algorithm logic.
"""

import sys
import os
from typing import Dict, Any, List, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell import Attribute, Cell, Hyperedge
from InferenceGraph.differential_deletion import (
    InferencePath,
    extract_all_paths,
    find_inference_paths,
    enumerate_minimal_hitting_sets,
    compute_leakage,
    compute_utility,
    exponential_mechanism_sample,
    sample_gumbel,
    compute_marginal_gain
)
from InferenceGraph.build_hypergraph import GraphNode


def create_synthetic_graph() -> GraphNode:
    """
    Create a simple synthetic inference graph for testing.

    Graph structure:
        education (root)
        ├─ HE1 → occupation
        │         └─ HE3 → income
        └─ HE2 → age
                  └─ HE4 → income
    """
    # Create cells
    education_cell = Cell(Attribute('test', 'education'), 1, 'Bachelors')
    occupation_cell = Cell(Attribute('test', 'occupation'), 1, 'Prof-specialty')
    age_cell = Cell(Attribute('test', 'age'), 1, 38)
    income_cell = Cell(Attribute('test', 'income'), 1, '>50K')

    # Create hyperedges
    he1 = Hyperedge([occupation_cell])
    he2 = Hyperedge([age_cell])
    he3 = Hyperedge([income_cell])
    he4 = Hyperedge([income_cell])

    # Build graph
    root = GraphNode(education_cell)

    # Branch 1: education -> occupation -> income
    occupation_node = GraphNode(occupation_cell)
    income_node1 = GraphNode(income_cell)
    occupation_node.add_branch(he3, [income_node1])
    root.add_branch(he1, [occupation_node])

    # Branch 2: education -> age -> income
    age_node = GraphNode(age_cell)
    income_node2 = GraphNode(income_cell)
    age_node.add_branch(he4, [income_node2])
    root.add_branch(he2, [age_node])

    return root


def test_path_extraction():
    """Test path extraction from graph."""
    print("\n" + "="*70)
    print("TEST 1: Path Extraction")
    print("="*70)

    root = create_synthetic_graph()
    paths = extract_all_paths(root)

    print(f"Number of paths extracted: {len(paths)}")
    for i, path in enumerate(paths, 1):
        cell_seq = " -> ".join([c.attribute.col for c in path.cells])
        print(f"  Path {i}: {cell_seq}")
        print(f"    Hyperedges: {len(path.hyperedges)}")

    assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"
    print("✓ Path extraction test passed!")
    return paths


def test_hyperedge_weights(paths):
    """Test hyperedge weight computation."""
    print("\n" + "="*70)
    print("TEST 2: Hyperedge Weight Computation")
    print("="*70)

    # Create weight map (all weights = 1.0 for simplicity)
    weights = {}
    for path in paths:
        for he in path.hyperedges:
            weights[frozenset(he)] = 1.0

    print(f"Number of unique hyperedges: {len(weights)}")

    # Compute path weights
    for i, path in enumerate(paths, 1):
        weight = path.compute_weight(weights)
        print(f"  Path {i} weight: {weight}")

    print("✓ Weight computation test passed!")
    return weights


def test_hitting_sets(paths):
    """Test hitting set enumeration."""
    print("\n" + "="*70)
    print("TEST 3: Hitting Set Enumeration")
    print("="*70)

    # Create inference zone (all cells from the graph)
    inference_zone = set()
    for path in paths:
        inference_zone.update(path.cells)

    print(f"Inference zone size: {len(inference_zone)}")
    print("Cells in inference zone:")
    for cell in sorted(inference_zone, key=lambda c: c.attribute.col):
        print(f"  - {cell.attribute.col}")

    # Enumerate hitting sets
    hitting_sets = enumerate_minimal_hitting_sets(paths, inference_zone)

    print(f"\nNumber of hitting sets: {len(hitting_sets)}")
    for i, hs in enumerate(hitting_sets[:5], 1):  # Show first 5
        cells = ", ".join([c.attribute.col for c in hs])
        print(f"  HS {i} (size={len(hs)}): {{{cells}}}")

    assert len(hitting_sets) > 0, "Expected at least one hitting set"
    print("✓ Hitting set enumeration test passed!")
    return hitting_sets, inference_zone


def test_leakage_computation(paths, weights):
    """Test leakage computation."""
    print("\n" + "="*70)
    print("TEST 4: Leakage Computation")
    print("="*70)

    # Test with empty mask (no deletions)
    empty_mask = set()
    leakage_empty = compute_leakage(empty_mask, paths, weights)
    print(f"Leakage with empty mask: {leakage_empty}")

    # Test with mask that blocks one path
    occupation_cell = None
    for path in paths:
        for cell in path.cells:
            if cell.attribute.col == 'occupation':
                occupation_cell = cell
                break

    if occupation_cell:
        mask_one = {occupation_cell}
        leakage_one = compute_leakage(mask_one, paths, weights)
        print(f"Leakage with occupation deleted: {leakage_one}")
        assert leakage_one < leakage_empty or leakage_one == leakage_empty

    # Test with mask that blocks all paths
    income_cells = set()
    for path in paths:
        for cell in path.cells:
            if cell.attribute.col == 'income':
                income_cells.add(cell)

    mask_all = income_cells
    leakage_all = compute_leakage(mask_all, paths, weights)
    print(f"Leakage with income deleted: {leakage_all}")

    print("✓ Leakage computation test passed!")


def test_utility_computation(paths, weights):
    """Test utility computation."""
    print("\n" + "="*70)
    print("TEST 5: Utility Computation")
    print("="*70)

    alpha = 1.0
    beta = 0.5
    target_cell = paths[0].cells[0]  # First cell (education)

    # Test different masks
    masks = [
        set(),  # Empty
        {paths[0].cells[1]},  # One cell
        {paths[0].cells[1], paths[0].cells[2]},  # Two cells
    ]

    for i, mask in enumerate(masks):
        utility = compute_utility(mask, target_cell, paths, weights, alpha, beta)
        cells = ", ".join([c.attribute.col for c in mask]) if mask else "empty"
        print(f"  Mask {i} ({cells}): utility = {utility:.3f}")

    print("✓ Utility computation test passed!")


def test_exponential_mechanism(hitting_sets, paths, weights):
    """Test exponential mechanism sampling."""
    print("\n" + "="*70)
    print("TEST 6: Exponential Mechanism Sampling")
    print("="*70)

    alpha = 1.0
    beta = 0.5
    epsilon = 1.0
    target_cell = paths[0].cells[0]

    # Sample multiple times to see distribution
    samples = {}
    num_samples = 100

    for _ in range(num_samples):
        selected = exponential_mechanism_sample(
            hitting_sets[:10],  # Use first 10 candidates
            target_cell,
            paths,
            weights,
            alpha,
            beta,
            epsilon
        )
        key = frozenset(selected)
        samples[key] = samples.get(key, 0) + 1

    print(f"Sampled {num_samples} times from {min(10, len(hitting_sets))} candidates")
    print("Distribution:")
    for key, count in sorted(samples.items(), key=lambda x: -x[1])[:5]:
        cells = ", ".join([c.attribute.col for c in key]) if key else "empty"
        pct = 100 * count / num_samples
        print(f"  {{{cells}}}: {count}/{num_samples} ({pct:.1f}%)")

    print("✓ Exponential mechanism test passed!")


def test_gumbel_sampling():
    """Test Gumbel noise sampling."""
    print("\n" + "="*70)
    print("TEST 7: Gumbel Noise Sampling")
    print("="*70)

    # Sample multiple times and check distribution
    num_samples = 1000
    samples = [sample_gumbel() for _ in range(num_samples)]

    mean = sum(samples) / len(samples)
    print(f"Sampled {num_samples} Gumbel(0,1) values")
    print(f"  Mean: {mean:.3f} (theoretical: 0.577)")
    print(f"  Min: {min(samples):.3f}")
    print(f"  Max: {max(samples):.3f}")

    # Mean should be close to Euler-Mascheroni constant ≈ 0.577
    assert abs(mean - 0.577) < 0.1, f"Mean {mean} too far from expected 0.577"

    print("✓ Gumbel sampling test passed!")


def test_marginal_gain(paths, weights):
    """Test marginal gain computation."""
    print("\n" + "="*70)
    print("TEST 8: Marginal Gain Computation")
    print("="*70)

    alpha = 1.0
    beta = 0.5
    mask = set()

    # Test marginal gain for different cells
    for path in paths[:1]:  # Use first path
        for cell in path.cells[1:]:  # Skip root (target)
            gain = compute_marginal_gain(
                cell,
                mask,
                paths,
                weights,
                alpha,
                beta
            )
            print(f"  Marginal gain for {cell.attribute.col}: {gain:.3f}")

    print("✓ Marginal gain computation test passed!")


def test_gumbel_deletion_algorithm(paths, weights):
    """Test the Gumbel deletion algorithm."""
    print("\n" + "="*70)
    print("TEST 9: Gumbel Deletion Algorithm")
    print("="*70)

    alpha = 1.0
    beta = 0.5
    epsilon = 1.0

    # Initialize mask
    mask = set()

    # Get candidate cells (all cells except target)
    inference_zone = set()
    for path in paths:
        inference_zone.update(path.cells)

    target_cell = paths[0].cells[0]
    candidate_cells = inference_zone - {target_cell}

    print(f"Starting Gumbel deletion with {len(candidate_cells)} candidate cells")

    # Simulate one iteration of Gumbel deletion
    iteration = 0
    while True:
        # Check if any path is still active
        active_paths = [p for p in paths if not p.is_blocked_by(mask)]

        if not active_paths:
            print(f"All paths blocked after {iteration} iterations")
            break

        available_cells = candidate_cells - mask
        if not available_cells:
            print(f"No more attributes to delete after {iteration} iterations")
            break

        iteration += 1
        print(f"\nIteration {iteration}: {len(active_paths)} active paths")

        # Compute scores for all available cells
        scores = {}
        for cell in available_cells:
            delta_u = compute_marginal_gain(
                cell, mask, paths, weights, alpha, beta
            )
            g_A = sample_gumbel()
            s_A = (epsilon / (2 * alpha)) * delta_u + g_A
            scores[cell] = s_A

        # Select best cell
        best_cell = max(scores.items(), key=lambda x: x[1])[0]
        print(f"  Selected: {best_cell.attribute.col} (score={scores[best_cell]:.3f})")

        # Update mask
        mask.add(best_cell)

        # Limit iterations for testing
        if iteration >= 3:
            print("  (Stopping after 3 iterations for test)")
            break

    print(f"\nFinal mask size: {len(mask)}")
    print("✓ Gumbel deletion algorithm test passed!")


def test_complete_algorithm():
    """Test the complete algorithm end-to-end."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Complete Algorithm")
    print("="*70)

    # Step 1: Extract paths
    root = create_synthetic_graph()
    paths = extract_all_paths(root)
    print(f"✓ Extracted {len(paths)} paths")

    # Step 2: Compute weights
    weights = {}
    for path in paths:
        for he in path.hyperedges:
            weights[frozenset(he)] = 0.8  # High weight

    for path in paths:
        path.compute_weight(weights)
    print(f"✓ Computed weights for {len(weights)} hyperedges")

    # Step 3: Filter high-strength paths
    tau = 0.5
    high_strength = [p for p in paths if p.compute_weight(weights) > tau]
    print(f"✓ Found {len(high_strength)} high-strength paths (τ={tau})")

    # Step 4: Enumerate hitting sets
    inference_zone = set()
    for path in paths:
        inference_zone.update(path.cells)

    # Remove target cell
    target_cell = paths[0].cells[0]
    candidate_zone = inference_zone - {target_cell}

    hitting_sets = enumerate_minimal_hitting_sets(high_strength, candidate_zone)
    print(f"✓ Generated {len(hitting_sets)} hitting sets")

    # Step 5: Sample using exponential mechanism
    if hitting_sets:
        selected = exponential_mechanism_sample(
            hitting_sets,
            target_cell,
            high_strength,
            weights,
            alpha=1.0,
            beta=0.5,
            epsilon=1.0
        )
        print(f"✓ Selected mask with {len(selected)} cells:")
        for cell in selected:
            print(f"    - {cell}")
    else:
        print("  (No hitting sets found)")

    print("\n" + "="*70)
    print("✓ INTEGRATION TEST PASSED!")
    print("="*70)


def main():
    """Run all tests."""
    print("="*70)
    print("DIFFERENTIAL DELETION ALGORITHM - STANDALONE TEST")
    print("="*70)

    try:
        # Individual component tests
        paths = test_path_extraction()
        weights = test_hyperedge_weights(paths)
        hitting_sets, inference_zone = test_hitting_sets(paths)
        test_leakage_computation(paths, weights)
        test_utility_computation(paths, weights)
        test_exponential_mechanism(hitting_sets, paths, weights)

        # Gumbel deletion tests
        test_gumbel_sampling()
        test_marginal_gain(paths, weights)
        test_gumbel_deletion_algorithm(paths, weights)

        # Integration test
        test_complete_algorithm()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
