#!/usr/bin/env python3
"""
Differential Deletion Mechanisms
==================================

Implements two variants of the differential deletion mechanism for privacy-preserving
database sanitization with differential privacy guarantees:

1. Algorithm 1 (Exponential Deletion): Uses the exponential mechanism to sample
   from a set of candidate hitting sets.

2. Algorithm 2 (Gumbel Deletion): Uses the Gumbel trick for greedy sequential
   attribute selection with differential privacy.

Usage:
    from differential_deletion import DifferentialDeletion

    dd = DifferentialDeletion(dataset='adult', alpha=1.0, beta=0.5, epsilon=1.0, tau=0.1)

    # Algorithm 1: Exponential mechanism
    mask1 = dd.exponential_deletion(row, key, target_attr)

    # Algorithm 2: Gumbel trick
    mask2 = dd.gumbel_deletion(row, key, target_attr)
"""

import sys
import os
from typing import Any, Dict, List, Set, Tuple, FrozenSet
from collections import defaultdict, deque
from itertools import chain, combinations
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell import Attribute, Cell, Hyperedge
from fetch_row import RTFDatabaseManager
from InferenceGraph.bulid_hyperedges import HyperedgeBuilder
from InferenceGraph.build_hypergraph import build_hypergraph_tree, GraphNode


# ==============================================================================
# Path Representation and Utilities
# ==============================================================================

class InferencePath:
    """Represents a path in the inference graph as a sequence of hyperedges."""

    def __init__(self, hyperedges: List[Hyperedge], cells: List[Cell]):
        self.hyperedges = hyperedges  # Sequence of hyperedges in the path
        self.cells = cells  # Sequence of cells in the path
        self._weight = None
        self._cell_set = frozenset(cells)

    def compute_weight(self, hyperedge_weights: Dict[FrozenSet[Cell], float]) -> float:
        """Compute path weight as product of hyperedge weights."""
        if self._weight is None:
            self._weight = 1.0
            for he in self.hyperedges:
                he_key = frozenset(he)
                weight = hyperedge_weights.get(he_key, 1.0)
                self._weight *= weight
        return self._weight

    def is_blocked_by(self, mask: Set[Cell]) -> bool:
        """Check if this path is blocked by the deletion mask."""
        # A path is blocked if any cell in the path is in the mask
        return bool(self._cell_set & mask)

    def __repr__(self):
        cell_seq = " -> ".join([f"{c.attribute.col}" for c in self.cells])
        return f"Path({cell_seq}, weight={self._weight})"


# ==============================================================================
# Path Extraction
# ==============================================================================

def extract_all_paths(root: GraphNode) -> List[InferencePath]:
    """
    Extract all paths from root to leaves in the inference graph.
    Each path is a sequence of hyperedges and cells.
    """
    all_paths = []

    def dfs(node: GraphNode,
            current_hyperedges: List[Hyperedge],
            current_cells: List[Cell],
            visited: Set[Cell]):
        """DFS to find all paths from node to leaves."""

        # Add current node's cell to the path
        path_cells = current_cells + [node.cell]
        path_visited = visited | {node.cell}

        # Base case: leaf node (no branches)
        if not node.branches:
            if current_hyperedges:  # Only add paths with at least one hyperedge
                all_paths.append(InferencePath(
                    hyperedges=list(current_hyperedges),
                    cells=list(path_cells)
                ))
            return

        # Recursive case: traverse each branch
        for hyperedge, children in node.branches:
            path_hyperedges = current_hyperedges + [hyperedge]

            for child in children:
                # Avoid cycles
                if child.cell not in path_visited:
                    dfs(child, path_hyperedges, path_cells, path_visited)

        # If no valid children were explored, this is effectively a leaf
        if not any(child.cell not in path_visited
                   for _, children in node.branches
                   for child in children):
            if current_hyperedges:
                all_paths.append(InferencePath(
                    hyperedges=list(current_hyperedges),
                    cells=list(path_cells)
                ))

    dfs(root, [], [], set())
    return all_paths


def find_inference_paths(
    known_attrs: Set[str],
    target_attr: str,
    root: GraphNode,
    hyperedge_weights: Dict[FrozenSet[Cell], float],
    tau: float
) -> List[InferencePath]:
    """
    Extract high-strength inference paths.

    Args:
        known_attrs: Set of known attribute names (S in the algorithm)
        target_attr: Target attribute name (A_t in the algorithm)
        root: Root node of the inference graph
        hyperedge_weights: Mapping from hyperedge to weight
        tau: Threshold for path strength

    Returns:
        List of paths with weight > tau
    """
    # Extract all paths from the graph
    all_paths = extract_all_paths(root)

    # Filter high-strength paths (weight > tau)
    high_strength_paths = []
    for path in all_paths:
        weight = path.compute_weight(hyperedge_weights)
        if weight > tau:
            high_strength_paths.append(path)

    return high_strength_paths


# ==============================================================================
# Hitting Set Enumeration
# ==============================================================================

def powerset(iterable, max_size=None):
    """Generate powerset of an iterable up to max_size."""
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(max_size + 1))


def enumerate_hitting_sets(
    paths: List[InferencePath],
    inference_zone: Set[Cell],
    max_size: int = None
) -> List[Set[Cell]]:
    """
    Enumerate all candidate hitting sets that hit all paths.

    A hitting set M "hits" a path if M ∩ path ≠ ∅

    Args:
        paths: List of inference paths to hit
        inference_zone: Set of cells in the inference zone (candidate cells)
        max_size: Maximum size of hitting sets to consider

    Returns:
        List of hitting sets (each is a set of cells)
    """
    if not paths:
        return [set()]  # Empty set hits everything if there are no paths

    # Extract cells from each path
    path_cells = [path._cell_set for path in paths]

    # If max_size not specified, use size of inference zone
    if max_size is None:
        max_size = len(inference_zone)

    hitting_sets = []

    # Generate candidates from powerset of inference zone
    for candidate_tuple in powerset(inference_zone, max_size):
        candidate = set(candidate_tuple)

        # Check if this candidate hits all paths
        hits_all = True
        for path_set in path_cells:
            if not (candidate & path_set):  # Doesn't hit this path
                hits_all = False
                break

        if hits_all:
            hitting_sets.append(candidate)

    return hitting_sets


def enumerate_minimal_hitting_sets(
    paths: List[InferencePath],
    inference_zone: Set[Cell],
    max_candidates: int = 100
) -> List[Set[Cell]]:
    """
    Enumerate hitting sets using a combination of greedy and exhaustive search.

    This balances completeness with efficiency.

    Args:
        paths: List of inference paths to hit
        inference_zone: Set of cells in the inference zone
        max_candidates: Maximum number of candidates to generate

    Returns:
        List of hitting sets (each is a set of cells)
    """
    if not paths:
        return [set()]

    path_cells = [path._cell_set for path in paths]
    hitting_sets = []
    seen = set()  # Track unique hitting sets

    def add_unique(hs: Set[Cell]):
        """Add hitting set if not already seen."""
        key = frozenset(hs)
        if key not in seen:
            seen.add(key)
            hitting_sets.append(hs)

    # Strategy 1: Greedy hitting set (most frequent cells first)
    def greedy_hitting_set() -> Set[Cell]:
        """Build a single hitting set greedily."""
        unhit_paths = set(range(len(path_cells)))
        hitting_set = set()

        while unhit_paths:
            # Count how many unhit paths each cell hits
            cell_counts = defaultdict(int)
            for path_idx in unhit_paths:
                for cell in path_cells[path_idx]:
                    if cell in inference_zone:
                        cell_counts[cell] += 1

            if not cell_counts:
                break

            # Pick cell that hits most paths
            best_cell = max(cell_counts.items(), key=lambda x: x[1])[0]
            hitting_set.add(best_cell)

            # Remove hit paths
            unhit_paths = {i for i in unhit_paths if best_cell not in path_cells[i]}

        return hitting_set

    # Generate greedy solution
    greedy_hs = greedy_hitting_set()
    add_unique(greedy_hs)

    # Strategy 2: Generate variations by removing cells from greedy solution
    for size in range(len(greedy_hs)):
        for subset_tuple in combinations(greedy_hs, size):
            subset = set(subset_tuple)
            # Check if it's still a hitting set
            if all(subset & path_set for path_set in path_cells):
                add_unique(subset)
                if len(hitting_sets) >= max_candidates:
                    return hitting_sets

    # Strategy 3: Try single cells that appear in many paths
    cell_frequency = defaultdict(int)
    for path_set in path_cells:
        for cell in path_set:
            if cell in inference_zone:
                cell_frequency[cell] += 1

    # Try top frequent cells
    for cell, _ in sorted(cell_frequency.items(), key=lambda x: -x[1])[:10]:
        single_cell_set = {cell}
        if all(single_cell_set & path_set for path_set in path_cells):
            add_unique(single_cell_set)
            if len(hitting_sets) >= max_candidates:
                return hitting_sets

    # Strategy 4: Try pairs of frequent cells
    frequent_cells = [c for c, _ in sorted(cell_frequency.items(), key=lambda x: -x[1])[:5]]
    for i, cell1 in enumerate(frequent_cells):
        for cell2 in frequent_cells[i+1:]:
            pair = {cell1, cell2}
            if all(pair & path_set for path_set in path_cells):
                add_unique(pair)
                if len(hitting_sets) >= max_candidates:
                    return hitting_sets

    # Strategy 5: For small problems, do exhaustive search
    if len(inference_zone) <= 10:
        for size in range(1, min(len(inference_zone), len(paths)) + 1):
            for candidate_tuple in combinations(inference_zone, size):
                candidate = set(candidate_tuple)
                if all(candidate & path_set for path_set in path_cells):
                    add_unique(candidate)
                    if len(hitting_sets) >= max_candidates:
                        return hitting_sets

    return hitting_sets if hitting_sets else [greedy_hs]


# ==============================================================================
# Utility and Leakage Computation
# ==============================================================================

def compute_leakage(
    mask: Set[Cell],
    paths: List[InferencePath],
    hyperedge_weights: Dict[FrozenSet[Cell], float]
) -> float:
    """
    Compute inferential leakage L(M, c_t).

    L(M, c_t) = max{w(π) : π ∈ Π_active}
    where Π_active are paths not blocked by mask M.
    """
    active_paths = [p for p in paths if not p.is_blocked_by(mask)]

    if not active_paths:
        return 0.0

    # Return maximum weight of active paths
    max_weight = max(p.compute_weight(hyperedge_weights) for p in active_paths)
    return max_weight


def compute_utility(
    mask: Set[Cell],
    target_cell: Cell,
    paths: List[InferencePath],
    hyperedge_weights: Dict[FrozenSet[Cell], float],
    alpha: float,
    beta: float
) -> float:
    """
    Compute mask utility u(M, c_t).

    u(M, c_t) = -α · L(M, c_t) - β · |M|

    where:
    - α > 0: leakage penalty weight
    - β > 0: deletion cost weight
    - L(M, c_t): inferential leakage
    - |M|: number of cells deleted
    """
    leakage = compute_leakage(mask, paths, hyperedge_weights)
    utility = -alpha * leakage - beta * len(mask)
    return utility


# ==============================================================================
# Exponential Mechanism
# ==============================================================================

def exponential_mechanism_sample(
    candidates: List[Set[Cell]],
    target_cell: Cell,
    paths: List[InferencePath],
    hyperedge_weights: Dict[FrozenSet[Cell], float],
    alpha: float,
    beta: float,
    epsilon: float
) -> Set[Cell]:
    """
    Sample a mask using the exponential mechanism.

    Probability ∝ exp(ε · u(M, c_t) / (2α))

    Args:
        candidates: List of candidate masks
        target_cell: Target cell
        paths: High-strength inference paths
        hyperedge_weights: Hyperedge weight mapping
        alpha: Leakage penalty weight
        beta: Deletion cost weight
        epsilon: Privacy budget

    Returns:
        Selected mask
    """
    if not candidates:
        return set()

    # Compute utilities for all candidates
    utilities = []
    for mask in candidates:
        u = compute_utility(mask, target_cell, paths, hyperedge_weights, alpha, beta)
        utilities.append(u)

    utilities = np.array(utilities)

    # Compute probabilities using exponential mechanism
    # p(M) ∝ exp(ε · u(M) / (2α))
    scores = epsilon * utilities / (2 * alpha)

    # Normalize to avoid overflow
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    probabilities = exp_scores / np.sum(exp_scores)

    # Sample according to probabilities
    selected_idx = np.random.choice(len(candidates), p=probabilities)
    return candidates[selected_idx]


# ==============================================================================
# Gumbel Trick Utilities
# ==============================================================================

def sample_gumbel(size=1):
    """
    Sample from standard Gumbel(0, 1) distribution.

    Using the inverse CDF method: G = -log(-log(U)) where U ~ Uniform(0, 1)

    Args:
        size: Number of samples (default: 1)

    Returns:
        Gumbel sample(s) - scalar if size=1, array otherwise
    """
    u = np.random.uniform(0, 1, size)
    result = -np.log(-np.log(u))
    return result.item() if size == 1 else result


def compute_marginal_gain(
    attribute_cell: Cell,
    current_mask: Set[Cell],
    paths: List[InferencePath],
    hyperedge_weights: Dict[FrozenSet[Cell], float],
    alpha: float,
    beta: float
) -> float:
    """
    Compute marginal gain Δu(A | M) for adding attribute A to mask M.

    Δu(A | M) = α · (L_curr - L_new) - β

    where:
    - L_curr = L(M, c_t): current leakage
    - L_new = L(M ∪ {A}, c_t): new leakage after adding A

    Args:
        attribute_cell: Cell to add (attribute A)
        current_mask: Current deletion mask M
        paths: High-strength inference paths
        hyperedge_weights: Hyperedge weight mapping
        alpha: Leakage penalty weight
        beta: Deletion cost weight

    Returns:
        Marginal gain value
    """
    # Compute current leakage L(M, c_t)
    L_curr = compute_leakage(current_mask, paths, hyperedge_weights)

    # Compute new leakage L(M ∪ {A}, c_t)
    new_mask = current_mask | {attribute_cell}
    L_new = compute_leakage(new_mask, paths, hyperedge_weights)

    # Δu(A | M) = α · (L_curr - L_new) - β
    marginal_gain = alpha * (L_curr - L_new) - beta

    return marginal_gain


# ==============================================================================
# Main Algorithm
# ==============================================================================

class DifferentialDeletion:
    """
    Differential Deletion Mechanisms.

    Implements both Algorithm 1 (Exponential Deletion) and Algorithm 2 (Gumbel Deletion)
    for privacy-preserving database sanitization.
    """

    def __init__(
        self,
        dataset: str,
        alpha: float = 1.0,
        beta: float = 0.5,
        epsilon: float = 1.0,
        tau: float = 0.1,
        hyperedge_weight_fn=None
    ):
        """
        Initialize the differential deletion mechanism.

        Args:
            dataset: Dataset name (e.g., 'adult', 'tax', 'hospital')
            alpha: Leakage penalty weight (α > 0)
            beta: Deletion cost weight (β > 0)
            epsilon: Privacy budget (ε > 0)
            tau: Path strength threshold (τ)
            hyperedge_weight_fn: Optional function to compute hyperedge weights
        """
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tau = tau
        self.hyperedge_weight_fn = hyperedge_weight_fn or self._default_weight_fn

        # Initialize hyperedge builder
        self.builder = HyperedgeBuilder(dataset=dataset)

    def _default_weight_fn(self, hyperedge: Hyperedge) -> float:
        """
        Default hyperedge weight function.

        For now, use uniform weights. Can be extended to use:
        - Constraint selectivity
        - Domain size ratios
        - Information-theoretic measures
        """
        return 1.0

    def _compute_hyperedge_weights(
        self,
        hyperedge_map: Dict[Cell, List[Hyperedge]]
    ) -> Dict[FrozenSet[Cell], float]:
        """Compute weights for all hyperedges."""
        weights = {}

        # Collect all unique hyperedges
        all_hyperedges = set()
        for hyperedges in hyperedge_map.values():
            for he in hyperedges:
                all_hyperedges.add(frozenset(he))

        # Compute weight for each hyperedge
        for he_frozen in all_hyperedges:
            he = Hyperedge(he_frozen)
            weights[he_frozen] = self.hyperedge_weight_fn(he)

        return weights

    def exponential_deletion(
        self,
        row: Dict[str, Any],
        key: Any,
        target_attr: str
    ) -> Set[Cell]:
        """
        Execute the exponential deletion mechanism (Algorithm 1).

        Args:
            row: Database row (tuple) as dictionary
            key: Row identifier
            target_attr: Target attribute to protect

        Returns:
            Deletion mask M (set of cells to delete)
        """
        # Step 1: Compute inference zone I(c_t) from hypergraph H
        # I(c_t) = all cells from the same tuple
        primary_table = self.builder.primary_table
        inference_zone = {
            Cell(Attribute(primary_table, attr), key, val)
            for attr, val in row.items()
        }

        # Create target cell
        target_cell = Cell(
            Attribute(primary_table, target_attr),
            key,
            row[target_attr]
        )

        # Step 2: Identify known attributes S ← {A : t[A] ≠ ⊥, A ≠ A_t}
        known_attrs = {
            attr for attr, val in row.items()
            if val is not None and attr != target_attr
        }

        # Build hypergraph
        hyperedge_map = self.builder.build_hyperedge_map(row, key, target_attr)
        root = build_hypergraph_tree(row, key, target_attr, hyperedge_map)

        # Compute hyperedge weights
        hyperedge_weights = self._compute_hyperedge_weights(hyperedge_map)

        # Step 3: Extract high-strength paths Π ← FindInferencePaths(S, A_t, H, τ)
        high_strength_paths = find_inference_paths(
            known_attrs,
            target_attr,
            root,
            hyperedge_weights,
            self.tau
        )

        print(f"Found {len(high_strength_paths)} high-strength paths (τ={self.tau})")

        # Step 4: Enumerate candidate hitting sets
        # Remove target cell from inference zone (can't delete target)
        candidate_cells = inference_zone - {target_cell}

        # Use greedy enumeration for efficiency
        candidates = enumerate_minimal_hitting_sets(
            high_strength_paths,
            candidate_cells
        )

        # Also add empty set as a candidate
        if set() not in candidates:
            candidates.insert(0, set())

        print(f"Generated {len(candidates)} candidate hitting sets")

        # Steps 5-8: For each candidate, compute utility
        # (This is done inside exponential_mechanism_sample)

        # Step 9: Sample M with probability ∝ exp(ε · u(M, c_t)/(2α))
        selected_mask = exponential_mechanism_sample(
            candidates,
            target_cell,
            high_strength_paths,
            hyperedge_weights,
            self.alpha,
            self.beta,
            self.epsilon
        )

        # Step 10: return M
        return selected_mask

    def gumbel_deletion(
        self,
        row: Dict[str, Any],
        key: Any,
        target_attr: str
    ) -> Set[Cell]:
        """
        Execute the Gumbel deletion mechanism (Algorithm 2).

        This variant uses the Gumbel trick for greedy sequential selection.

        Args:
            row: Database row (tuple) as dictionary
            key: Row identifier
            target_attr: Target attribute to protect

        Returns:
            Deletion mask M (set of cells to delete)
        """
        # Step 1: Compute inference zone I(c_t) from hypergraph H
        primary_table = self.builder.primary_table
        inference_zone = {
            Cell(Attribute(primary_table, attr), key, val)
            for attr, val in row.items()
        }

        # Create target cell
        target_cell = Cell(
            Attribute(primary_table, target_attr),
            key,
            row[target_attr]
        )

        # Step 2: Identify known attributes S ← {A : t[A] ≠ ⊥, A ≠ A_t}
        known_attrs = {
            attr for attr, val in row.items()
            if val is not None and attr != target_attr
        }

        # Build hypergraph
        hyperedge_map = self.builder.build_hyperedge_map(row, key, target_attr)
        root = build_hypergraph_tree(row, key, target_attr, hyperedge_map)

        # Compute hyperedge weights
        hyperedge_weights = self._compute_hyperedge_weights(hyperedge_map)

        # Step 3: Extract high-strength paths Π ← FindInferencePaths(S, A_t, H, τ)
        high_strength_paths = find_inference_paths(
            known_attrs,
            target_attr,
            root,
            hyperedge_weights,
            self.tau
        )

        print(f"Found {len(high_strength_paths)} high-strength paths (τ={self.tau})")

        # Step 4: Initialize M ← ∅
        mask = set()

        # Remove target cell from inference zone (can't delete target)
        candidate_cells = inference_zone - {target_cell}

        # Step 5: while ∃π ∈ Π that is active (not blocked by M) do
        iteration = 0
        while True:
            # Check if any path is still active
            active_paths = [p for p in high_strength_paths if not p.is_blocked_by(mask)]

            if not active_paths:
                print(f"All paths blocked after {iteration} iterations")
                break

            # Get available attributes: I(c_t) \ M
            available_cells = candidate_cells - mask

            if not available_cells:
                print(f"No more attributes to delete after {iteration} iterations")
                break

            iteration += 1
            print(f"\nIteration {iteration}: {len(active_paths)} active paths, {len(available_cells)} available attributes")

            # Step 6: for each attribute A ∈ I(c_t) \ M do
            scores = {}
            marginal_gains = {}

            for cell in available_cells:
                # Step 7-9: Compute marginal gain Δu(A | M)
                delta_u = compute_marginal_gain(
                    cell,
                    mask,
                    high_strength_paths,
                    hyperedge_weights,
                    self.alpha,
                    self.beta
                )
                marginal_gains[cell] = delta_u

                # Step 10: Sample Gumbel noise: g_A ~ Gumbel(0, 1)
                g_A = sample_gumbel()

                # Step 11: Compute score: s_A ← (ε/(2α)) · Δu(A | M) + g_A
                s_A = (self.epsilon / (2 * self.alpha)) * delta_u + g_A
                scores[cell] = s_A

            # Step 12: Select: A* ← arg max_A s_A
            best_cell = max(scores.items(), key=lambda x: x[1])[0]
            best_score = scores[best_cell]
            best_gain = marginal_gains[best_cell]

            print(f"  Selected: {best_cell.attribute.col} (score={best_score:.4f}, Δu={best_gain:.4f})")

            # Step 13: Update mask: M ← M ∪ {A*}
            mask.add(best_cell)

            # Step 14: Update active paths (implicit in next iteration's check)

        # Step 15: return M
        return mask


# ==============================================================================
# Testing and Validation
# ==============================================================================

def main():
    """Test both differential deletion mechanisms."""
    print("=" * 70)
    print("Differential Deletion Mechanisms Test")
    print("=" * 70)

    # Test parameters
    dataset = 'adult'
    key = 2
    target_attr = 'education'

    # Fetch row
    print(f"\nFetching row {key} from {dataset} dataset...")
    with RTFDatabaseManager(dataset) as db:
        row = db.fetch_row(key)

    print(f"Target: {target_attr} = '{row[target_attr]}'")
    print(f"Row attributes: {list(row.keys())}")

    # Initialize differential deletion
    dd = DifferentialDeletion(
        dataset=dataset,
        alpha=1.0,
        beta=0.5,
        epsilon=1.0,
        tau=0.1
    )

    # ============================================================================
    # Algorithm 1: Exponential Deletion
    # ============================================================================
    print(f"\n{'='*70}")
    print("Algorithm 1: Exponential Deletion Mechanism")
    print(f"{'='*70}")

    mask_exponential = dd.exponential_deletion(row, key, target_attr)

    # Display results
    print(f"\n{'='*70}")
    print("Algorithm 1 Results")
    print(f"{'='*70}")
    print(f"Deletion mask size: {len(mask_exponential)}")
    print(f"Cells to delete:")
    for cell in sorted(mask_exponential, key=lambda c: c.attribute.col):
        print(f"  - {cell}")

    if not mask_exponential:
        print("  (No cells deleted - empty mask)")

    # ============================================================================
    # Algorithm 2: Gumbel Deletion
    # ============================================================================
    print(f"\n{'='*70}")
    print("Algorithm 2: Gumbel Deletion Mechanism")
    print(f"{'='*70}")

    mask_gumbel = dd.gumbel_deletion(row, key, target_attr)

    # Display results
    print(f"\n{'='*70}")
    print("Algorithm 2 Results")
    print(f"{'='*70}")
    print(f"Deletion mask size: {len(mask_gumbel)}")
    print(f"Cells to delete:")
    for cell in sorted(mask_gumbel, key=lambda c: c.attribute.col):
        print(f"  - {cell}")

    if not mask_gumbel:
        print("  (No cells deleted - empty mask)")

    # ============================================================================
    # Comparison
    # ============================================================================
    print(f"\n{'='*70}")
    print("Comparison")
    print(f"{'='*70}")
    print(f"Exponential deletion mask size: {len(mask_exponential)}")
    print(f"Gumbel deletion mask size: {len(mask_gumbel)}")

    # Find overlap
    overlap = mask_exponential & mask_gumbel
    only_exponential = mask_exponential - mask_gumbel
    only_gumbel = mask_gumbel - mask_exponential

    print(f"\nOverlap: {len(overlap)} cells")
    if overlap:
        for cell in sorted(overlap, key=lambda c: c.attribute.col):
            print(f"  - {cell.attribute.col}")

    print(f"\nOnly in exponential: {len(only_exponential)} cells")
    if only_exponential:
        for cell in sorted(only_exponential, key=lambda c: c.attribute.col):
            print(f"  - {cell.attribute.col}")

    print(f"\nOnly in Gumbel: {len(only_gumbel)} cells")
    if only_gumbel:
        for cell in sorted(only_gumbel, key=lambda c: c.attribute.col):
            print(f"  - {cell.attribute.col}")

    print(f"\n{'='*70}")
    print("Test completed successfully!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
