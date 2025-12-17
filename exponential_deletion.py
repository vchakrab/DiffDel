#!/usr/bin/env python3
"""
Differential Deletion Mechanisms
==================================
... (imports and placeholder classes remain the same)
"""

import sys
import os
from typing import Any, Dict, List, Set, Tuple, FrozenSet
from collections import defaultdict, deque, Counter
from itertools import chain, combinations
import time
from sys import getsizeof
import numpy as np
from importlib import import_module
import mysql.connector
from mysql.connector import Error


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    print("Warning: 'config.py' not found. Database operations will fail.")
    config = None

# The following imports are assumed to be available in the execution environment
try:
    from cell import Attribute, Cell, Hyperedge
    from fetch_row import RTFDatabaseManager
    from InferenceGraph.bulid_hyperedges import HyperedgeBuilder
    from InferenceGraph.build_hypergraph import build_hypergraph_tree, GraphNode
except ImportError as e:
    # Define placeholder classes/functions for running/inspection purposes
    class Attribute:
        def __init__(self, table, col): self.table = table; self.col = col

        def __repr__(self): return f"Attr({self.col})"


    class Cell:
        def __init__(self, attribute, key,
                     val): self.attribute = attribute; self.key = key; self.val = val

        def __repr__(self): return f"Cell({self.attribute.col})"

        def __hash__(self): return hash((self.attribute.col, self.key, self.val))

        def __eq__(self, other): return isinstance(other, Cell) and (
            self.attribute.col, self.key, self.val) == (other.attribute.col, other.key, other.val)


    class Hyperedge:
        def __init__(self, cells): self.cells = cells

        def __iter__(self): return iter(self.cells)

        def __repr__(self): return f"HE({[c.attribute.col for c in self.cells]})"

        @property
        def cell_names(self): return tuple(c.attribute.col for c in self.cells)


    class RTFDatabaseManager:
        def __init__(self, dataset): self.dataset = dataset

        def __enter__(self): return self

        def __exit__(self, exc_type, exc_val, exc_tb): pass

        def fetch_row(self, key):
            if key == 2 and self.dataset == 'adult':
                return {'age': 39, 'workclass': 'State-gov', 'fnlwgt': 77516,
                        'education': 'Bachelors', 'education-num': 13}
            return {}


    class HyperedgeBuilder:
        def __init__(self, dataset): self.primary_table = dataset

        def build_hyperedge_map(self, row, key, target_attr): return {}


    class GraphNode:
        def __init__(self, cell): self.cell = cell; self.branches = []

        def __repr__(self): return f"Node({self.cell})"


    def build_hypergraph_tree(row, key, target_attr, hyperedge_map):
        return GraphNode(Cell(Attribute('adult', target_attr), key, row.get(target_attr, '')))


# ==============================================================================
# Path Representation and Utilities (UNCHANGED)
# ==============================================================================

class InferencePath:
    # ... (InferencePath class remains unchanged)
    """Represents a path in the inference graph as a sequence of hyperedges."""

    def __init__(self, hyperedges: List[Hyperedge], cells: List[Cell]):
        self.hyperedges = hyperedges
        self.cells = cells
        self._weight = None
        self._cell_set = frozenset(cells)

    def compute_weight(self, hyperedge_weights: Dict[FrozenSet[Cell], float]) -> float:
        """Compute path weight as product of hyperedge weights."""
        if self._weight is None:
            self._weight = 1.0
            for he in self.hyperedges:
                he_key = frozenset(he.cells if hasattr(he, 'cells') else he)
                weight = hyperedge_weights.get(he_key, 1.0)
                self._weight *= weight
        return self._weight

    def is_blocked_by(self, mask: Set[Cell]) -> bool:
        """Check if this path is blocked by the deletion mask."""
        return bool(self._cell_set & mask)

    def __repr__(self):
        try:
            weight_repr = f"{self._weight:.4f}" if self._weight is not None else "uncomputed"
        except Exception:
            weight_repr = "error"

        cell_seq = " -> ".join([f"{c.attribute.col}" for c in self.cells])
        return f"Path({cell_seq}, weight={weight_repr})"


# ==============================================================================
# Path Extraction (Original file logic, using GraphNode) (UNCHANGED)
# ==============================================================================
# ... (extract_all_paths and find_inference_paths remain unchanged)


# ==============================================================================
# Hitting Set Enumeration (UNCHANGED - Contains the generic powerset utility)
# ==============================================================================
# ... (powerset, enumerate_hitting_sets, enumerate_minimal_hitting_sets remain unchanged)


# ==============================================================================
# Utility and Leakage Computation (CELL-OBJECT BASED - UNCHANGED)
# ==============================================================================

def filter_active_paths(
        # ... (filter_active_paths remains unchanged)
        mask: Set[Cell],
        paths: List[InferencePath]
) -> List[InferencePath]:
    """
    Filter the list of high-strength paths to return only those NOT blocked by the mask M.
    """
    active_paths = [p for p in paths if not p.is_blocked_by(mask)]
    return active_paths


def compute_max_leakage(
        # ... (compute_max_leakage remains unchanged)
        active_paths: List[InferencePath],
        hyperedge_weights: Dict[FrozenSet[Cell], float]
) -> float:
    """
    Compute the inferential leakage L(M, c_t) as the maximum weight among active paths.
    """
    if not active_paths:
        return 0.0

    max_weight = max(p.compute_weight(hyperedge_weights) for p in active_paths)
    return max_weight


def compute_leakage(
        # ... (compute_leakage remains unchanged)
        mask: Set[Cell],
        paths: List[InferencePath],
        hyperedge_weights: Dict[FrozenSet[Cell], float]
) -> float:
    """
    Compute inferential leakage L(M, c_t) by chaining the two new methods.
    """
    active_paths = filter_active_paths(mask, paths)
    leakage = compute_max_leakage(active_paths, hyperedge_weights)
    return leakage


def get_path_inference_zone(paths: List[InferencePath], target_cell: Cell) -> Set[Cell]:
    # ... (get_path_inference_zone remains unchanged)
    """
    Calculates the union of all unique cells contained within the given list of
    inference paths, excluding the target cell itself.
    """
    all_cells_in_paths = set(
        cell
        for path in paths
        for cell in path.cells
    )

    path_zone = all_cells_in_paths - {target_cell}

    return path_zone


def compute_utility(
        # ... (compute_utility remains unchanged)
        mask: Set[Cell],
        target_cell: Cell,
        paths: List[InferencePath],
        hyperedge_weights: Dict[FrozenSet[Cell], float],
        alpha: float,
        beta: float
) -> float:
    """
    Compute mask utility u(M, c_t).
    """
    leakage = compute_leakage(mask, paths, hyperedge_weights)
    utility = -alpha * leakage - beta * len(mask)
    return utility


# ==============================================================================
# Exponential Mechanism (CELL-OBJECT BASED - UNCHANGED)
# ==============================================================================

def exponential_mechanism_sample(
        # ... (exponential_mechanism_sample remains unchanged)
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
    scores = epsilon * utilities / (2 * alpha)

    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    probabilities = exp_scores / np.sum(exp_scores)

    # Sample according to probabilities
    selected_idx = np.random.choice(len(candidates), p = probabilities)
    return candidates[selected_idx]


# ==============================================================================
# Gumbel Trick Utilities (UNCHANGED)
# ==============================================================================
# ... (sample_gumbel and compute_marginal_gain remain unchanged)


# ==============================================================================
# Main Algorithm (UNCHANGED)
# ==============================================================================

class DifferentialDeletion:
    # ... (DifferentialDeletion class and its methods remain unchanged)

    def __init__(
            self,
            dataset: str,
            alpha: float = 1.0,
            beta: float = 0.5,
            epsilon: float = 1.0,
            tau: float = 0.1,
            hyperedge_weight_fn = None
    ):
        """
        Initialize the differential deletion mechanism.
        """
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tau = tau
        self.hyperedge_weight_fn = hyperedge_weight_fn or self._default_weight_fn

        self.builder = HyperedgeBuilder(dataset = dataset)

    def _default_weight_fn(self, hyperedge: Hyperedge) -> float:
        """
        Default hyperedge weight function (uniform 1.0).
        """
        return 1.0

    def _compute_hyperedge_weights(
            self,
            hyperedge_map: Dict[Cell, List[Hyperedge]]
    ) -> Dict[FrozenSet[Cell], float]:
        """Compute weights for all hyperedges."""
        weights = {}

        all_hyperedges = set()
        for hyperedges in hyperedge_map.values():
            for he in hyperedges:
                he_cells = he.cells if hasattr(he, 'cells') else he
                all_hyperedges.add(frozenset(he_cells))

        for he_frozen in all_hyperedges:
            he = Hyperedge(list(he_frozen))
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
        """
        primary_table = self.builder.primary_table
        inference_zone = {
            Cell(Attribute(primary_table, attr), key, val)
            for attr, val in row.items()
        }

        target_cell = Cell(
            Attribute(primary_table, target_attr),
            key,
            row.get(target_attr)
        )

        known_attrs = {
            attr for attr, val in row.items()
            if val is not None and attr != target_attr
        }

        hyperedge_map = self.builder.build_hyperedge_map(row, key, target_attr)
        root = build_hypergraph_tree(row, key, target_attr, hyperedge_map)

        hyperedge_weights = self._compute_hyperedge_weights(hyperedge_map)

        high_strength_paths = find_inference_paths(
            known_attrs,
            target_attr,
            root,
            hyperedge_weights,
            self.tau
        )

        print(f"Found {len(high_strength_paths)} high-strength paths (τ={self.tau})")

        candidate_cells = inference_zone - {target_cell}

        candidates = enumerate_minimal_hitting_sets(
            high_strength_paths,
            candidate_cells
        )

        if set() not in candidates:
            candidates.insert(0, set())

        print(f"Generated {len(candidates)} candidate hitting sets")

        selected_mask = exponential_mechanism_sample(
            candidates,
            target_cell,
            high_strength_paths,
            hyperedge_weights,
            self.alpha,
            self.beta,
            self.epsilon
        )

        return selected_mask

    def gumbel_deletion(
            self,
            row: Dict[str, Any],
            key: Any,
            target_attr: str
    ) -> Set[Cell]:
        """
        Execute the Gumbel deletion mechanism (Algorithm 2).
        """
        primary_table = self.builder.primary_table
        inference_zone = {
            Cell(Attribute(primary_table, attr), key, val)
            for attr, val in row.items()
        }

        target_cell = Cell(
            Attribute(primary_table, target_attr),
            key,
            row.get(target_attr)
        )

        known_attrs = {
            attr for attr, val in row.items()
            if val is not None and attr != target_attr
        }

        hyperedge_map = self.builder.build_hyperedge_map(row, key, target_attr)
        root = build_hypergraph_tree(row, key, target_attr, hyperedge_map)

        hyperedge_weights = self._compute_hyperedge_weights(hyperedge_map)

        high_strength_paths = find_inference_paths(
            known_attrs,
            target_attr,
            root,
            hyperedge_weights,
            self.tau
        )

        print(f"Found {len(high_strength_paths)} high-strength paths (τ={self.tau})")

        mask = set()

        candidate_cells = inference_zone - {target_cell}

        iteration = 0
        while True:
            active_paths = filter_active_paths(mask, high_strength_paths)

            if not active_paths:
                print(f"All paths blocked after {iteration} iterations")
                break

            available_cells = candidate_cells - mask

            if not available_cells:
                print(f"No more attributes to delete after {iteration} iterations")
                break

            iteration += 1
            print(
                f"\nIteration {iteration}: {len(active_paths)} active paths, {len(available_cells)} available attributes")

            scores = {}
            marginal_gains = {}

            for cell in available_cells:
                delta_u = compute_marginal_gain(
                    cell,
                    mask,
                    high_strength_paths,
                    hyperedge_weights,
                    self.alpha,
                    self.beta
                )
                marginal_gains[cell] = delta_u

                g_A = sample_gumbel()

                s_A = (self.epsilon / (2 * self.alpha)) * delta_u + g_A
                scores[cell] = s_A

            best_cell = max(scores.items(), key = lambda x: x[1])[0]
            best_score = scores[best_cell]
            best_gain = marginal_gains[best_cell]

            print(
                f"  Selected: {best_cell.attribute.col} (score={best_score:.4f}, Δu={best_gain:.4f})")

            mask.add(best_cell)

        return mask


# ==============================================================================
# Testing and Validation (UNCHANGED)
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
    np.random.seed(42)

    # Fetch row (using placeholder if imports failed)
    print(f"\nFetching row {key} from {dataset} dataset...")
    db = RTFDatabaseManager(dataset)
    row = db.fetch_row(key)

    print(f"Target: {target_attr} = '{row.get(target_attr, 'N/A')}'")
    print(f"Row attributes: {list(row.keys())}")

    # Initialize differential deletion
    dd = DifferentialDeletion(
        dataset = dataset,
        alpha = 1.0,
        beta = 0.5,
        epsilon = 1.0,
        tau = 0.1
    )

    # --- Run Algorithm 1 ---
    print(f"\n{'=' * 70}")
    print("Algorithm 1: Exponential Deletion Mechanism")
    print(f"{'=' * 70}")
    mask_exponential = dd.exponential_deletion(row, key, target_attr)
    print(f"\nAlgorithm 1 Results (size: {len(mask_exponential)})")

    # --- Run Algorithm 2 ---
    print(f"\n{'=' * 70}")
    print("Algorithm 2: Gumbel Deletion Mechanism")
    print(f"{'=' * 70}")
    mask_gumbel = dd.gumbel_deletion(row, key, target_attr)
    print(f"\nAlgorithm 2 Results (size: {len(mask_gumbel)})")

    print(f"\n{'=' * 70}")
    print("Test completed successfully!")
    print(f"{'=' * 70}")


#
# if __name__ == '__main__':
#
#     # --- Raw String Function Test for Utility and Sampling ---
#     print("\n" + "=" * 70)
#     print("Testing Raw String Functions: Utility and Exponential Sampling")
#     print("=" * 70)
#
#     # Setup for string-based test (mimics a simple hypergraph)
#     test_hyperedges = [
#         ('a', 'b', 'c'),  # Edge 0: {a, b} -> c, weight=0.8
#         ('c', 'd', 'e')  # Edge 1: {c, d} -> e, weight=0.5
#     ]
#     test_target = 'c'
#     test_known = {'a', 'b', 'd', 'e'}
#     test_weights = {0: 0.8, 1: 0.5}  # edge_idx: weight
#     test_paths = [[0]]  # Only one path: {a, b} --(0)--> c
#     test_alpha = 1.0
#     test_beta = 0.5
#     test_epsilon = 1.0
#
#     # 1. Compute Inference Zone
#     test_zone = get_path_inference_zone_str(test_paths, test_hyperedges, test_target)
#     # Expected zone: {'a', 'b'} (cells in path 0 excluding target c)
#     print(f"Inference Zone: {test_zone}")
#
#     # 2. Compute all possible masks
#     candidates = compute_possible_mask_set_str(test_zone)
#     print(f"Candidates: {candidates}")  # Expected: [{}, {'a'}, {'b'}, {'a', 'b'}]
#
#     # 3. Compute utilities
#     utilities = []
#     print("\nUtility Calculation (u(M, c)):")
#     for mask in candidates:
#         u = compute_utility_str(
#             mask, test_target, test_paths, test_hyperedges, test_known,
#             test_weights, test_alpha, test_beta
#         )
#         utilities.append(u)
#
#         # Calculate expected leakage for verification
#         # L(M, c) = 0.0 if blocked, 1 - (1 - 0.8) = 0.8 if active
#         if 'a' in mask or 'b' in mask:
#             # Mask {'a'}, {'b'}, or {'a', 'b'} blocks path 0
#             expected_leakage = 0.0
#         else:
#             # Mask {} leaves path 0 active
#             expected_leakage = 0.8
#
#         # Expected Utility: -alpha * L - beta * |M|
#         expected_utility = -test_alpha * expected_leakage - test_beta * len(mask)
#
#         print(
#             f"  Mask {mask}: L={expected_leakage:.4f}, |M|={len(mask)}, u={u:.4f} (Expected: {expected_utility:.4f})")
#         assert np.isclose(u,
#                           expected_utility), f"Mismatch for mask {mask}: Got {u}, Expected {expected_utility}"
#
#     # 4. Sample a mask using the Exponential Mechanism
#     selected_mask = exponential_mechanism_sample_str(
#         candidates, test_target, test_paths, test_hyperedges, test_known,
#         test_weights, test_alpha, test_beta, test_epsilon
#     )
#
#     print(f"\nExponential Mechanism Sampled Mask: {selected_mask}")
#
#     print("Raw String Function Test successful.")
#     print("=" * 70)
#
#     main()

# -------- STR HyperEdges Based Exponential Deletion ----------
# _____________________________________________________________

def powerset(iterable):
    """
    Computes the powerset of an iterable.
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_inference_paths_str(hyperedges: List[Tuple[str, ...]],
                             target_cell: str,
                             initial_known: Set[str] = None) -> List[List[int]]:
    # ... (find_inference_paths_str remains unchanged)
    if initial_known is None:
        raise ValueError("initial_known must be provided")

    all_paths = []
    seen_paths = set()

    def dfs(known_cells: Set[str], used_edges: Set[int], current_path: List[int],
            can_assume_known: Set[str]):

        for edge_idx, edge in enumerate(hyperedges):
            if edge_idx in used_edges:
                continue

            for inferred_cell in edge:
                if inferred_cell in known_cells:
                    continue

                other_cells = [c for c in edge if c != inferred_cell]

                if all(c in known_cells for c in other_cells):
                    new_known = known_cells | {inferred_cell}
                    new_used = used_edges | {edge_idx}
                    new_path = current_path + [edge_idx]

                    if inferred_cell == target_cell:
                        path_tuple = tuple(new_path)
                        if path_tuple not in seen_paths:
                            seen_paths.add(path_tuple)
                            all_paths.append(new_path)

                    dfs(new_known, new_used, new_path, can_assume_known)

                elif inferred_cell == target_cell:
                    unknown_cells = [c for c in other_cells if c not in known_cells]
                    if all(c in can_assume_known for c in unknown_cells):
                        new_path = current_path + [edge_idx]
                        path_tuple = tuple(new_path)
                        if path_tuple not in seen_paths:
                            seen_paths.add(path_tuple)
                            all_paths.append(new_path)

    all_cells = set()
    for edge in hyperedges:
        all_cells.update(edge)
    potentially_inferrable = all_cells - initial_known - {target_cell}

    dfs(initial_known, set(), [], potentially_inferrable)
    return all_paths


def filter_active_paths_str(hyperedges: List[Tuple[str, ...]],
                            paths: List[List[int]],
                            mask: Set[str],
                            initial_known: Set[str]) -> List[List[int]]:
    # ... (filter_active_paths_str remains unchanged)
    active_paths = []

    for path in paths:
        is_blocked = False
        known_so_far = initial_known - mask

        for edge_idx in path:
            edge = hyperedges[edge_idx]

            unknown_in_edge = [c for c in edge if c not in known_so_far]

            if len(unknown_in_edge) == 0:
                continue
            elif len(unknown_in_edge) == 1:
                inferred_cell = unknown_in_edge[0]
                known_so_far.add(inferred_cell)
            else:
                is_blocked = True
                break

        if not is_blocked:
            active_paths.append(path)

    return active_paths


def compute_product_leakage_str(active_paths: List[List[int]],
                                hyperedges: List[Tuple[str, ...]],
                                edge_weights: Dict[int, float]) -> float:
    # ... (compute_product_leakage_str remains unchanged)
    if edge_weights is None:
        edge_weights = {i: 1.0 for i in range(len(hyperedges))}

    if len(active_paths) == 0:
        return 0.0

    product = 1.0
    for path in active_paths:
        path_weight = 1.0
        for edge_idx in path:
            path_weight *= edge_weights[edge_idx]

        product *= (1 - path_weight)

    leakage = 1 - product
    return leakage


def calculate_leakage_str(hyperedges: List[Tuple[str, ...]],
                          paths: List[List[int]],
                          mask: Set[str],
                          target_cell: str,
                          initial_known: Set[str],
                          edge_weights: Dict[int, float] = None) -> float:
    # ... (calculate_leakage_str remains unchanged)
    active_paths = filter_active_paths_str(
        hyperedges,
        paths,
        mask,
        initial_known
    )

    leakage = compute_product_leakage_str(
        active_paths,
        hyperedges,
        edge_weights
    )

    return leakage


def get_path_inference_zone_str(paths: List[List[int]], hyperedges: List[Tuple[str, ...]],
                                target_cell: str) -> Set[str]:
    # ... (get_path_inference_zone_str remains unchanged)
    all_cells_in_paths = set()

    for path in paths:
        for edge_idx in path:
            edge = hyperedges[edge_idx]
            all_cells_in_paths.update(edge)

    path_zone = all_cells_in_paths - {target_cell}

    return path_zone


def compute_possible_mask_set_str(inference_zone_str: Set[str]) -> List[Set[str]]:
    # ... (compute_possible_mask_set_str remains unchanged)
    all_subsets = [set(subset) for subset in powerset(inference_zone_str)]
    return all_subsets


def compute_utility_str(
        mask: Set[str],
        target_cell: str,
        paths: List[List[int]],
        hyperedges: List[Tuple[str, ...]],
        initial_known: Set[str],
        edge_weights: Dict[int, float],
        alpha: float,
        beta: float
) -> float:
    """
    Compute mask utility u(M, c_t) for the string-based approach.

    u(M, c_t) = -α · L(M, c_t) - β · |M|
    """
    leakage = calculate_leakage_str(
        hyperedges,
        paths,
        mask,
        target_cell,
        initial_known,
        edge_weights
    )
    # The term |M| is simply the size of the deletion mask (set of strings)
    utility = -alpha * leakage - beta * len(mask)
    return utility


def exponential_mechanism_sample_str(
        candidates: List[Set[str]],
        target_cell: str,
        paths: List[List[int]],
        hyperedges: List[Tuple[str, ...]],
        initial_known: Set[str],
        edge_weights: Dict[int, float],
        alpha: float,
        beta: float,
        epsilon: float
) -> Set[str]:
    """
    Sample a mask using the exponential mechanism for the string-based approach.

    Probability ∝ exp(ε · u(M, c_t) / (2α))
    """
    if not candidates:
        return set()

    # 1. Compute utilities for all candidates
    utilities = []
    for mask in candidates:
        u = compute_utility_str(
            mask,
            target_cell,
            paths,
            hyperedges,
            initial_known,
            edge_weights,
            alpha,
            beta
        )
        utilities.append(u)

    utilities = np.array(utilities)

    # 2. Compute probabilities using the exponential mechanism
    # Scores = ε * u(M, c_t) / (2α)
    scores = epsilon * utilities / (2 * alpha)

    # Use log-sum-exp trick for numerical stability
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    probabilities = exp_scores / np.sum(exp_scores)

    # 3. Sample according to probabilities
    selected_idx = np.random.choice(len(candidates), p = probabilities)
    return candidates[selected_idx]


def measure_memory_overhead_str(hyperedges, paths, candidate_masks):
    """
    Calculates the approximate memory overhead of the key data structures.
    This is a Python-specific way to estimate memory, summing the size of the
    main lists and their contained elements.
    """
    memory = getsizeof(hyperedges)
    for he in hyperedges:
        memory += getsizeof(he)
        for attr in he:
            memory += getsizeof(attr)

    memory += getsizeof(paths)
    for path in paths:
        memory += getsizeof(path)

    memory += getsizeof(candidate_masks)
    for mask in candidate_masks:
        memory += getsizeof(mask)
        for attr in mask:
            memory += getsizeof(attr)
    return memory


def clean_raw_dcs(raw_dcs: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, ...]]:
    """
    Cleans raw denial constraints into simple hyperedges (tuples of attribute strings).
    Example: [('t1.education', '!=', 't2.education'), ('t1.education_num', '==', 't2.education_num')]
             --> ('education', 'education_num')
    """
    cleaned_hyperedges = []
    for dc in raw_dcs:
        attributes = set()
        for pred in dc:
            # pred is like ('t1.education', '!=', 't2.education')
            # or ('t1.age', '==', '30')
            for item in [pred[0], pred[2]]:
                if '.' in item:  # It's an attribute like 't1.education'
                    attr_name = item.split('.')[-1]
                    attributes.add(attr_name)
        if attributes:
            cleaned_hyperedges.append(tuple(sorted(list(attributes))))
    return cleaned_hyperedges


def exponential_deletion_main(dataset: str, key: int, target_cell: str):
    """
    Main orchestrator for the string-based exponential deletion mechanism.
    Captures and returns a dictionary of performance and result metrics.
    Deletions are performed on the '{dataset}_copy_data' table.
    """
    print("\n" + "=" * 70)
    print(
        f"Running Main Orchestrator for Dataset: '{dataset}', Key: {key}, Target: '{target_cell}'")
    print("=" * 70)

    # --- Initialization Phase ---
    init_start = time.time()
    try:
            # Determine the correct casing for the dataset module name
        if dataset == 'ncvoter':
            dataset_module_name = 'NCVoter'
        else:
            dataset_module_name = dataset.capitalize()
    
        dc_module_path = f'DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed'
        dc_module = __import__(dc_module_path, fromlist=['denial_constraints'])
        raw_dcs = dc_module.denial_constraints
        print(f"Successfully loaded {len(raw_dcs)} raw denial constraints for '{dataset}'.")
    except ImportError:
        print(f"Error: Could not find or load denial constraints for dataset '{dataset}'.")
        print(f"Attempted to load module: {dc_module_path}") # Added for better debugging
        return None
    hyperedges = clean_raw_dcs(raw_dcs)
    print("Cleaned DCs into simple hyperedges.")
    init_time = time.time() - init_start

    # --- Modeling Phase ---
    model_start = time.time()
    all_attributes = set(attr for he in hyperedges for attr in he)
    attribute_counts = Counter(attr for he in hyperedges for attr in he)
    intersecting_cells = {attr for attr, count in attribute_counts.items() if count > 1}
    initial_known = all_attributes - {target_cell} - intersecting_cells

    # --- DEBUGGING FOR NCVOTER ---
    if dataset == 'ncvoter':
        print("\n--- NCVOTER DEBUG ---")
        print(f"Target Cell: {target_cell}")
        print(f"All Attributes ({len(all_attributes)}): {all_attributes}")
        print(f"Intersecting Cells ({len(intersecting_cells)}): {intersecting_cells}")
        print(f"Final Initial Known Set: {initial_known}")
        print("---------------------\n")
    # --- END DEBUGGING ---

    paths = find_inference_paths_str(hyperedges, target_cell, initial_known)
    num_paths = len(paths)

    if dataset == 'ncvoter':
        print(f"\n--- NCVOTER DEBUG ---")
        print(f"Number of paths found: {num_paths}")
        print("---------------------\n")

    inference_zone = get_path_inference_zone_str(paths, hyperedges, target_cell)
    candidate_masks = compute_possible_mask_set_str(inference_zone)

    memory_overhead = measure_memory_overhead_str(hyperedges, paths, candidate_masks)
    
    # --- Execution/Sampling (Part of Modeling Time) ---
    edge_weights = {i: 0.8 for i in range(len(hyperedges))}
    alpha = 1.0
    beta = 0.5
    epsilon = 1.0

    final_mask = exponential_mechanism_sample_str(
        candidate_masks, target_cell, paths, hyperedges, initial_known,
        edge_weights, alpha, beta, epsilon
    )
    mask_size = len(final_mask)
    model_time = time.time() - model_start

    # --- Metrics Calculation for Chosen Mask ---
    leakage = calculate_leakage_str(
        hyperedges, paths, final_mask, target_cell, initial_known, edge_weights
    )
    utility = compute_utility_str(
        final_mask, target_cell, paths, hyperedges, initial_known,
        edge_weights, alpha, beta
    )

    # --- Database Update Phase ---
    del_start = time.time()
    if config:
        cells_to_delete = final_mask | {target_cell}
        conn = None
        cursor = None
        try:
            db_details = config.get_database_config(dataset)
            primary_table = f"{dataset}_copy_data"
            conn = mysql.connector.connect(
                host = db_details['host'], user = db_details['user'],
                password = db_details['password'],
                database = db_details['database'],
                ssl_disabled = db_details.get('ssl_disabled', True)
            )
            if conn.is_connected():
                cursor = conn.cursor()
                DELETION_QUERY = "UPDATE {table_name} SET `{column_name}` = NULL WHERE id = {key};"
                for cell_to_delete in cells_to_delete:
                    query = DELETION_QUERY.format(
                        table_name = primary_table, column_name = cell_to_delete, key = key
                    )
                    cursor.execute(query)
                conn.commit()
        except Error as e:
            print(f"Error during database update: {e}")
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
    del_time = time.time() - del_start

    # --- Compile and Return Results ---
    results = {
        'init_time': init_time,
        'model_time': model_time,
        'del_time': del_time,
        'leakage': leakage,
        'utility': utility,
        'mask_size': mask_size,
        'final_mask': final_mask,
        'num_paths': num_paths,
        'memory_overhead_bytes': memory_overhead,
        'num_instantiated_cells': len(inference_zone)
    }

    print("\n" + "=" * 70)
    print("Main Orchestrator Finished. Results:")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  - {key}: {val:.4f}")
        else:
            print(f"  - {key}: {val}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    # Run the main orchestrator for a single test case
    results = exponential_deletion_main(
        dataset='adult',
        key=2,
        target_cell='education'
    )


