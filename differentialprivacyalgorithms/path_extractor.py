from typing import List, Set, Tuple, Dict


def find_inference_paths(hyperedges: List[Tuple[str, ...]],
                         target_cell: str,
                         initial_known: Set[str] = None) -> List[List[int]]:
    """
    Find all valid inference paths to a target cell in a dependency hypergraph.

    A hyperedge allows inference of any one cell if all other cells in that
    hyperedge are known. A path is a sequence of hyperedges where each step
    infers exactly one new cell using previously known/inferred cells.

    This function explores all possible ways to reach the target, including
    paths that assume different intermediate cells might be known.

    Args:
        hyperedges: List of hyperedges, each is a tuple of cell names
                   Example: [('a','b','c','d'), ('d','e','f')]
        target_cell: The cell we want to infer (e.g., 'c')
        initial_known: Set of initially known cells. Must be provided.

    Returns:
        List of paths, where each path is a list of hyperedge indices
        Example: [[0], [1, 0]] means two paths - one using edge 0,
                 another using edges 1 then 0
    """
    if initial_known is None:
        raise ValueError("initial_known must be provided")

    all_paths = []
    seen_paths = set()  # To avoid duplicate paths

    def dfs(known_cells: Set[str], used_edges: Set[int], current_path: List[int],
            can_assume_known: Set[str]):
        """
        Depth-first search to find all paths.

        Args:
            known_cells: Set of currently known/inferred cells
            used_edges: Set of hyperedge indices already used in this path
            current_path: Current sequence of hyperedge indices
            can_assume_known: Cells that could potentially be assumed known
        """
        # Try to infer any cell (including target) and continue searching
        for edge_idx, edge in enumerate(hyperedges):
            if edge_idx in used_edges:
                continue

            # For each cell in this edge, check if it can be inferred
            for inferred_cell in edge:
                if inferred_cell in known_cells:
                    continue

                # Check if all other cells are known or can be assumed known
                other_cells = [c for c in edge if c != inferred_cell]

                # Case 1: All other cells are actually known right now
                if all(c in known_cells for c in other_cells):
                    new_known = known_cells | {inferred_cell}
                    new_used = used_edges | {edge_idx}
                    new_path = current_path + [edge_idx]

                    # If we inferred the target, record this path
                    if inferred_cell == target_cell:
                        path_tuple = tuple(new_path)
                        if path_tuple not in seen_paths:
                            seen_paths.add(path_tuple)
                            all_paths.append(new_path)

                    # Continue exploring from this state
                    dfs(new_known, new_used, new_path, can_assume_known)

                # Case 2: Some cells need to be assumed known (for shorter paths)
                elif inferred_cell == target_cell:
                    unknown_cells = [c for c in other_cells if c not in known_cells]
                    if all(c in can_assume_known for c in unknown_cells):
                        # We can create a path by assuming these cells are known
                        new_path = current_path + [edge_idx]
                        path_tuple = tuple(new_path)
                        if path_tuple not in seen_paths:
                            seen_paths.add(path_tuple)
                            all_paths.append(new_path)

    # Identify cells that could be inferred (appear in edges but aren't initially known)
    all_cells = set()
    for edge in hyperedges:
        all_cells.update(edge)
    potentially_inferrable = all_cells - initial_known - {target_cell}

    dfs(initial_known, set(), [], potentially_inferrable)
    return all_paths


def calculate_leakage(hyperedges: List[Tuple[str, ...]],
                      paths: List[List[int]],
                      mask: Set[str],
                      target_cell: str,
                      initial_known: Set[str],
                      edge_weights: Dict[int, float] = None) -> float:
    """
    Calculate inferential leakage given a mask and inference paths.

    Key insight: A hyperedge is blocked only if 2+ of its cells are masked/unknown.
    We can infer 1 masked cell if all other cells in the hyperedge are available.

    Args:
        hyperedges: List of hyperedges, each is a tuple of cell names
        paths: List of inference paths (each path is a list of edge indices)
        mask: Set of cells to mask (delete from knowledge)
        target_cell: The target cell being protected
        initial_known: Set of initially known cells (before masking)
        edge_weights: Dictionary mapping edge index to weight (probability of
                     successful inference). If None, assumes all weights are 1.0.

    Returns:
        Leakage value between 0 and 1
    """
    if edge_weights is None:
        edge_weights = {i: 1.0 for i in range(len(hyperedges))}

    # Determine which paths are blocked by the mask
    active_paths = []

    for path in paths:
        is_blocked = False
        known_so_far = initial_known - mask  # Remove masked cells from known

        # Check each edge in the path
        for edge_idx in path:
            edge = hyperedges[edge_idx]

            # For this edge, check if we can make progress
            # We can infer a cell if all other cells in the edge are available (known or can be inferred)

            # Separate cells into categories
            known_in_edge = [c for c in edge if c in known_so_far]
            unknown_in_edge = [c for c in edge if c not in known_so_far]

            # Can we infer something from this edge?
            if len(unknown_in_edge) == 0:
                # All cells already known, nothing to infer
                continue
            elif len(unknown_in_edge) == 1:
                # Exactly 1 unknown - we can infer it (even if it's masked)
                inferred_cell = unknown_in_edge[0]
                known_so_far.add(inferred_cell)
            else:
                # 2+ unknown cells - cannot infer anything, path is blocked
                is_blocked = True
                break

        if not is_blocked:
            active_paths.append(path)

    # Calculate leakage using formula: L = 1 - ∏(1 - w(π)) for active paths
    if len(active_paths) == 0:
        return 0.0

    product = 1.0
    for path in active_paths:
        # Calculate path weight: w(π) = ∏ w(e) for e in π
        path_weight = 1.0
        for edge_idx in path:
            path_weight *= edge_weights[edge_idx]

        # Multiply (1 - w(π))
        product *= (1 - path_weight)

    leakage = 1 - product
    return leakage


# Example usage
if __name__ == "__main__":
    # Example: edges are 'abcd' and 'def', target is 'c'
    edges = [
        ('a', 'b', 'c', 'd'),
        ('d', 'e', 'f')
    ]
    target = 'c'

    # Edge weights
    weights = {0: 0.7, 1: 0.5}

    # Find all cells and identify which appear in multiple edges (intersecting cells)
    from collections import Counter


    cell_counts = Counter()
    all_cells = set()
    for edge in edges:
        all_cells.update(edge)
        for cell in edge:
            cell_counts[cell] += 1

    # Intersecting cells are those that appear in more than one edge
    intersecting_cells = {cell for cell, count in cell_counts.items() if count > 1}

    # Initially known: all cells except target and intersecting cells
    known = all_cells - {target} - intersecting_cells

    print("Example: Finding paths to infer cell 'c'")
    print(f"Hyperedges: {edges}")
    print(f"Target cell: {target}")
    print(f"Edge weights: {weights}")
    print(f"Intersecting cells: {intersecting_cells}")
    print(f"Initially known: {known}\n")

    paths = find_inference_paths(edges, target, known)
    print(f"Found {len(paths)} path(s):")
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: edges {path}")

    # Test leakage calculation with different masks
    print("\n--- Leakage Calculation ---")
    print("For leakage, we assume adversary knows ALL cells except target")

    # For leakage calculation, initial_known = all cells except target
    leakage_initial_known = all_cells - {target}
    print(f"Initial known for leakage: {leakage_initial_known}\n")

    # Case 1: Mask cell 'd'
    print("\nCase 1: Mask {d}")
    print(f"  Initial known before mask: {leakage_initial_known}")
    mask1 = {'d'}
    print(f"  Known after applying mask: {leakage_initial_known - mask1}")
    leakage1 = calculate_leakage(edges, paths, mask1, target, leakage_initial_known, weights)
    print(f"  Checking paths:")
    # Manual trace for debugging
    for path in paths:
        print(f"    Path {path}:")
        known_so_far = leakage_initial_known - mask1
        is_blocked = False
        for i, edge_idx in enumerate(path):
            edge = edges[edge_idx]
            print(f"      Step {i + 1}: Edge {edge_idx} = {edge}")
            print(f"              Known so far: {known_so_far}")

            unknown_in_edge = [c for c in edge if c not in known_so_far]
            print(f"              Unknown in edge: {unknown_in_edge}")

            if len(unknown_in_edge) == 0:
                print(f"              → All known, nothing to infer")
            elif len(unknown_in_edge) == 1:
                known_so_far.add(unknown_in_edge[0])
                print(f"              → Can infer {unknown_in_edge[0]}")
            else:
                is_blocked = True
                print(f"              → BLOCKED (2+ unknown)")
                break

        if not is_blocked:
            path_weight = 1.0
            for edge_idx in path:
                path_weight *= weights[edge_idx]
            print(f"      → Path ACTIVE, weight = {path_weight:.4f}")
        else:
            print(f"      → Path BLOCKED")
    print(f"  Leakage = {leakage1:.4f}")

    # Case 2: Mask cell 'a'
    print("\nCase 2: Mask {a}")
    mask2 = {'a'}
    leakage2 = calculate_leakage(edges, paths, mask2, target, leakage_initial_known, weights)
    print(f"  Leakage = {leakage2:.4f}")

    # Case 3: Mask cells 'd' and 'f'
    print("\nCase 3: Mask {d, f}")
    mask3 = {'d', 'f'}
    leakage3 = calculate_leakage(edges, paths, mask3, target, leakage_initial_known, weights)
    print(f"  Leakage = {leakage3:.4f}")

    # Case 4: Mask cell 'f'
    print("\nCase 4: Mask {f}")
    mask4 = {'f'}
    leakage4 = calculate_leakage(edges, paths, mask4, target, leakage_initial_known, weights)
    print(f"  Leakage = {leakage4:.4f}")