
def filter_active_paths_str(hyperedges: List[Tuple[str, ...]],
                            paths: List[List[int]],
                            mask: Set[str],
                            initial_known: Set[str]) -> List[List[int]]:
    """
    Correctly filters for active paths by simulating an attacker's inference process.
    A path is active if all cells in it can be inferred starting from the initial_known set
    minus the cells in the mask.
    """
    active_paths = []
    
    for path in paths:
        # Start with the attacker's base knowledge, minus the deleted cells
        known_so_far = initial_known - mask
        
        # We must iterate multiple times to allow for inference chains to resolve.
        # A simple heuristic is to loop as many times as there are edges in the path,
        # plus a buffer.
        for _ in range(len(path) + 2):
            newly_inferred_this_pass = True
            while newly_inferred_this_pass:
                newly_inferred_this_pass = False
                for edge_idx in path:
                    edge = set(hyperedges[edge_idx])
                    
                    # If all but one cell in the edge are known, the last one can be inferred.
                    unknowns = edge - known_so_far
                    if len(unknowns) == 1:
                        inferred_cell = unknowns.pop()
                        if inferred_cell not in known_so_far:
                            known_so_far.add(inferred_cell)
                            newly_inferred_this_pass = True

        # After the simulation, check if the path is fully knowable.
        # A path is active if ALL of its constituent cells are now known.
        path_cells = {cell for edge_idx in path for cell in hyperedges[edge_idx]}
        if path_cells.issubset(known_so_far):
            active_paths.append(path)

    return active_paths
