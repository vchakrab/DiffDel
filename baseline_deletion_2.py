import time
import sys
from rtf_core import initialization_phase
import mysql.connector
from mysql.connector import Error
import config
import random
from collections import deque
from rtf_core.Algorithms import enumerate_explanations as explanations


DELETION_QUERY = """
            UPDATE {table_name}
            SET `{column_name}` = NULL
            WHERE id = {key};
            """


def measure_approximate_memory(nodes_instantiated, edges_instantiated, cell_to_edges):
    """
    Calculate memory based on P2E2/Java logic for the approximate approach.
    """
    size = 0

    # per cell: 4 bytes table index, 4 bytes row index, 4 bytes insertionTime, 1 byte state
    size += len(nodes_instantiated) * 13

    for cell in edges_instantiated:
        # In Java, this iterates through model.cell2Edge.getOrDefault(cell, EMPTY_LIST)
        # We replicate this by looking up the cell in our constructed cell_to_edges map.
        if cell in cell_to_edges:
            for edge in cell_to_edges[cell]:
                # 8 bytes per element in hyperedge + 8 bytes for pointer from head to edge + 4 bytes for the cheapest node
                size += len(edge) * 8 + 8 + 4
            
    return size


def get_random_key(dataset: str):
    """
    Get a random key from the dataset.

    Parameters:
        dataset (str): Name of the dataset

    Returns:
        int: Random ID from the dataset
    """
    db_details = config.get_database_config(dataset)

    conn = mysql.connector.connect(
        host = db_details['host'],
        user = db_details['user'],
        password = db_details['password'],
        database = db_details['database'],
        ssl_disabled = db_details['ssl_disabled']
    )

    if not conn.is_connected():
        print("Connection failed.")
        return None

    cursor = conn.cursor()
    try:
        cursor.execute(f"""SELECT ID
                       FROM {dataset + "_copy_data"}
                       ORDER BY RAND()
                       LIMIT 1;
        """)
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        cursor.close()
        conn.close()


def delete_one_path_dependent_cell(target: str, key: int, dataset: str, threshold: float):
    """
    Deletes 1 cell per path/dependence of the target cell in the given dataset and of the given key.

    This implements a path-based deletion strategy where one cell is deleted per explanation path.

    According to P2E2 paper methodology:
    1. Instantiation: Initialize and get denial constraints for target
    2. Modeling: Build hypergraph and enumerate explanations
    3. Optimization: Find weighted explanations (paths)
    4. Update to NULL: Delete one cell per path

    Parameters:
        target (string): Target attribute
        key (int): Gives the ID of the row in the primary table
        dataset (string): Name of the dataset that the table belongs to
        threshold (float): Threshold for deleting cells

    Returns:
        tuple: (total_cells_deleted, memory_bytes, max_depth, num_explanations,
                instantiation_time, model_time, deletion_time)

    Side Effects:
        Deletes one cell per explanation in the primary table row that the target cell
        depends on and the ones that fit in the threshold.
    """

    # Phase 1: INSTANTIATION - Initialize and get denial constraints
    instantiation_start = time.time()

    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target},
        dataset,
        threshold
    )
    init_manager.initialize()

    # Filter denial constraints that involve the target attribute
    target_denial_constraints = []
    for dc in init_manager.denial_constraints:
        attrs_in_dc = set(pred.split('.')[-1] for pred in
                          [p[0] for p in dc] + [p[2] for p in dc if isinstance(p[2], str)])
        if target in attrs_in_dc:
            target_denial_constraints.append(dc)

    instantiation_time = time.time() - instantiation_start

    # Phase 2: MODELING - Build hypergraph structure
    model_start = time.time()

    # Build graph data structures
    graph_boundary_edges, graph_internal_edges, graph_boundary_cells = explanations.build_graph_data(
        target_denial_constraints
    )

    # Set edge weights
    weighted_edges = explanations.set_edge_weight(graph_internal_edges)

    # --- Start of new logic to replicate Java's approximateDelete traversal ---
    
    # First, build the cell_to_edges map from the hypergraph
    cell_to_edges = {}
    for hyperedge in weighted_edges.keys():
        # In this model, any cell can be a "head". We map each cell in a hyperedge
        # to the list of hyperedges it belongs to.
        for cell in hyperedge:
            if cell not in cell_to_edges:
                cell_to_edges[cell] = []
            cell_to_edges[cell].append(hyperedge)
            
    # Now, traverse the graph to find instantiated nodes and edges, like in Java
    nodes_instantiated = set()
    edges_instantiated = set()
    q = deque([f"t1.{target}"])
    visited = {f"t1.{target}"}
    
    while q:
        curr = q.popleft()
        nodes_instantiated.add(curr)
        edges_instantiated.add(curr) # In Java, edges_instantiated tracks heads
        
        # Get edges for the current cell
        if curr in cell_to_edges:
            for edge in cell_to_edges[curr]:
                nodes_instantiated.update(edge)
                for cell in edge:
                    # Look at grandchildren to populate nodes_instantiated
                    if cell in cell_to_edges:
                        for grandchild_edge in cell_to_edges[cell]:
                            nodes_instantiated.update(grandchild_edge)
                    
                    if cell not in visited:
                        # In the approximate version, we don't traverse fully,
                        # but we do need to add all cells from the edge.
                        # The traversal in Java is more complex, this is a simplification
                        # to get the sets of nodes and edges.
                        pass # Don't add to queue, as we're not finding deletion set here
    
    # --- End of new logic ---

    # Phase 3: OPTIMIZATION - Enumerate explanations (paths)
    found_explanations = explanations.find_all_weighted_explanations(
        weighted_edges,
        "t1." + target,
        graph_boundary_cells,
        5  # max_depth parameter
    )

    model_time = time.time() - model_start

    # Calculate memory AFTER graph construction but BEFORE deletions
    memory_bytes = measure_approximate_memory(
        nodes_instantiated,
        edges_instantiated,
        cell_to_edges
    )

    # Track maximum depth and cells to delete
    max_depth = 0
    total_cells_deleted = 0
    deletion_time = 0.0

    conn = None
    cursor = None

    try:
        # Get database configuration
        db_details = config.get_database_config(dataset)
        primary_table = dataset + "_copy_data"

        # Establish connection
        conn = mysql.connector.connect(
            host = db_details['host'],
            user = db_details['user'],
            password = db_details['password'],
            database = db_details['database'],
            ssl_disabled = db_details['ssl_disabled']
        )

        if not conn.is_connected():
            print("Warning: Database connection failed")
            return (
                0,
                memory_bytes,
                0,
                len(found_explanations),
                instantiation_time,
                model_time,
                0.0
            )

        cursor = conn.cursor()

        # Phase 4: UPDATE TO NULL - Execute deletions
        deletion_start = time.time()

        # Delete one cell per explanation path
        for exp in found_explanations:
            actual_exp = exp[0]  # The explanation (set of cells)
            depth = exp[2]  # Depth of the path

            if depth > max_depth:
                max_depth = depth

            # Select one cell from this explanation to delete
            # Avoid deleting the target cell unless it's the only option
            while True:
                cell_chosen = random.choice(list(actual_exp))[3:]  # Remove "t1." prefix

                if cell_chosen == target:
                    # Only delete target if it's the only cell in single-constraint case
                    if len(target_denial_constraints) == 1:
                        break
                    continue
                else:
                    total_cells_deleted += 1
                    cursor.execute(
                        DELETION_QUERY.format(
                            table_name = primary_table,
                            column_name = cell_chosen,
                            key = key
                        )
                    )
                    break

        # Always delete the target cell at the end
        total_cells_deleted += 1
        cursor.execute(
            DELETION_QUERY.format(
                table_name = primary_table,
                column_name = target,
                key = key
            )
        )

        conn.commit()
        deletion_time = time.time() - deletion_start

    except Error as e:
        print(f"Database error: {e}")
        deletion_time = 0.0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    # Return all metrics as per P2E2 evaluation methodology
    return (
        len(found_explanations),
        total_cells_deleted,  # Number of cells deleted
        memory_bytes,  # Memory overhead in bytes
        max_depth,  # Maximum depth of explanation paths,
        instantiation_time,  # Phase 1: Instantiation time
        model_time,  # Phase 2+3: Modeling + Optimization time
        deletion_time  # Phase 4: Update to NULL time
    )

if __name__ == "__main__":
    dataset = "airport"
    target = "type"
    key = 6323
    threshold = 5.0
    
    num_explanations, total_cells_deleted, memory_bytes, max_depth, instantiation_time, model_time, deletion_time = delete_one_path_dependent_cell(target, key, dataset, threshold)
    
    print(f"{'='*70}")
    print(f"FINAL RESULTS FOR BASELINE 2:")
    print(f"  - Explanations found: {num_explanations}")
    print(f"  - Cells deleted: {total_cells_deleted}")
    print(f"  - Memory usage (bytes): {memory_bytes}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Instantiation time: {instantiation_time:.4f}s")
    print(f"  - Model & optimization time: {model_time:.4f}s")
    print(f"  - Deletion time: {deletion_time:.4f}s")
    print(f"{'='*70}")