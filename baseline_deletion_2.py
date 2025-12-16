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

    # Phase 2: MODELING & OPTIMIZATION - Build hypergraph, find explanations
    model_start = time.time()

    graph_boundary_edges, graph_internal_edges, graph_boundary_cells = explanations.build_graph_data(
        target_denial_constraints
    )
    weighted_edges = explanations.set_edge_weight(graph_internal_edges)
    
    # The memory calculation logic needs the graph, so it's part of modeling
    cell_to_edges = {cell: [h for h in weighted_edges.keys() if cell in h] for h in weighted_edges.keys() for cell in h}
    nodes_instantiated, edges_instantiated = set(), set()
    q = deque([f"t1.{target}"])
    visited = {f"t1.{target}"}
    while q:
        curr = q.popleft()
        nodes_instantiated.add(curr)
        edges_instantiated.add(curr)
        if curr in cell_to_edges:
            for edge in cell_to_edges[curr]:
                nodes_instantiated.update(edge)
                for cell in edge:
                    if cell in cell_to_edges:
                        for grandchild_edge in cell_to_edges[cell]:
                            nodes_instantiated.update(grandchild_edge)
    
    found_explanations = explanations.find_all_weighted_explanations(
        weighted_edges, "t1." + target, graph_boundary_cells, 5
    )
    model_time = time.time() - model_start

    memory_bytes = measure_approximate_memory(nodes_instantiated, edges_instantiated, cell_to_edges)

    # Phase 4: DELETION
    deletion_start = time.time()
    max_depth = 0
    total_cells_deleted = 0
    conn = None
    try:
        db_details = config.get_database_config(dataset)
        conn = mysql.connector.connect(**db_details)
        if not conn.is_connected():
            return (len(found_explanations), 0, memory_bytes, 0, instantiation_time, model_time, 0.0)
        
        cursor = conn.cursor()
        primary_table = f"{dataset}_copy_data"

        for exp in found_explanations:
            actual_exp, depth = exp[0], exp[2]
            if depth > max_depth: max_depth = depth
            
            # Simplified deletion choice
            cell_to_del = next((c for c in actual_exp if c != f"t1.{target}"), f"t1.{target}")
            
            total_cells_deleted += 1
            cursor.execute(DELETION_QUERY.format(table_name=primary_table, column_name=cell_to_del.split('.')[-1], key=key))

        # Always delete the target
        total_cells_deleted += 1
        cursor.execute(DELETION_QUERY.format(table_name=primary_table, column_name=target, key=key))
        conn.commit()
    except Error as e:
        # Error case
        pass
    finally:
        if conn: conn.close()
    deletion_time = time.time() - deletion_start

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
