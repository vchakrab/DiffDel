import time
import sys
from rtf_core import initialization_phase
import mysql.connector
from mysql.connector import Error
import config
import random
from rtf_core.Algorithms import enumerate_explanations as explanations


DELETION_QUERY = """
            UPDATE {table_name}
            SET `{column_name}` = NULL
            WHERE id = {key};
            """


def calculate_deletion_memory(init_manager, target, target_denial_constraints,
                              graph_boundary_edges, graph_internal_edges,
                              graph_boundary_cells, found_explanations):
    """
    Calculate memory based on Java logic for baseline deletion 2:
    - Cells visited during graph construction
    - Edges (denial constraints and graph edges)
    - Data structures used (explanations, graphs)
    """
    memory = 0

    # 1. Count all unique cells that were instantiated
    visited_cells = set()

    # Add cells from target_denial_constraints
    for dc in target_denial_constraints:
        # Each denial constraint tuple contains predicates
        # Extract cells from the predicates
        for pred in dc:
            # pred is a tuple like (attr1, op, attr2) or (attr1, op, value)
            if len(pred) >= 1:
                visited_cells.add(pred[0])  # Left side attribute
            if len(pred) >= 3 and isinstance(pred[2], str):
                visited_cells.add(pred[2])  # Right side attribute if exists

    # Add boundary cells
    visited_cells.update(graph_boundary_cells)

    # Per cell: table_index(4) + row_index(4) + insertion_time(4) + state(1) + cost(4) = 17 bytes
    memory += len(visited_cells) * 17

    # 2. Count edges (hyperedges from denial constraints)
    for dc_tuple in target_denial_constraints:
        edge_size = len(dc_tuple)
        # Per edge: pointers to cells (8 * edge_size) + parent pointer (8) + minCell (4)
        memory += edge_size * 8 + 8 + 4

    # 3. Graph edges (boundary and internal)
    # Boundary edges
    for edge in graph_boundary_edges:
        edge_size = len(edge) if hasattr(edge, '__len__') else 2
        memory += edge_size * 8 + 8 + 4

    # Internal edges (with weights)
    for edge in graph_internal_edges:
        edge_size = len(edge) if hasattr(edge, '__len__') else 2
        memory += edge_size * 8 + 8 + 4
        memory += 8  # Weight (double/float)

    # 4. Algorithm overhead - data structures
    # init_manager structures
    if hasattr(init_manager, 'constraint_cells'):
        memory += sys.getsizeof(init_manager.constraint_cells)
        if isinstance(init_manager.constraint_cells, list):
            for cell in init_manager.constraint_cells:
                memory += sys.getsizeof(cell)

    if hasattr(init_manager, 'denial_constraints'):
        memory += sys.getsizeof(init_manager.denial_constraints)
        for dc in init_manager.denial_constraints:
            memory += sys.getsizeof(dc)

    # target_denial_constraints list
    memory += sys.getsizeof(target_denial_constraints)
    for dc in target_denial_constraints:
        memory += sys.getsizeof(dc)

    # Graph data structures
    memory += sys.getsizeof(graph_boundary_edges)
    for edge in graph_boundary_edges:
        memory += sys.getsizeof(edge)

    memory += sys.getsizeof(graph_internal_edges)
    for edge in graph_internal_edges:
        memory += sys.getsizeof(edge)

    memory += sys.getsizeof(graph_boundary_cells)

    # Explanations storage
    memory += sys.getsizeof(found_explanations)
    for exp in found_explanations:
        memory += sys.getsizeof(exp)  # Each explanation tuple
        if len(exp) > 0:
            memory += sys.getsizeof(exp[0])  # The actual explanation set/list

    return memory


def get_random_key(dataset: str):
    db_details = config.get_database_config(dataset)

    # 2. Establish Connection
    conn = mysql.connector.connect(
        host = db_details['host'],
        user = db_details['user'],
        password = db_details['password'],
        database = db_details['database'],
        ssl_disabled = db_details['ssl_disabled']
    )

    if not conn.is_connected():
        print("Connection failed.")
        return 0.0
    cursor = conn.cursor()
    cursor.execute(f"""SELECT ID
                   FROM {dataset + "_copy_data"}
                   ORDER BY RAND()
                   LIMIT 1;
    """)
    return cursor.fetchone()[0]


def delete_one_path_dependent_cell(target: str, key: int, dataset: str, threshold: float):
    """
    Deletes 1 cell per path/dependence of the target cell in the given dataset and of the given key.
    Using Code from the initialization phase, we can get the cells that one cell depends on (target_constraints)
    Using enumerate_explanations, the first algorithm from the paper, we can get the weights, depth, paths of the hypergraph
    Using the DELETION_QUERY, we delete one cell per explanation.

        Parameters:
            target (string): Target attribute
            key (int): Gives the ID of the row in the primary table
            dataset (string): Name of the dataset that the table belongs to
            threshold (int): Threshold for deleting cells
        Returns:
            cells_deleted (int): The total number of cells deleted
            explanation_time (float): Time spent on explanation enumeration
            memory_bytes (int): Memory used by the deletion algorithm in bytes
        Side Effects:
            Deletes one cell per explanation in the primary table row that the target cell depends on and the ones that fit in the threshold.

    """
    init_manager = initialization_phase.InitializationManager({"key": key, "attribute": target},
                                                              dataset, threshold)
    init_manager.initialize()
    target_denial_constraints = []
    total_cells_deleted = 0
    max_depth = 0

    # Initialize variables for memory calculation
    graph_boundary_edges = []
    graph_internal_edges = []
    graph_boundary_cells = set()
    found_explanations = []

    try:
        # 1. Get Configuration and Query Details
        db_details = config.get_database_config(dataset)
        primary_table = dataset + "_copy_data"

        # 2. Establish Connection
        conn = mysql.connector.connect(
            host = db_details['host'],
            user = db_details['user'],
            password = db_details['password'],
            database = db_details['database'],
            ssl_disabled = db_details['ssl_disabled']
        )

        if not conn.is_connected():
            print("Connection failed.")
            return 0, 0.0, 0
        cursor = conn.cursor()

        for dc in init_manager.denial_constraints:
            attrs_in_dc = set(pred.split('.')[-1] for pred in
                              [p[0] for p in dc] + [p[2] for p in dc if isinstance(p[2], str)])
            if target in attrs_in_dc:
                target_denial_constraints.append(dc)

        start_time = time.time()
        graph_boundary_edges, graph_internal_edges, graph_boundary_cells = explanations.build_graph_data(
            target_denial_constraints)
        found_explanations = explanations.find_all_weighted_explanations(
            explanations.set_edge_weight(graph_internal_edges),
            "t1." + target,
            graph_boundary_cells,
            5
        )
        end_time = time.time() - start_time

        # Calculate memory AFTER graph construction but BEFORE deletions
        memory_bytes = calculate_deletion_memory(
            init_manager,
            target,
            target_denial_constraints,
            graph_boundary_edges,
            graph_internal_edges,
            graph_boundary_cells,
            found_explanations
        )

        for exp in found_explanations:
            actual_exp = exp[0]
            depth = exp[2]
            if depth > max_depth:
                max_depth = depth
            while True:
                cell_chosen = random.choice(list(actual_exp))[3:]
                if target == cell_chosen:
                    if (len(target_denial_constraints) == 1):
                        break
                    continue
                else:
                    total_cells_deleted += 1
                    cursor.execute(
                        DELETION_QUERY.format(table_name = primary_table, column_name = cell_chosen,
                                              key = key))
                    break

        total_cells_deleted += 1
        cursor.execute(
            DELETION_QUERY.format(table_name = primary_table, column_name = target, key = key))
        conn.commit()

    except Error as e:
        print(e)
        return 0, 0.0, 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return total_cells_deleted, end_time, memory_bytes, max_depth, len(found_explanations)