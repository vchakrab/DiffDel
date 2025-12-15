import time
import sys
from rtf_core import initialization_phase
import mysql.connector
from mysql.connector import Error
import config


DELETION_QUERY = """
            UPDATE {table_name}
            SET `{column_name}` = NULL
            WHERE id = {key};
            """


def measure_optimal_memory(init_manager, target, key):
    """
    Calculate memory based on P2E2/Java logic for the optimal approach.
    This involves traversing the instantiated cells and edges.
    """
    memory = 0
    
    # Create a graph-like structure for traversal
    cell_to_edges = {}
    all_cells = {f"t1.{target}"}

    for dc in init_manager.denial_constraints:
        # Each DC is a hyperedge. Let's assume the first element is the head for simplicity.
        # This is a simplification, but it allows us to build a traversable graph.
        head_attr = dc[0][0] if dc else None
        if not head_attr:
            continue
            
        head_cell = f"t1.{head_attr.split('.')[-1]}"
        all_cells.add(head_cell)
        
        edge = set()
        for pred in dc:
            attr = pred[0].split('.')[-1]
            cell = f"t1.{attr}"
            edge.add(cell)
            all_cells.add(cell)

        if head_cell not in cell_to_edges:
            cell_to_edges[head_cell] = []
        cell_to_edges[head_cell].append(edge)

    # Traverse from the target cell
    q = [f"t1.{target}"]
    visited = {f"t1.{target}"}
    
    while q:
        curr_cell_attr = q.pop(0)
        
        # Per cell: table_index(4) + row_index(4) + insertion_time(4) + state(1) + cost(4) = 17 bytes
        memory += 17
        
        if curr_cell_attr in cell_to_edges:
            for edge in cell_to_edges[curr_cell_attr]:
                # Per edge: pointers to cells (8 * edge_size) + parent pointer (8) + minCell (4) = 12 + 8 * size
                memory += 12 + len(edge) * 8
                
                for cell in edge:
                    if cell not in visited:
                        visited.add(cell)
                        q.append(cell)
                        
    return memory


def delete_all_dependent_cells(target: str, key: int, dataset, threshold):
    """
    Delete all dependent cells of the target cell in the key row in the dataset's primary table.
    Using Code from the initialization phase, we can get the cells that one cell depends on.
    Using the DELETION_QUERY, we delete each cell.

    According to P2E2 paper (Section 4), the phases are:
    1. Instantiation: Instantiate RDRs to identify dependencies (Algorithm 1)
    2. Modeling: Construct the optimization model (ILP/Hypergraph)
    3. Optimization: Solve to find minimal deletion set
    4. Update to NULL: Execute the actual deletions in database

    Parameters:
        target (string): Target attribute
        key (int): Gives the ID of the row in the primary table
        dataset (string): Name of the dataset that the table belongs to
        threshold (int): Threshold for deleting cells, used in baseline deletion two, but placeholder here

    Returns:
        tuple: (num_constraints, num_cells_deleted, memory_bytes,
                instantiation_time, model_time, deletion_time)

    Side Effects:
        Deletes all the cells in the primary table row that the target cell depends on.
    """

    # Phase 1: INSTANTIATION - Instantiate RDRs and identify dependencies
    instantiation_start = time.time()
    init_manager = initialization_phase.InitializationManager(
        {"key": key, "attribute": target},
        dataset,
        threshold
    )
    init_manager.initialize()
    instantiation_time = time.time() - instantiation_start

    # Phase 2: MODELING - Extract and process constraint cells
    model_start = time.time()

    # Parse constraint cells from initialization manager
    cleaned_content = str(init_manager.constraint_cells).strip('{}')
    items = cleaned_content.split(', ')

    stripped_attributes = []
    for item in items:
        key_part = item.split('=>')[0].strip()
        dot_index = key_part.find('.')
        bracket_index = key_part.find('[')

        if dot_index != -1 and bracket_index != -1 and dot_index < bracket_index:
            attribute = key_part[dot_index + 1: bracket_index]
            stripped_attributes.append(attribute)

    constraint_cells_stripped = stripped_attributes

    # Calculate memory usage for the model
    memory_bytes = measure_optimal_memory(
        init_manager,
        target,
        key
    )

    model_time = time.time() - model_start

    # Phase 3: OPTIMIZATION (implicit in this baseline approach)
    # For baseline, we delete all dependent cells, so optimization time is negligible
    # and included in model_time

    # Phase 4: UPDATE TO NULL - Execute deletions in database
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
                len(init_manager.denial_constraints),
                len(constraint_cells_stripped) + 1,
                memory_bytes,
                instantiation_time,
                model_time,
                0.0
            )

        cursor = conn.cursor()

        # Start timing the actual deletion operations
        deletion_start = time.time()

        # Delete all constraint cells
        for constraint_cell in constraint_cells_stripped:
            cursor.execute(
                DELETION_QUERY.format(
                    table_name = primary_table,
                    column_name = constraint_cell,
                    key = key
                )
            )

        # Delete the target cell
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

    # Return metrics as per P2E2 evaluation
    return (
        len(init_manager.denial_constraints),  # Number of instantiated RDRs
        len(constraint_cells_stripped) + 1,  # Number of cells deleted
        memory_bytes,  # Memory overhead
        instantiation_time,  # Time for Phase 1: Instantiation
        model_time,  # Time for Phase 2: Modeling
        deletion_time  # Time for Phase 4: Update to NULL
    )