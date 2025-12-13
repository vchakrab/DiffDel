import time

from rtf_core import initialization_phase
import mysql.connector
from mysql.connector import Error
import config
import sys
DELETION_QUERY = """
            UPDATE {table_name}
            SET `{column_name}` = NULL
            WHERE id = {key};
            """


def calculate_deletion_memory(init_manager, constraint_cells_stripped, target, key):
    """
    Calculate memory based on Java logic:
    - denial_constraints: list of tuples (each tuple is a hyperedge)
    - constraint_cells: list of cells
    """
    memory = 0

    # 1. Count cells visited/instantiated
    # The target cell + all constraint cells
    total_cells = len(constraint_cells_stripped) + 1

    # Per cell: table_index(4) + row_index(4) + insertion_time(4) + state(1) + cost(4) = 17 bytes
    memory += total_cells * 17

    # 2. Count edges (denial constraints = hyperedges)
    for constraint_tuple in init_manager.denial_constraints:
        # Each tuple represents a hyperedge connecting multiple cells
        edge_size = len(constraint_tuple)

        # Per edge: pointers to cells (8 * edge_size) + parent pointer (8) + minCell (4)
        memory += edge_size * 8 + 8 + 4

    # 3. Algorithm overhead - data structures used
    # Python list overhead for constraint_cells
    memory += sys.getsizeof(init_manager.constraint_cells)
    for cell in init_manager.constraint_cells:
        memory += sys.getsizeof(cell)  # Each cell object

    # List overhead for denial_constraints
    memory += sys.getsizeof(init_manager.denial_constraints)
    for constraint_tuple in init_manager.denial_constraints:
        memory += sys.getsizeof(constraint_tuple)  # Each tuple

    # List overhead for stripped_attributes
    memory += sys.getsizeof(constraint_cells_stripped)

    # If init_manager has other internal structures (queue, visited set, etc.)
    # Add them here if they exist:

    return memory

def delete_all_dependent_cells(target: str, key: int, dataset, threshold):
    """
    Delete all dependent cells of the target cell in the key row in the dataset's primary table.
    Using Code from the initialization phase, we can get the cells that one cell depends on.
    Using the DELETION_QUERY, we delete each cell.

        Parameters:
            target (string): Target attribute
            key (int): Gives the ID of the row in the primary table
            dataset (string): Name of the dataset that the table belongs to
            threshold (int): Threshold for deleting cells, used in baseline deletion two, but plqceholder here
        Returns:
            cells deleted (int): The total number of cells deleted
        Side Effects:
            Deletes all the cells in the primary table row that the target cell depends on.`

    """
    memory = 0
    initialization_time = time.time()
    init_manager = initialization_phase.InitializationManager({"key": key, "attribute": target}, dataset, threshold)
    init_manager.initialize()

    cleaned_content = str(init_manager.constraint_cells).strip('{}')
    total_init_time = time.time() - initialization_time
    model_time = time.time()
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
    model_time = time.time() - model_time
    memory_bytes = calculate_deletion_memory(init_manager, constraint_cells_stripped, target, key)
    deletion_time = None
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
            return 0.0
        cursor = conn.cursor()
        deletion_time = time.time()
        for constraint_cell in constraint_cells_stripped:
            cursor.execute(DELETION_QUERY.format(table_name=primary_table, column_name = constraint_cell, key=key))
        cursor.execute(DELETION_QUERY.format(table_name = primary_table, column_name = target, key = key))
        conn.commit()
        deletion_time = time.time() - deletion_time
    except Error as e:
        print(e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return len(init_manager.denial_constraints), len(constraint_cells_stripped) + 1, memory_bytes, total_init_time, model_time, deletion_time

