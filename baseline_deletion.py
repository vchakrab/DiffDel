from rtf_core import initialization_phase
import mysql.connector
from mysql.connector import Error
import config
import random

DELETION_QUERY = """
            UPDATE {table_name}
            SET `{column_name}` = NULL
            WHERE id = {key};
            """
def baseline_deletion_delete_all(target: str, key: int, dataset, threshold):
    init_manager = initialization_phase.InitializationManager({"key": key, "attribute": target}, dataset, threshold)
    init_manager.initialize()
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
        for constraint_cell in constraint_cells_stripped:
            cursor.execute(DELETION_QUERY.format(table_name=primary_table, column_name = constraint_cell, key=key))
        cursor.execute(DELETION_QUERY.format(table_name = primary_table, column_name = target, key = key))
        conn.commit()
    except Error as e:
        print(e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return len(constraint_cells_stripped) + 1
# DO NOT RUN AS IT WILL ACTUALLY DELETE
baseline_deletion_delete_all("type", 3, "airport", 0.8)
#def baseline_deletion_1(target: str, key: int, dataset, threshold):
def baseline_deletion_delete_1_from_constraints(target: str, key: int, dataset, threshold):
    init_manager = initialization_phase.InitializationManager({"key": key, "attribute": target}, dataset, threshold)
    target_denial_constraints = []
    total_cells_deleted = 0
    try:
        # 1. Get Configuration and Query Details
        db_details = config.get_database_config(dataset)
        primary_table = dataset + "_copy_data_2"

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
        for dc in init_manager.denial_constraints:
            attrs_in_dc = set(pred.split('.')[-1] for pred in
                              [p[0] for p in dc] + [p[2] for p in dc if isinstance(p[2], str)])
            if target in attrs_in_dc:
                target_denial_constraints.append(dc)
        for dc in target_denial_constraints:
            while True:
                cell_chosen = random.choice(dc)[0].split('.')[1]
                if target == cell_chosen:
                    if(len(target_denial_constraints) == 1):
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
    finally:
        if cursor:
            cursor.close()

    return total_cells_deleted
# this works idk why it struggles on data collection
