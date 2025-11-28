from rtf_core import initialization_phase
import mysql.connector
from mysql.connector import Error
import config
def baseline_deletion(target: str, key: int, dataset, threshold):
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
        primary_table = config.get_primary_table(dataset)

        print(f"--- 🔌 Connecting to DB: {db_details['database']} ---")

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
        for constraint_cell in constraint_cells_stripped:
            QUERY = """
            UPDATE {table_name}
            SET {column_name} = NULL
            WHERE id = {key};
            """
            cursor.execute(QUERY.format(table_name=primary_table, column_name = constraint_cell, key=key))
        TARGET_CELL_QUERY = """
        UPDATE {table_name}
            SET {column_name} = NULL
            WHERE id = {key};
        """
        cursor.execute(QUERY.format(table_name = primary_table, column_name = target, key = key))
        conn.commit()
    except Error as e:
        print(e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
# DO NOT RUN AS IT WILL ACTUALLY DELETE
# baseline_deletion("type", 3, "airport", 0.8)
