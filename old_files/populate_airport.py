import mysql.connector
from mysql.connector import errorcode


# --- CONFIGURATION ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'my_password',
    'database': 'airport',  # <--- REQUIRED: You need a database name!
    'ssl_disabled': True,
    'charset': 'utf8mb4',
    # Optional: Set 'local_infile' to True for bulk loading later
    'allow_local_infile': True
}


CSV_FILE_PATH = '/csv_files/airport.csv'  # <-- CHANGE THIS to the actual file path

# SQL to create the table structure
CREATE_TABLE_SQL = """
CREATE TABLE airports (
    id INT PRIMARY KEY,
    ident VARCHAR(10) NOT NULL UNIQUE,
    type VARCHAR(50),
    name VARCHAR(255),
    latitude_deg FLOAT,
    longitude_deg FLOAT,
    elevation_ft INT,
    continent VARCHAR(5),
    iso_country VARCHAR(5),
    iso_region VARCHAR(10),
    municipality VARCHAR(100),
    scheduled_service VARCHAR(5),
    gps_code VARCHAR(10),
    iata_code VARCHAR(5),
    local_code VARCHAR(10),
    home_link VARCHAR(255),
    wikipedia_link VARCHAR(255),
    keywords TEXT
);
"""

# SQL for bulk loading the data
LOAD_DATA_SQL = f"""
LOAD DATA LOCAL INFILE '{CSV_FILE_PATH}'
INTO TABLE airports
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 ROWS;
"""


def initialize_database():
    """Drops the table, creates the table, and loads data from CSV."""
    db = None
    try:
        print("Attempting to connect to the MySQL database...")
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor()

        # 1. Drop the table (clean slate)
        print("Dropping old 'airports' table if it exists...")
        cursor.execute("DROP TABLE IF EXISTS airports;")

        # 2. Create the new table
        print("Creating new 'airports' table structure...")
        cursor.execute(CREATE_TABLE_SQL)

        # 3. Bulk Load Data
        print(f"Loading data from CSV: {CSV_FILE_PATH}")
        # Note: 'LOAD DATA LOCAL INFILE' requires 'allow_local_infile=True' in the connection options
        # or globally enabled on the server.
        cursor.execute(LOAD_DATA_SQL)

        # 4. Commit changes
        db.commit()
        print("\n✅ Database initialization complete! Data loaded successfully.")

        # Optional: Verify row count
        cursor.execute("SELECT COUNT(*) FROM airports;")
        count = cursor.fetchone()[0]
        print(f"Total rows loaded: {count}")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("❌ Error: Access denied. Check your user name and password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("❌ Error: Database does not exist. Check your database name.")
        elif err.errno == 2068:
            print(
                "❌ Error: LOCAL INFILE support is disabled. You must enable it in your MySQL server configuration.")
            print("Consider running 'SET GLOBAL local_infile = ON;' in your MySQL client.")
        else:
            print(f"❌ An error occurred: {err}")

    finally:
        if db is not None and db.is_connected():
            cursor.close()
            db.close()
            print("Connection closed.")


if __name__ == "__main__":
    initialize_database()