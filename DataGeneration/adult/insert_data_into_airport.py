import mysql.connector
import csv

DB_NAME = "airport"
TABLE_NAME = "airports" # change this to the table name
CSV_PATH = "/DataGeneration/csv_files/cleaned_airport.csv"  # <-- change this path
USER = "root"
PASSWORD = "my_password" # change password
HOST = "localhost"

conn = mysql.connector.connect(
    host=HOST,
    user=USER,
    password=PASSWORD
)
cursor = conn.cursor()

cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
cursor.execute(f"USE {DB_NAME}")

create_table_query = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INT PRIMARY KEY,
    ident VARCHAR(10),
    type VARCHAR(50),
    name VARCHAR(255),
    latitude_deg FLOAT,
    longitude_deg FLOAT,
    elevation_ft INT,
    iso_country VARCHAR(10),
    iso_region VARCHAR(10),
    municipality VARCHAR(100),
    scheduled_service VARCHAR(1000)
);
"""
cursor.execute(create_table_query)
conn.commit()

ALTER_TABLE_SQL = """
ALTER TABLE airports MODIFY COLUMN scheduled_service VARCHAR(100);
"""
cursor.execute(ALTER_TABLE_SQL)
conn.commit()
print(f"Table '{TABLE_NAME}' created (if not already).")

insert_query = f"""
INSERT INTO {TABLE_NAME} (
    id, ident, type, name, latitude_deg, longitude_deg, elevation_ft,
    iso_country, iso_region, municipality, scheduled_service
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# --- READ CSV AND INSERT ---
with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = []
    for row in reader:
        # Handle possible empty lines or missing fields
        if not row['id']:
            continue
        values = (
            int(row['id']),
            row['ident'],
            row['type'],
            row['name'],
            float(row['latitude_deg']) if row['latitude_deg'] else None,
            float(row['longitude_deg']) if row['longitude_deg'] else None,
            int(float(row['elevation_ft'])) if row['elevation_ft'] else None,
            row['iso_country'],
            row['iso_region'],
            row['municipality'],
            row['scheduled_service']
        )
        rows.append(values)

    cursor.executemany(insert_query, rows)
    conn.commit()

print("✅ Inserted {cursor.rowcount} rows from {CSV_PATH} into '{TABLE_NAME}'.")

cursor.close()
conn.close()
print("✅ Database initialization complete.")
