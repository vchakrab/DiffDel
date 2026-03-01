import mysql.connector
import csv

DB_NAME = "airport"
TABLE_NAME = "airports"
CSV_PATH = "/csv_files/airport.csv"
USER = "root"
PASSWORD = "my_password"
HOST = "localhost"

conn = mysql.connector.connect(
    host=HOST,
    user=USER,
    password=PASSWORD
)
cursor = conn.cursor()

cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
cursor.execute(f"USE {DB_NAME}")
cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

# --- CREATE FULL TABLE ---
create_table_query = f"""
CREATE TABLE {TABLE_NAME} (
    id INT PRIMARY KEY,
    ident VARCHAR(20),
    type VARCHAR(50),
    name VARCHAR(255),
    latitude_deg FLOAT,
    longitude_deg FLOAT,
    elevation_ft INT,
    continent VARCHAR(10),
    iso_country VARCHAR(10),
    iso_region VARCHAR(10),
    municipality VARCHAR(100),
    scheduled_service VARCHAR(50),
    gps_code VARCHAR(20),
    iata_code VARCHAR(10),
    local_code VARCHAR(20),
    home_link TEXT,
    wikipedia_link TEXT,
    keywords TEXT
);
"""
cursor.execute(create_table_query)
conn.commit()

print(f"Table '{TABLE_NAME}' created fresh.")

# --- INSERT QUERY (ALL 17 COLUMNS) ---
insert_query = f"""
INSERT INTO {TABLE_NAME} (
    id, ident, type, name, latitude_deg, longitude_deg, elevation_ft,
    continent, iso_country, iso_region, municipality,
    scheduled_service, gps_code, iata_code, local_code,
    home_link, wikipedia_link, keywords
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# --- READ CSV AND INSERT ---
with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = []

    for row in reader:
        values = (
            int(row['id']) if row.get('id') else None,
            row.get('ident'),
            row.get('type'),
            row.get('name'),
            float(row['latitude_deg']) if row.get('latitude_deg') else None,
            float(row['longitude_deg']) if row.get('longitude_deg') else None,
            int(float(row['elevation_ft'])) if row.get('elevation_ft') else None,
            row.get('continent'),
            row.get('iso_country'),
            row.get('iso_region'),
            row.get('municipality'),
            row.get('scheduled_service'),
            row.get('gps_code'),
            row.get('iata_code'),
            row.get('local_code'),
            row.get('home_link'),
            row.get('wikipedia_link'),
            row.get('keywords')
        )
        rows.append(values)

    cursor.executemany(insert_query, rows)
    conn.commit()

print(f"✅ Inserted {cursor.rowcount} rows from CSV into '{TABLE_NAME}'.")

cursor.close()
conn.close()
print("✅ Database initialization complete.")
