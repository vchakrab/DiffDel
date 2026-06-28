import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mysql.connector
import csv
from config import DB_CONFIG, PROJECT_ROOT

DB_NAME    = "airport"
TABLE_NAME = "airports"
CSV_PATH   = PROJECT_ROOT / 'csv_files' / 'airport.csv'


def main():
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
    )
    cursor = conn.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cursor.execute(f"USE {DB_NAME}")
    cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

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
    print(f"✅ Table '{TABLE_NAME}' created fresh.")

    insert_query = f"""
INSERT INTO {TABLE_NAME} (
    id, ident, type, name, latitude_deg, longitude_deg, elevation_ft,
    continent, iso_country, iso_region, municipality,
    scheduled_service, gps_code, iata_code, local_code,
    home_link, wikipedia_link, keywords
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

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
    print("✅ Done.")


if __name__ == '__main__':
    main()
