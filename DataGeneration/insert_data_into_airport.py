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
    id INT AUTO_INCREMENT PRIMARY KEY,
    ident VARCHAR(20),
    type VARCHAR(50),
    name VARCHAR(255),
    latitude_deg FLOAT,
    longitude_deg FLOAT,
    elevation_ft INT,
    continent VARCHAR(10),
    iso_country VARCHAR(10),
    iso_region VARCHAR(20),
    municipality VARCHAR(100),
    scheduled_service VARCHAR(100),
    gps_code VARCHAR(20),
    iata_code VARCHAR(10),
    local_code VARCHAR(20),
    home_link VARCHAR(255),
    wikipedia_link VARCHAR(255),
    keywords TEXT
);
"""
    cursor.execute(create_table_query)
    conn.commit()

    rows = []

    with open(CSV_PATH, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        # Strip type annotations: "latitude_deg float" → "latitude_deg"
        clean_header = [h.split()[0].strip().lower().replace('-', '_') for h in reader.fieldnames]
        reader.fieldnames = clean_header

        for i, raw_row in enumerate(reader):
            row = {k.strip(): v.strip() for k, v in raw_row.items() if v is not None}

            if i == 0:

            try:
                values = (
                    row.get("ident"),
                    row.get("type"),
                    row.get("name"),
                    float(row["latitude_deg"]) if row.get("latitude_deg") else None,
                    float(row["longitude_deg"]) if row.get("longitude_deg") else None,
                    int(row["elevation_ft"]) if row.get("elevation_ft") else None,
                    row.get("continent") or None,
                    row.get("iso_country") or None,
                    row.get("iso_region") or None,
                    row.get("municipality") or None,
                    row.get("scheduled_service") or None,
                    row.get("gps_code") or None,
                    row.get("iata_code") or None,
                    row.get("local_code") or None,
                    row.get("home_link") or None,
                    row.get("wikipedia_link") or None,
                    row.get("keywords") or None,
                )
                rows.append(values)
            except (ValueError, KeyError) as e:
                continue


    if not rows:
        cursor.close()
        conn.close()
        return

    insert_query = f"""
INSERT INTO {TABLE_NAME} (
    ident, type, name, latitude_deg, longitude_deg, elevation_ft,
    continent, iso_country, iso_region, municipality, scheduled_service,
    gps_code, iata_code, local_code, home_link, wikipedia_link, keywords
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

    try:
        cursor.executemany(insert_query, rows)
        conn.commit()
    except mysql.connector.Error as e:
        conn.rollback()
        raise

    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()