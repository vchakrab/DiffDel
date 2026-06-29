import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mysql.connector
import csv
from config import DB_CONFIG, PROJECT_ROOT

DB_NAME    = "hospital"
TABLE_NAME = "hospital_data"
CSV_PATH   = PROJECT_ROOT / 'csv_files' / 'hospital.csv'


def main():
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
    )
    cursor = conn.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cursor.execute(f"USE {DB_NAME}")

    # Drop and recreate so the new nullable schema takes effect
    cursor.execute(f"DROP TABLE IF EXISTS `{TABLE_NAME}`")

    create_table_query = f"""
CREATE TABLE `{TABLE_NAME}` (
  `id`               INT AUTO_INCREMENT PRIMARY KEY,
  `ProviderNumber`   VARCHAR(20)  NULL,
  `HospitalName`     VARCHAR(255) NULL,
  `City`             VARCHAR(100) NULL,
  `State`            CHAR(2)      NULL,
  `ZIPCode`          VARCHAR(10)  NULL,
  `CountyName`       VARCHAR(100) NULL,
  `PhoneNumber`      VARCHAR(20)  NULL,
  `HospitalType`     VARCHAR(255) NULL,
  `HospitalOwner`    VARCHAR(255) NULL,
  `EmergencyService` VARCHAR(3)   NULL,
  `Condition`        VARCHAR(255) NULL,
  `MeasureCode`      VARCHAR(50)  NULL,
  `MeasureName`      TEXT         NULL,
  `Sample`           TEXT         NULL,
  `StateAvg`         VARCHAR(50)  NULL
);
"""
    cursor.execute(create_table_query)
    conn.commit()
    # print(f"✅ Table '{TABLE_NAME}' created fresh.")

    insert_query = f"""
INSERT INTO {TABLE_NAME} (
  `ProviderNumber`, `HospitalName`, `City`, `State`, `ZIPCode`,
  `CountyName`, `PhoneNumber`, `HospitalType`, `HospitalOwner`,
  `EmergencyService`, `Condition`, `MeasureCode`, `MeasureName`,
  `Sample`, `StateAvg`
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        rows = []
        skipped = 0
        for row in reader:
            if not row or len(row) < 15:
                skipped += 1
                continue
            # Skip rows with no ProviderNumber (the only NOT NULL column)
            if not row[0].strip():
                skipped += 1
                continue
            values = tuple(
                cell.strip() if cell.strip() != '' else None
                for cell in row[:15]
            )
            rows.append(values)

        cursor.executemany(insert_query, rows)
        conn.commit()

    # print(f"✅ Inserted {cursor.rowcount} rows into '{TABLE_NAME}' (skipped {skipped}).")
    cursor.close()
    conn.close()
    # print("✅ Done.")


if __name__ == '__main__':
    main()