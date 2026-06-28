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

    create_table_query = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
  `id`               INT AUTO_INCREMENT PRIMARY KEY,
  `ProviderNumber`   VARCHAR(20)  NOT NULL,
  `HospitalName`     VARCHAR(255) NOT NULL,
  `City`             VARCHAR(100) NOT NULL,
  `State`            CHAR(2)      NOT NULL,
  `ZIPCode`          VARCHAR(10)  NOT NULL,
  `CountyName`       VARCHAR(100) NOT NULL,
  `PhoneNumber`      VARCHAR(20)  NOT NULL,
  `HospitalType`     VARCHAR(255) NOT NULL,
  `HospitalOwner`    VARCHAR(255) NOT NULL,
  `EmergencyService` VARCHAR(3)   NOT NULL,
  `Condition`        VARCHAR(255) NOT NULL,
  `MeasureCode`      VARCHAR(50)  NOT NULL,
  `MeasureName`      TEXT         NOT NULL,
  `Sample`           TEXT         NULL,
  `StateAvg`         VARCHAR(50)  NULL
);
"""
    cursor.execute(create_table_query)
    conn.commit()
    print(f"✅ Table '{TABLE_NAME}' created (if not already).")

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
        next(reader, None)
        rows = []
        for row in reader:
            if not row or len(row) < 15:
                continue
            values = tuple(
                cell.strip() if cell.strip() != '' else None
                for cell in row[:15]
            )
            rows.append(values)
        cursor.executemany(insert_query, rows)
        conn.commit()

    print(f"✅ Inserted {cursor.rowcount} rows from {CSV_PATH} into '{TABLE_NAME}'.")
    cursor.close()
    conn.close()
    print("✅ Done.")


if __name__ == '__main__':
    main()
