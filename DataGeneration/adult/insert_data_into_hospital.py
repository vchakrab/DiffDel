import mysql.connector
import csv

# --- CONFIG ---
DB_NAME = "hospital"
TABLE_NAME = "hospital_data"
CSV_PATH = "/DataGeneration/csv_files/hospital.csv"  # <-- change this path
USER = "root"
PASSWORD = "my_password"
HOST = "localhost"

# --- CONNECT TO MYSQL ---
conn = mysql.connector.connect(
    host=HOST,
    user=USER,
    password=PASSWORD
)
cursor = conn.cursor()

# --- CREATE DATABASE ---
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
cursor.execute(f"USE {DB_NAME}")

# --- CREATE TABLE ---
create_table_query = f"""
CREATE TABLE IF NOT EXISTS `hospital_data` (
  `id`             INT AUTO_INCREMENT PRIMARY KEY,  
  `ProviderNumber` VARCHAR(20)    NOT NULL,
  `HospitalName`   VARCHAR(255)   NOT NULL,
  `City`           VARCHAR(100)   NOT NULL,
  `State`          CHAR(2)        NOT NULL,
  `ZIPCode`        VARCHAR(10)    NOT NULL,
  `CountyName`     VARCHAR(100)   NOT NULL,
  `PhoneNumber`    VARCHAR(20)    NOT NULL,
  `HospitalType`   VARCHAR(255)   NOT NULL,
  `HospitalOwner`  VARCHAR(255)   NOT NULL,
  `EmergencyService` VARCHAR(3)   NOT NULL,
  `Condition`      VARCHAR(255)   NOT NULL,
  `MeasureCode`    VARCHAR(50)    NOT NULL,
  `MeasureName`    TEXT           NOT NULL,
  `Sample`         TEXT           NULL,
  `StateAvg`       VARCHAR(50)    NULL
);
"""
print(f"✅ Table '{TABLE_NAME}' created (if not already).")

# --- PREPARE INSERT QUERY ---
insert_query = f"""
INSERT INTO {TABLE_NAME} (
  `ProviderNumber`, 
  `HospitalName`, 
  `City`, 
  `State`, 
  `ZIPCode`, 
  `CountyName`, 
  `PhoneNumber`, 
  `HospitalType`, 
  `HospitalOwner`, 
  `EmergencyService`, 
  `Condition`, 
  `MeasureCode`, 
  `MeasureName`, 
  `Sample`, 
  `StateAvg`
) VALUES (
  %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s
);
"""

# --- READ CSV AND INSERT ---
with open(CSV_PATH, newline = '', encoding = 'utf-8') as csvfile:
    reader = csv.reader(csvfile)

    # Skip header row
    next(reader, None)

    rows = []
    for row in reader:
        # Skip empty or incomplete lines
        if not row or len(row) < 15:
            continue

        # Clean up and handle blanks
        values = tuple(
            cell.strip() if cell.strip() != '' else None
            for cell in row[:15]  # Only first 15 columns
        )

        rows.append(values)
    cursor.executemany(insert_query, rows)
    conn.commit()

print(f"✅ Inserted {cursor.rowcount} rows from {CSV_PATH} into '{TABLE_NAME}'.")

# --- CLEANUP ---
cursor.close()
conn.close()
print("✅ Database initialization complete.")

