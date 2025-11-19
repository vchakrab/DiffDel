from datetime import datetime
import mysql.connector
import csv

# --- CONFIG ---
DB_NAME = "ncvoter"
TABLE_NAME = "ncvoter_data"
CSV_PATH = "/DataGeneration/csv_files/ncvoter.csv"
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
create_table_query = f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `voter_id` VARCHAR(20),
  `voter_reg_num` DECIMAL(15,2),
  `name_prefix` VARCHAR(10),
  `first_name` VARCHAR(100),
  `middle_name` VARCHAR(100),
  `last_name` VARCHAR(100),
  `name_suffix` VARCHAR(10),
  `age` INT,
  `gender` CHAR(1),
  `race` CHAR(1),
  `ethnic` VARCHAR(10),
  `street_address` VARCHAR(255),
  `city` VARCHAR(100),
  `state` CHAR(2),
  `zip_code` VARCHAR(10),
  `full_phone_num` VARCHAR(20),
  `birth_place` VARCHAR(50),
  `register_date` DATE,
  `download_month` DATE
);
"""
cursor.execute(create_table_query)
print(f"✅ Table '{TABLE_NAME}' created (if not already).")

# --- PREPARE INSERT QUERY ---
insert_query = f"""
INSERT INTO {TABLE_NAME} (
  `voter_id`, `voter_reg_num`, `name_prefix`, `first_name`, `middle_name`,
  `last_name`, `name_suffix`, `age`, `gender`, `race`,
  `ethnic`, `street_address`, `city`, `state`, `zip_code`,
  `full_phone_num`, `birth_place`, `register_date`, `download_month`
) VALUES (
  %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s,
  %s, %s, %s, %s
);
"""

# --- READ AND CLEAN CSV ---
rows = []
with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip header

    for line_num, row in enumerate(reader, start=2):
        if not row or len(row) < 19:
            print(f"⚠️ Skipping line {line_num} (malformed): {row}")
            continue

        cleaned = []
        for i, cell in enumerate(row[:19]):
            cell = cell.strip().strip('"')
            if cell == "":
                cleaned.append(None)
            elif i == 7:  # age
                try:
                    cleaned.append(int(cell))
                except ValueError:
                    cleaned.append(None)
            elif i == 17:  # register_date
                try:
                    date_obj = datetime.strptime(cell, "%m/%d/%Y")
                    cleaned.append(date_obj.strftime("%Y-%m-%d"))
                except ValueError:
                    cleaned.append(None)
            elif i == 18:  # download_month
                try:
                    # Convert YYYY-MM → YYYY-MM-01 for MySQL DATE
                    if "-" in cell and len(cell) == 7:
                        cell += "-01"
                    cleaned.append(cell)
                except Exception:
                    cleaned.append(None)
            else:
                cleaned.append(cell)
        rows.append(tuple(cleaned))

# --- EXECUTE ---
cursor.executemany(insert_query, rows)
conn.commit()

# --- CLEANUP ---
cursor.close()
conn.close()
print("✅ Database initialization complete.")
