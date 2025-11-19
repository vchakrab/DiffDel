import mysql.connector
import csv

# --- CONFIG ---
DB_NAME = "tax"
TABLE_NAME = "tax_data"
CSV_PATH = "/DataGeneration/csv_files/tax.csv"  # <-- change this path
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
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `fname` VARCHAR(100),
  `lname` VARCHAR(100),
  `gender` CHAR(1),
  `area_code` CHAR(3),
  `phone` VARCHAR(20),
  `city` VARCHAR(100),
  `state` CHAR(2),
  `zip` VARCHAR(10),
  `marital_status` CHAR(1),
  `has_child` CHAR(1),
  `salary` INT,
  `rate` DECIMAL(8,6),
  `single_exemp` INT,
  `married_exemp` INT,
  `child_exemp` INT
);

"""
print(f"✅ Table '{TABLE_NAME}' created (if not already).")

# --- PREPARE INSERT QUERY ---
insert_query = f"""
INSERT INTO {TABLE_NAME} (
  `fname`, `lname`, `gender`, `area_code`, `phone`,
  `city`, `state`, `zip`, `marital_status`, `has_child`,
  `salary`, `rate`, `single_exemp`, `married_exemp`, `child_exemp`
) VALUES (
  %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s
);
"""

# --- READ CSV AND INSERT ---

with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, quotechar='"')
    next(reader, None)  # Skip header row

    rows = []
    for row in reader:
        if not row or len(row) < 15:
            print(f"⚠️ Skipping incomplete row: {row}")
            continue

        # Clean up and convert types
        fname, lname, gender, area_code, phone, city, state, zip_code, marital_status, has_child, salary, rate, single_exemp, married_exemp, child_exemp = row

        # Convert numeric fields safely
        def to_float(x): return float(x) if x.strip() not in ("", "NA", "None", "#") else None
        def to_int(x): return int(float(x)) if x.strip() not in ("", "NA", "None", "#") else None

        values = (
            fname.strip(),
            lname.strip(),
            gender.strip(),
            area_code.strip(),
            phone.strip(),
            city.strip(),
            state.strip(),
            zip_code.strip(),
            marital_status.strip(),
            has_child.strip(),
            to_float(salary),
            to_float(rate),
            to_int(single_exemp),
            to_int(married_exemp),
            to_int(child_exemp),
        )

        rows.append(values)

# === EXECUTE INSERT ===
cursor.executemany(insert_query, rows)
conn.commit()
print(f"✅ Inserted {cursor.rowcount} rows from {CSV_PATH} into '{TABLE_NAME}'.")

# --- CLEANUP ---
cursor.close()
conn.close()
print("✅ Database initialization complete.")


