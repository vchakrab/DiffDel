import mysql.connector
import csv

# --- CONFIG ---
DB_NAME = "tax"
TABLE_NAME = "tax_data"
CSV_PATH = "/Users/adhariya/src/DiffDel/csv_files/tax.csv"
USER = "root"
PASSWORD = "my_password"
HOST = "localhost"

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    return float(s) if s not in ("", "NA", "None", "#") else None

def to_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "NA", "None", "#"):
        return None
    # handles "12.0" safely too
    return int(float(s))

conn = mysql.connector.connect(
    host=HOST,
    user=USER,
    password=PASSWORD,
    autocommit=False,
)
cursor = conn.cursor()

try:
    # --- CREATE DATABASE + USE IT ---
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
    cursor.execute(f"USE `{DB_NAME}`")

    # --- CREATE TABLE (YOU FORGOT TO EXECUTE THIS) ---
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
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
    cursor.execute(create_table_query)
    conn.commit()
    print(f"✅ Table `{DB_NAME}`.`{TABLE_NAME}` created (if not already).")

    insert_query = f"""
    INSERT INTO `{TABLE_NAME}` (
      `fname`, `lname`, `gender`, `area_code`, `phone`,
      `city`, `state`, `zip`, `marital_status`, `has_child`,
      `salary`, `rate`, `single_exemp`, `married_exemp`, `child_exemp`
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    rows = []
    skipped = 0

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = [
            "FName", "LName", "Gender", "AreaCode", "Phone",
            "City", "State", "Zip", "MaritalStatus", "HasChild",
            "Salary", "Rate", "SingleExemp", "MarriedExemp", "ChildExemp"
        ]

        for r in reader:
            if any(k not in r for k in required):
                skipped += 1
                continue

            values = (
                (r["FName"] or "").strip(),
                (r["LName"] or "").strip(),
                (r["Gender"] or "").strip(),
                (r["AreaCode"] or "").strip(),
                (r["Phone"] or "").strip(),
                (r["City"] or "").strip(),
                (r["State"] or "").strip(),
                (r["Zip"] or "").strip(),
                (r["MaritalStatus"] or "").strip(),
                (r["HasChild"] or "").strip(),
                to_int(r["Salary"]),         # <-- INT
                to_float(r["Rate"]),         # <-- DECIMAL
                to_int(r["SingleExemp"]),
                to_int(r["MarriedExemp"]),
                to_int(r["ChildExemp"]),
            )

            # basic guard: skip totally empty lines
            if all(v in ("", None) for v in values):
                skipped += 1
                continue

            rows.append(values)

    if not rows:
        print(f"⚠️ No rows parsed from CSV. Skipped={skipped}. Check header names in the CSV.")
    else:
        cursor.executemany(insert_query, rows)
        conn.commit()
        print(f"✅ Inserted {len(rows)} rows into `{DB_NAME}`.`{TABLE_NAME}` (skipped {skipped}).")

        # quick sanity check
        cursor.execute(f"SELECT COUNT(*) FROM `{TABLE_NAME}`")
        count = cursor.fetchone()[0]
        print(f"📌 Table now contains {count} rows.")

except mysql.connector.Error as e:
    conn.rollback()
    print("❌ MySQL error:", e)
    raise
finally:
    cursor.close()
    conn.close()
