import mysql.connector
import csv

# =====================
# CONFIG
# =====================
DB_NAME = "adult"
TABLE_NAME = "adult_data"
CSV_PATH = "/Users/adhariya/src/DiffDel/csv_files/adult.csv"  # <-- replace with your CSV path
USER = "root"
PASSWORD = "my_password"  # <-- replace with your MySQL password
HOST = "localhost"

# =====================
# CONNECT TO MYSQL
# =====================
conn = mysql.connector.connect(
    host=HOST,
    user=USER,
    password=PASSWORD
)
cursor = conn.cursor()

# =====================
# CREATE DATABASE
# =====================
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
cursor.execute(f"USE {DB_NAME}")
print(f"✅ Database '{DB_NAME}' ready.")

# =====================
# DROP TABLE IF EXISTS
# =====================
cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
print(f"✅ Table '{TABLE_NAME}' dropped (if it existed).")

# =====================
# CREATE TABLE
# =====================
create_table_query = f"""
CREATE TABLE {TABLE_NAME} (
    age INT,
    workclass VARCHAR(50),
    fnlwgt INT,
    education VARCHAR(50),
    education_num INT,
    marital_status VARCHAR(50),
    occupation VARCHAR(50),
    relationship VARCHAR(50),
    race VARCHAR(50),
    sex VARCHAR(10),
    capital_gain INT,
    capital_loss INT,
    hours_per_week INT,
    native_country VARCHAR(50),
    class VARCHAR(10)
);
"""
cursor.execute(create_table_query)
conn.commit()
print(f"✅ Table '{TABLE_NAME}' created.")

# =====================
# INSERT QUERY
# =====================
insert_query = f"""
INSERT INTO {TABLE_NAME} (
    age, workclass, fnlwgt, education, education_num,
    marital_status, occupation, relationship, race, sex,
    capital_gain, capital_loss, hours_per_week, native_country, class
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# =====================
# READ CSV AND INSERT
# =====================
rows = []

required_numeric_fields = ["age","fnlwgt","Education-num","capital-gain","capital-loss","Hours-per-week"]

with open(CSV_PATH, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    # Strip headers
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    for row in reader:
        # Strip keys and values
        row = {k.strip(): v.strip() for k, v in row.items() if v is not None}

        # Skip rows with missing numeric fields
        if any(not row.get(f) for f in required_numeric_fields):
            continue

        values = (
            int(row["age"]),
            row.get("workclass"),
            int(row["fnlwgt"]),
            row.get("education"),
            int(row["Education-num"]),
            row.get("Marital-status"),
            row.get("occupation"),
            row.get("relationship"),
            row.get("race"),
            row.get("sex"),
            int(row["capital-gain"]),
            int(row["capital-loss"]),
            int(row["Hours-per-week"]),
            row.get("Native-country"),
            row.get("class")
        )
        rows.append(values)

# Bulk insert
if rows:
    cursor.executemany(insert_query, rows)
    conn.commit()
    print(f"✅ Inserted {cursor.rowcount} rows from {CSV_PATH} into '{TABLE_NAME}'.")

# =====================
# CLEANUP
# =====================
cursor.close()
conn.close()
print("✅ Database initialization complete.")
