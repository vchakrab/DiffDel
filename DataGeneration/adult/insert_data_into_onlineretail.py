import mysql.connector
import csv
from datetime import datetime

# =====================
# CONFIG
# =====================
DB_NAME = "online_retail_db"
TABLE_NAME = "OnlineRetail_data"
CSV_PATH = "/Users/adhariya/src/DiffDel/csv_files/OnlineRetail.csv"  # <-- change path
USER = "root"
PASSWORD = "my_password"  # <-- change password
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
    InvoiceNo    VARCHAR(15) NOT NULL,
    StockCode    VARCHAR(30) NOT NULL,
    Description  TEXT,
    Quantity     INT NOT NULL,
    InvoiceDate  DATETIME NOT NULL,
    UnitPrice    DECIMAL(10,2) NOT NULL,
    CustomerID   VARCHAR(10),
    Country      VARCHAR(50) NOT NULL
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
    InvoiceNo, StockCode, Description, Quantity,
    InvoiceDate, UnitPrice, CustomerID, Country
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
"""

# =====================
# READ CSV AND INSERT
# =====================
rows = []

with open(CSV_PATH, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    # Normalize headers
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    for row in reader:
        # Normalize keys
        row = {k.strip(): v for k, v in row.items()}

        # Skip invalid rows
        if not row.get("InvoiceNo") or not row.get("Quantity") or not row.get("InvoiceDate"):
            continue

        try:
            # Parse InvoiceDate: MM/DD/YY HH:MM
            invoice_date = datetime.strptime(row["InvoiceDate"].strip(), "%m/%d/%y %H:%M")
        except ValueError:
            # Skip rows with invalid dates
            continue

        values = (
            row["InvoiceNo"].strip(),
            row["StockCode"].strip(),
            row["Description"].strip(),
            int(row["Quantity"]),
            invoice_date,
            float(row["UnitPrice"]),
            row["CustomerID"].strip() if row.get("CustomerID") else None,
            row["Country"].strip()
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
