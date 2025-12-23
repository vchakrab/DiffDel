import mysql.connector
import csv

#ALTER TABLE flight_data
#ADD COLUMN id INT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST

# =====================
# CONFIG
# =====================
DB_NAME = "flight"
TABLE_NAME = "flight_data"
CSV_PATH = "/Users/adhariya/src/DiffDel/csv_files/flights.csv"  # <-- replace with your CSV path
USER = "root"
PASSWORD = "my_password"  # <-- replace with your MySQL password
HOST = "localhost"
BATCH_SIZE = 1000  # number of rows per batch insert

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
    Year INT,
    Quarter INT,
    Month INT,
    DayofMonth INT,
    DayOfWeek INT,
    FlightDate DATE,
    UniqueCarrier VARCHAR(10),
    AirlineID INT,
    Carrier VARCHAR(10),
    TailNum VARCHAR(20),
    FlightNum INT,
    OriginAirportID INT,
    OriginAirportSeqID INT,
    OriginCityMarketID INT,
    Origin VARCHAR(10),
    OriginCityName VARCHAR(100),
    OriginState VARCHAR(10),
    OriginStateFips INT,
    OriginStateName VARCHAR(50),
    OriginWac INT
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
    Year, Quarter, Month, DayofMonth, DayOfWeek, FlightDate, UniqueCarrier,
    AirlineID, Carrier, TailNum, FlightNum, OriginAirportID, OriginAirportSeqID,
    OriginCityMarketID, Origin, OriginCityName, OriginState, OriginStateFips,
    OriginStateName, OriginWac
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# =====================
# READ CSV AND INSERT
# =====================
rows = []

with open(CSV_PATH, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    # Read header and clean it
    header = next(reader)
    clean_header = [h.split()[0].strip() for h in header]  # remove types and spaces

    for row in reader:
        if not row:
            continue
        # Strip spaces from all values
        row = [v.strip() if isinstance(v, str) else v for v in row]
        values = []
        for i, val in enumerate(row):
            col_name = clean_header[i]
            if col_name in ["Year","Quarter","Month","DayofMonth","DayOfWeek","AirlineID","FlightNum",
                            "OriginAirportID","OriginAirportSeqID","OriginCityMarketID","OriginStateFips","OriginWac"]:
                values.append(int(val))
            elif col_name == "FlightDate":
                values.append(val)  # MySQL DATE can take 'YYYY-MM-DD' string
            else:
                values.append(val)
        rows.append(tuple(values))

# =====================
# BATCH INSERT
# =====================
for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i+BATCH_SIZE]
    cursor.executemany(insert_query, batch)
    conn.commit()
    print(f"✅ Inserted batch {i+1} to {i+len(batch)}")

# =====================
# CLEANUP
# =====================
cursor.close()
conn.close()
print("✅ Database initialization complete.")
