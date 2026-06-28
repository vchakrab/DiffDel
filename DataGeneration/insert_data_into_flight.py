import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mysql.connector
import csv
from config import DB_CONFIG, PROJECT_ROOT

DB_NAME    = "flight"
TABLE_NAME = "flight_data"
CSV_PATH   = PROJECT_ROOT / 'csv_files' / 'flights.csv'
BATCH_SIZE = 1000


def main():
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
    )
    cursor = conn.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cursor.execute(f"USE {DB_NAME}")
    print(f"✅ Database '{DB_NAME}' ready.")

    cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    print(f"✅ Table '{TABLE_NAME}' dropped (if it existed).")

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

    insert_query = f"""
INSERT INTO {TABLE_NAME} (
    Year, Quarter, Month, DayofMonth, DayOfWeek, FlightDate, UniqueCarrier,
    AirlineID, Carrier, TailNum, FlightNum, OriginAirportID, OriginAirportSeqID,
    OriginCityMarketID, Origin, OriginCityName, OriginState, OriginStateFips,
    OriginStateName, OriginWac
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        clean_header = [h.split()[0].strip() for h in header]

        for row in reader:
            if not row:
                continue
            row = [v.strip() if isinstance(v, str) else v for v in row]
            values = []
            for i, val in enumerate(row):
                col_name = clean_header[i]
                if col_name in ["Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "AirlineID", "FlightNum",
                                "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID", "OriginStateFips", "OriginWac"]:
                    values.append(int(val))
                else:
                    values.append(val)
            rows.append(tuple(values))

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        cursor.executemany(insert_query, batch)
        conn.commit()
        print(f"✅ Inserted batch {i + 1} to {i + len(batch)}")

    cursor.close()
    conn.close()
    print("✅ Done.")


if __name__ == '__main__':
    main()
