import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mysql.connector
import csv
from config import DB_CONFIG, PROJECT_ROOT

DB_NAME    = "adult"
TABLE_NAME = "adult_data"
CSV_PATH   = PROJECT_ROOT / 'csv_files' / 'adult.csv'


def main():
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
    )
    cursor = conn.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cursor.execute(f"USE {DB_NAME}")

    cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

    create_table_query = f"""
CREATE TABLE {TABLE_NAME} (
    id INT AUTO_INCREMENT PRIMARY KEY,
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

    rows = []
    required_numeric_fields = [
        "age", "fnlwgt", "education_num",
        "capital_gain", "capital_loss", "hours_per_week",
    ]

    with open(CSV_PATH, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        # Strip type annotations and normalize: "Education-num int" → "education_num"
        clean_header = [h.split()[0].strip().lower().replace('-', '_') for h in reader.fieldnames]
        reader.fieldnames = clean_header

        for i, raw_row in enumerate(reader):
            row = {k.strip(): v.strip() for k, v in raw_row.items() if v is not None}

            missing = [f for f in required_numeric_fields if not row.get(f)]
            if missing:
                continue

            try:
                values = (
                    int(row["age"]),
                    row.get("workclass"),
                    int(row["fnlwgt"]),
                    row.get("education"),
                    int(row["education_num"]),
                    row.get("marital_status"),
                    row.get("occupation"),
                    row.get("relationship"),
                    row.get("race"),
                    row.get("sex"),
                    int(row["capital_gain"]),
                    int(row["capital_loss"]),
                    int(row["hours_per_week"]),
                    row.get("native_country"),
                    row.get("class"),
                )
                rows.append(values)
            except (ValueError, KeyError) as e:
                continue


    if not rows:
        cursor.close()
        conn.close()
        return

    insert_query = f"""
INSERT INTO {TABLE_NAME} (
    age, workclass, fnlwgt, education, education_num,
    marital_status, occupation, relationship, race, sex,
    capital_gain, capital_loss, hours_per_week, native_country, class
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

    try:
        cursor.executemany(insert_query, rows)
        conn.commit()
    except mysql.connector.Error as e:
        conn.rollback()
        raise

    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()