"""
Run this script to create and populate all MySQL databases in one shot.
Credentials are read from config.py — update DB_CONFIG there before running.

Usage:
    python DataGeneration/populate_all_datasets.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from insert_data_into_adult    import main as populate_adult
from insert_data_into_airport  import main as populate_airport
from insert_data_into_flight   import main as populate_flight
from insert_data_into_hospital import main as populate_hospital
from insert_data_into_tax      import main as populate_tax

DATASETS = [
    ("adult",    populate_adult),
    ("airport",  populate_airport),
    ("flight",   populate_flight),
    ("hospital", populate_hospital),
    ("tax",      populate_tax),
]

if __name__ == '__main__':
    for name, fn in DATASETS:
        print(f"\n{'=' * 50}")
        print(f"  Loading: {name}")
        print(f"{'=' * 50}")
        fn()
    print("\n✅ All datasets loaded successfully.")
