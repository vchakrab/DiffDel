import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'DataGeneration'))

import mysql.connector
import config
import collect_data, graph
import collect_all_masks_all_data
from DataGeneration.populate_all_datasets import DATASETS as _datasets


def check_mysql():
    try:
        conn = mysql.connector.connect(
            host=config.DB_CONFIG['host'],
            user=config.DB_CONFIG['user'],
            password=config.DB_CONFIG['password'],
        )
        conn.close()
    except mysql.connector.errors.InterfaceError:
        print(
            "\n❌ Cannot reach MySQL. Make sure it is installed and running.\n"
            "\n  Mac:   brew install mysql && brew services start mysql"
            "\n  Linux: sudo apt install mysql-server && sudo systemctl start mysql\n"
            "\nThen re-run: python main.py\n"
        )
        sys.exit(1)
    except mysql.connector.errors.ProgrammingError:
        print(
            "\n❌ MySQL is running but credentials are wrong.\n"
            "   Update DB_CONFIG in config.py with the correct user/password.\n"
        )
        sys.exit(1)


if __name__ == '__main__':
    check_mysql()
    print('Populating datasets...')
    for name, fn in _datasets:
        print(f'  Loading: {name}')
        fn()
    print('Collecting data...')
    collect_data.run_all_experiments()
    print("Collecting additional experiments...")
    collect_data.run_gum_score_ablation()
    collect_data.build_main_data()
    collect_all_masks_all_data.main()
    print('Constructing graphs...')
    graph.graph_all_experiments()
