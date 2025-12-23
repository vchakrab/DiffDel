
import unittest
import mysql.connector
import sys
import os

# Add the project root to the Python path to allow importing baseline_deletion_3
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from old_files.baseline_deletion_3 import ilp_approach_matching_java, Cell
    from config import DB_CONFIG
    GUROBI_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Failed to import necessary modules: {e}")
    GUROBI_AVAILABLE = False


@unittest.skipIf(not GUROBI_AVAILABLE, "Gurobi or other required modules are not available")
class TestILPJavaStyle(unittest.TestCase):

    def setUp(self):
        """Set up a temporary database and tables for testing."""
        self.db_config = DB_CONFIG.copy()
        # Connect to MySQL server, not a specific database initially
        self.db_config.pop('database', None)
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.cursor = self.conn.cursor()
        except mysql.connector.Error as err:
            self.fail(f"Failed to connect to MySQL: {err}")

        self.test_db_name = "test_diffdel_db"
        self.table_name = "test_data_copy"
        
        # Drop DB if exists and create a fresh one
        self.cursor.execute(f"DROP DATABASE IF EXISTS {self.test_db_name}")
        self.cursor.execute(f"CREATE DATABASE {self.test_db_name}")
        self.cursor.execute(f"USE {self.test_db_name}")

        # Create data table
        self.cursor.execute(f"""
            CREATE TABLE {self.table_name} (
                id INT PRIMARY KEY,
                attr1 VARCHAR(255),
                attr2 VARCHAR(255),
                attr3 VARCHAR(255)
            )
        """)

        # Create insertion time table
        self.cursor.execute(f"""
            CREATE TABLE {self.table_name}_insertiontime (
                insertionKey INT PRIMARY KEY,
                attr1 BIGINT,
                attr2 BIGINT,
                attr3 BIGINT
            )
        """)
        self.conn.commit()

    def tearDown(self):
        """Clean up the temporary database."""
        self.cursor.execute(f"DROP DATABASE IF EXISTS {self.test_db_name}")
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def test_simple_deletion_scenario(self):
        """Test a basic scenario where one deletion forces another."""
        # -- 1. Populate Data --
        # Data: A single row with id=1
        self.cursor.execute(f"""
            INSERT INTO {self.table_name} (id, attr1, attr2, attr3)
            VALUES (1, 'A', 'B', 'C')
        """)
        # Insertion Times: All cells inserted at different times
        self.cursor.execute(f"""
            INSERT INTO {self.table_name}_insertiontime (insertionKey, attr1, attr2, attr3)
            VALUES (1, 100, 110, 120)
        """)
        self.conn.commit()

        # -- 2. Define Test Inputs --
        key = 1
        target_attr = "attr1"
        target_time = 100  # Target cell's insertion time

        # Hypergraph representing a single DC: {attr1, attr2, attr3}
        # If attr1 is deleted, at least one of {attr2, attr3} must also be deleted.
        hypergraph = {
            frozenset({f't1.attr1', f't1.attr2', f't1.attr3'}): 1.0
        }
        
        # Boundaries (not directly used in this specific ILP function but good practice)
        boundaries = set()

        # -- 3. Run the ILP Solver --
        to_del, model_time, instantiation_time, max_depth, num_cells, activated_dependencies_count = ilp_approach_matching_java(
            self.cursor,
            self.table_name,
            key,
            target_attr,
            target_time,
            hypergraph,
            boundaries
        )

        # -- 4. Assert the Outcome --
        self.assertEqual(len(to_del), 3, "Should delete exactly three cells")
        self.assertIn(Cell(attribute='t1.attr1', key=1), to_del, "Target cell must be deleted")

        # Check that one of the other two cells was also deleted
        deleted_attr2 = Cell(attribute='t1.attr2', key=1) in to_del
        deleted_attr3 = Cell(attribute='t1.attr3', key=1) in to_del
        self.assertTrue(deleted_attr2 or deleted_attr3, "Either attr2 or attr3 must be deleted")
        
        # Assert activated dependencies: 1 RDR should be activated
        self.assertEqual(activated_dependencies_count, 1, "Should activate 1 dependency")
        
        # Assert memory bytes: 3 cells * 20 bytes/cell (placeholder)
        self.assertEqual(num_cells * 20, 60, "Memory bytes should be 60 (3 cells * 20)")
        
        # Assert max depth: 1 (target is 0, direct dependencies are 1)
        self.assertEqual(max_depth, 1, "Max depth should be 1")

        print(f"\nTest finished successfully.")
        print(f"  - Total cells deleted: {len(to_del)}")
        print(f"  - Activated dependencies: {activated_dependencies_count}")
        print(f"  - Memory bytes: {num_cells * 20}")
        print(f"  - Max depth: {max_depth}")
        print(f"  - Instantiation time: {instantiation_time:.4f}s")
        print(f"  - Model optimization time: {model_time:.4f}s")
        # Note: deletion_time is not returned by ilp_approach_matching_java directly



if __name__ == '__main__':
    if GUROBI_AVAILABLE:
        unittest.main()
    else:
        print("Skipping tests because required modules (e.g., Gurobi) are not installed.")
