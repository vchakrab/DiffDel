# import random
# import baseline_deletion_2
# import baseline_deletion_3
# import baseline_deletion_1
# import os
# import psutil
#
# hospital_attributes = ["ProviderNumber", "HospitalName", "City", "State", "ZIPCode",
#                            "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
#                            "EmergencyService", "Condition", "MeasureCode", "MeasureName", "Sample",
#                            "StateAvg"]
# tax_attributes = ["fname",
#                   "lname", "gender", "area_code", "phone", "city", "state", "zip", "marital_status", "has_child", "salary", "rate", "single_exemp", "married_exemp", "child_exemp"]
# ncvoter_attributes = [
#     "voter_id",
#     "voter_reg_num",
#     "name_prefix",
#     "first_name",
#     "middle_name",
#     "last_name",
#     "name_suffix",
#     "age",
#     "gender",
#     "race",
#     "ethnic",
#     "street_address",
#     "city",
#     "state",
#     "zip_code",
#     "full_phone_num",
#     "birth_place",
#     "register_date",
#     "download_month"
# ]
# airport_attributes = ['ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
#                     'iso_country', 'iso_region', 'municipality', 'scheduled_service']
# import time;
# sizes = {"airport": 45037, "hospital": 114920, "tax": 99905, "ncvoter": 1001}
#
#
#
# def collect_baseline_2_data_for_all_dbs():
#     data_file_name = "baseline_deletion_2_data_v3.csv"
#     #only choosing attributes with a high number of denial constraints
#
#     for dataset, attrs, in zip(["airport", "hospital", "ncvoter", "tax"], ["latitude_deg", "ProviderNumber", "voter_reg_num", "marital_status"]):
#             with open(data_file_name, mode = 'a') as csv_file:
#                 csv_file.write(f"-----{dataset}-----\n")
#                 csv_file.write("attribute,time,dependencies,cells,depth,space_overhead(B)\n")
#             for i in range(100):
#                 chosen_row = baseline_deletion_2.get_random_key(dataset)
#                 chosen_attr = attrs
#                 start_time = time.time()
#                 dependencies,cells_deleted,memory,depth,init_time,model_time,del_time = baseline_deletion_2.delete_one_path_dependent_cell(chosen_attr, chosen_row, dataset, 0.5)
#                 end_time = time.time() - start_time
#                 with open(data_file_name, mode = 'a') as csv_file:
#                     csv_file.write(f"{chosen_attr},{end_time},{dependencies},{cells_deleted},{depth},{memory},{init_time},{model_time},{del_time}\n")
# collect_baseline_2_data_for_all_dbs()
#
# def collect_baseline_1_data_for_all_dbs():
#     data_file_name = "baseline_deletion_1_data_v3.csv"
#     for dataset, attrs, in zip(["airport", "hospital", "ncvoter", "tax"], ["latitude_deg", "ProviderNumber", "voter_reg_num", "marital_status"]):
#             with open(data_file_name, mode = 'a') as csv_file:
#                 csv_file.write(f"-----{dataset}-----\n")
#                 csv_file.write("attribute,time,dependencies,cells_deleted,depth,memory,init_time,model_time,del_time\n")
#             for i in range(100):
#                 chosen_row = baseline_deletion_2.get_random_key(dataset)
#                 chosen_attr = attrs
#                 start_time = time.time()
#                 depth = 1
#                 dependencies,cells_deleted,memory,init_time,model_time,del_time = baseline_deletion_1.delete_all_dependent_cells(chosen_attr, chosen_row, dataset, 0.8)
#                 end_time = time.time() - start_time
#                 with open(data_file_name, mode = 'a') as csv_file:
#                     csv_file.write(f"{chosen_attr},{end_time},{dependencies},{cells_deleted},{depth},{memory},{init_time},{model_time},{del_time}\n")
# collect_baseline_1_data_for_all_dbs()
import random
import time
import os
import psutil
import baseline_deletion_1
import baseline_deletion_2
import baseline_deletion_3
import mysql.connector
import config


# Dataset attributes definitions
hospital_attributes = [
    "ProviderNumber", "HospitalName", "City", "State", "ZIPCode",
    "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
    "EmergencyService", "Condition", "MeasureCode", "MeasureName",
    "Sample", "StateAvg"
]

tax_attributes = [
    "fname", "lname", "gender", "area_code", "phone", "city", "state",
    "zip", "marital_status", "has_child", "salary", "rate", "single_exemp",
    "married_exemp", "child_exemp"
]

ncvoter_attributes = [
    "voter_id", "voter_reg_num", "name_prefix", "first_name", "middle_name",
    "last_name", "name_suffix", "age", "gender", "race", "ethnic",
    "street_address", "city", "state", "zip_code", "full_phone_num",
    "birth_place", "register_date", "download_month"
]

airport_attributes = [
    'ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
    'iso_country', 'iso_region', 'municipality', 'scheduled_service'
]

# Dataset sizes
sizes = {
    "airport": 45037,
    "hospital": 114920,
    "tax": 99905,
    "ncvoter": 1001
}

DATASETS_TO_RUN = ["airport", "hospital", "ncvoter", "tax"]
ORIGINAL_TABLE_NAMES = {
    "airport": "airports",
    "hospital": "hospital_data",
    "ncvoter": "ncvoter_data",
    "tax": "tax_data"
}


def setup_database_copies():
    """Create a temporary copy of the main table for each dataset."""
    print("Setting up database copies...")
    for dataset in DATASETS_TO_RUN:
        try:
            db_config = config.get_database_config(dataset)
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            
            original_table = ORIGINAL_TABLE_NAMES[dataset]
            copied_table = f"{dataset}_copy_data"
            
            print(f"  - Creating copy for '{dataset}': {copied_table}")
            
            cursor.execute(f"DROP TABLE IF EXISTS {copied_table};")
            cursor.execute(f"CREATE TABLE {copied_table} LIKE {original_table};")
            cursor.execute(f"INSERT INTO {copied_table} SELECT * FROM {original_table};")
            
            # Also copy the insertion time table
            original_time_table = f"{original_table}_insertiontime"
            copied_time_table = f"{copied_table}_insertiontime"
            cursor.execute(f"DROP TABLE IF EXISTS {copied_time_table};")
            cursor.execute(f"CREATE TABLE {copied_time_table} LIKE {original_time_table};")
            cursor.execute(f"INSERT INTO {copied_time_table} SELECT * FROM {original_time_table};")
            
            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"    ERROR for dataset {dataset}: {err}")
    print("Setup complete.\n")


def cleanup_database_copies():
    """Drop the temporary table for each dataset."""
    print("Cleaning up database copies...")
    for dataset in DATASETS_TO_RUN:
        try:
            db_config = config.get_database_config(dataset)
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            copied_table = f"{dataset}_copy_data"
            copied_time_table = f"{copied_table}_insertiontime"
            
            print(f"  - Dropping copy for '{dataset}': {copied_table}")
            
            cursor.execute(f"DROP TABLE IF EXISTS {copied_table};")
            cursor.execute(f"DROP TABLE IF EXISTS {copied_time_table};")
            
            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"    ERROR for dataset {dataset}: {err}")
    print("Cleanup complete.\n")


def collect_baseline_2_data_for_all_dbs():
    """
    Collect data for baseline deletion 2 (one cell per path).
    This corresponds to the MinRet baseline in the P2E2 paper.

    Returns from baseline_deletion_2.delete_one_path_dependent_cell():
        (num_explanations, cells_deleted, memory_bytes, max_depth,
         instantiation_time, model_time, deletion_time)
    """
    data_file_name = "baseline_deletion_2_data_v9.csv"

    # Only choosing attributes with a high number of denial constraints
    datasets = ["airport", "hospital", "ncvoter", "tax"]
    attributes = ["latitude_deg", "ProviderNumber", "voter_reg_num", "marital_status"]

    for dataset, attr in zip(datasets, attributes):
        print(f"Processing baseline 2 for dataset: {dataset}, attribute: {attr}")

        with open(data_file_name, mode = 'a') as csv_file:
            csv_file.write(f"-----{dataset}-----\n")
            csv_file.write(
                "attribute,total_time,num_explanations,cells_deleted,max_depth,memory_bytes,init_time,model_time,del_time\n")

        for i in range(100):
            try:
                chosen_row = baseline_deletion_2.get_random_key(dataset)

                if chosen_row is None:
                    print(f"Warning: Could not get random key for {dataset}, iteration {i}")
                    continue

                start_time = time.time()

                # Returns: (num_explanations, cells_deleted, memory_bytes, max_depth,
                #           instantiation_time, model_time, deletion_time)
                num_explanations, cells_deleted, memory, max_depth, init_time, model_time, del_time = \
                    baseline_deletion_2.delete_one_path_dependent_cell(attr, chosen_row, dataset,
                                                                       0.5)

                total_time = time.time() - start_time

                with open(data_file_name, mode = 'a') as csv_file:
                    csv_file.write(
                        f"{attr},{total_time},{num_explanations},{cells_deleted},"
                        f"{max_depth},{memory},{init_time},{model_time},{del_time}\n"
                    )

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/100 iterations for {dataset}")

            except Exception as e:
                print(f"Error in baseline 2, dataset {dataset}, iteration {i}: {e}")
                continue

        print(f"Completed baseline 2 for {dataset}\n")


def collect_baseline_1_data_for_all_dbs():
    """
    Collect data for baseline deletion 1 (delete all dependent cells).
    This corresponds to the AllDC baseline in the P2E2 paper.

    Returns from baseline_deletion_1.delete_all_dependent_cells():
        (num_constraints, cells_deleted, memory_bytes,
         instantiation_time, model_time, deletion_time)
    """
    data_file_name = "baseline_deletion_1_data_v9.csv"

    datasets = ["airport", "hospital", "ncvoter", "tax"]
    attributes = ["latitude_deg", "ProviderNumber", "voter_reg_num", "marital_status"]

    for dataset, attr in zip(datasets, attributes):
        print(f"Processing baseline 1 for dataset: {dataset}, attribute: {attr}")

        with open(data_file_name, mode = 'a') as csv_file:
            csv_file.write(f"-----{dataset}-----\n")
            csv_file.write(
                "attribute,total_time,num_constraints,cells_deleted,max_depth,memory_bytes,init_time,model_time,del_time\n")

        for i in range(100):
            try:
                chosen_row = baseline_deletion_2.get_random_key(dataset)

                if chosen_row is None:
                    print(f"Warning: Could not get random key for {dataset}, iteration {i}")
                    continue

                start_time = time.time()

                # Returns: (num_constraints, cells_deleted, memory_bytes,
                #           instantiation_time, model_time, deletion_time)
                num_constraints, cells_deleted, memory, init_time, model_time, del_time = \
                    baseline_deletion_1.delete_all_dependent_cells(attr, chosen_row, dataset, 0.8)

                total_time = time.time() - start_time
                max_depth = 1
                with open(data_file_name, mode = 'a') as csv_file:
                    csv_file.write(
                        f"{attr},{total_time},{num_constraints},{cells_deleted},"
                        f"{max_depth},{memory},{init_time},{model_time},{del_time}\n"
                    )

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/100 iterations for {dataset}")

            except Exception as e:
                print(f"Error in baseline 1, dataset {dataset}, iteration {i}: {e}")
                continue

        print(f"Completed baseline 1 for {dataset}\n")

def collect_baseline_deletion_data_3():
    """
    Collect data for baseline deletion 3 (ILP Java-style).
    This corresponds to the ILP-based approach in the P2E2 paper.

    Returns from baseline_deletion_3.delete_ilp_java_style():
        (num_explanations, cells_deleted, memory_bytes, max_depth,
         instantiation_time, model_time, deletion_time)
    """
    data_file_name = "baseline_deletion_3_data_v9.csv"

    datasets = ["airport", "hospital", "ncvoter", "tax"]
    attributes = ["latitude_deg", "ProviderNumber", "voter_reg_num", "marital_status"]

    for dataset, attr in zip(datasets, attributes):
        print(f"Processing baseline 3 for dataset: {dataset}, attribute: {attr}")

        with open(data_file_name, mode='a') as csv_file:
            csv_file.write(f"-----{dataset}-----\n")
            csv_file.write(
                "attribute,total_time,num_explanations,cells_deleted,max_depth,memory_bytes,init_time,model_time,del_time\n")

        for i in range(100):
            try:
                chosen_row = baseline_deletion_2.get_random_key(dataset)

                if chosen_row is None:
                    print(f"Warning: Could not get random key for {dataset}, iteration {i}")
                    continue

                start_time = time.time()

                # Returns: (num_explanations, cells_deleted, memory_bytes, max_depth,
                #           instantiation_time, model_time, deletion_time)
                num_explanations, cells_deleted, memory, max_depth, init_time, model_time, del_time = \
                    baseline_deletion_3.baseline_deletion_3(attr, chosen_row, dataset, 5.0)

                total_time = time.time() - start_time

                with open(data_file_name, mode='a') as csv_file:
                    csv_file.write(
                        f"{attr},{total_time},{num_explanations},{cells_deleted},"
                        f"{max_depth},{memory},{init_time},{model_time},{del_time}\n"
                    )

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/100 iterations for {dataset}")

            except Exception as e:
                print(f"Error in baseline 3, dataset {dataset}, iteration {i}: {e}")
                continue

        print(f"Completed baseline 3 for {dataset}\n")

def main():
    """
    Main function to run all data collection.
    """
    print("=" * 60)
    print("P2E2 Baseline Data Collection")
    print("=" * 60)
    print()

    # --- Baseline 1 (AllDC) ---
    print("Starting Baseline 1 (AllDC - Delete All Dependent Cells)...")
    setup_database_copies()
    collect_baseline_1_data_for_all_dbs()
    cleanup_database_copies()
    print("Finished Baseline 1.\n")

    # --- Baseline 2 (MinRet) ---
    print("=" * 60)
    print("Starting Baseline 2 (MinRet - Delete One Cell Per Path)...")
    setup_database_copies()
    collect_baseline_2_data_for_all_dbs()
    cleanup_database_copies()
    print("Finished Baseline 2.\n")
    
    # --- Baseline 3 (ILP) ---
    print("=" * 60)
    print("Starting Baseline 3 (ILP - Java Style)...")
    setup_database_copies()
    collect_baseline_deletion_data_3()
    cleanup_database_copies()
    print("Finished Baseline 3.\n")

    print("=" * 60)
    print("Data collection completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
