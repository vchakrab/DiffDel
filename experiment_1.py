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
import time
from old_files import baseline_deletion_2, baseline_deletion_1
import baseline_deletion_3
import exponential_deletion
from differentialprivacyalgorithms import greedy_gumbel
import two_phase_deletion
import mysql.connector
import config
from exponential_deletion import clean_raw_dcs, find_inference_paths_str, calculate_leakage_str


# --- Helper for new metric calculation ---
def _count_active_paths(hyperedges, paths, mask, initial_known):
    """Helper to count how many paths are NOT blocked by a mask."""
    active_paths = []
    for path in paths:
        is_blocked = False
        known_so_far = initial_known - mask
        for edge_idx in path:
            edge = hyperedges[edge_idx]
            unknown_in_edge = [c for c in edge if c not in known_so_far]
            if len(unknown_in_edge) == 1:
                known_so_far.add(unknown_in_edge[0])
            elif len(unknown_in_edge) > 1:
                is_blocked = True
                break
        if not is_blocked:
            active_paths.append(path)
    return len(active_paths)



# Dataset sizes


DATASETS_TO_RUN = ["airport", "hospital", "ncvoter", "tax", 'flights', 'Onlineretail', 'adult']
ORIGINAL_TABLE_NAMES = {
    "airport": "airports",
    "hospital": "hospital_data",
    "ncvoter": "ncvoter_data",
    "tax": "tax_data",
    "adult": 'adult_data',
    'Onlineretail': 'onlineretail_data',
    'flights': 'flight_data'
}

def get_random_key(dataset: str):
    """
    Get a random key from the dataset.

    Parameters:
        dataset (str): Name of the dataset

    Returns:
        int: Random ID from the dataset
    """
    db_details = config.get_database_config(dataset)

    conn = mysql.connector.connect(
        host = db_details['host'],
        user = db_details['user'],
        password = db_details['password'],
        database = db_details['database'],
        ssl_disabled = db_details['ssl_disabled']
    )

    if not conn.is_connected():
        return None

    cursor = conn.cursor()
    try:
        cursor.execute(f"""SELECT ID
                       FROM {dataset + "_copy_data"}
                       ORDER BY RAND()
                       LIMIT 1;
        """)
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        cursor.close()
        conn.close()

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
    data_file_name = "old_files/baseline_deletion_2_data_v9.csv"

    # Only choosing attributes with a high number of denial constraints
    datasets = ["airport", "hospital", "ncvoter", "tax", 'flights', 'OnlineRetail', 'adult']
    attributes = ["latitude_deg", "ProviderNumber", "voter_reg_num", "marital_status", "FlightNum",
                      "InvoiceNo", "education"]

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
    data_file_name = "old_files/baseline_deletion_1_data_v9.csv"

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
    Collect data for baseline deletion 3 (ILP Java-style), now with leakage and paths_blocked.
    """
    data_file_name = "delmin_data_v12.csv"  # New version for new data

    datasets = ['Onlineretail']
    attributes = ["InvoiceNo"]

    for dataset, attr in zip(datasets, attributes):
        print(f"Processing baseline 3 for dataset: {dataset}, attribute: {attr}")

        try:
            if dataset == 'ncvoter':
                dataset_module_name = 'NCVoter'
            else:
                dataset_module_name = dataset.capitalize()
            dc_module_path = f'DCandDelset.dc_configs.top{dataset_module_name}DCs_parsed'
            dc_module = __import__(dc_module_path, fromlist=['denial_constraints'])
            raw_dcs = dc_module.denial_constraints
            hyperedges = clean_raw_dcs(raw_dcs)
            edge_weights = {i: 0.8 for i in range(len(hyperedges))}
        except ImportError:
            print(f"Could not load DCs for {dataset}. Skipping.")
            continue
        
        all_attributes = set(attr for he in hyperedges for attr in he)
        initial_known_for_path_finding = set()
        all_paths = find_inference_paths_str(hyperedges, attr, initial_known_for_path_finding)
        total_paths = len(all_paths)

        with open(data_file_name, mode='a') as csv_file:
            csv_file.write(f"-----{dataset}-----\n")
            csv_file.write(
                "attribute,total_time,init_time,model_time,del_time,leakage,paths_blocked,mask_size,num_paths,memory_overhead_bytes,num_instantiated_cells\n")

        for i in range(100):
            try:
                chosen_row = get_random_key(dataset)
                if chosen_row is None: continue

                start_time = time.time()
                
                num_explanations, deleted_cells_set, memory, max_depth, init_time, model_time, del_time, num_instantiated = \
                    baseline_deletion_3.baseline_deletion_3(attr, chosen_row, dataset, 5.0)

                total_time = time.time() - start_time
                
                mask = deleted_cells_set
                mask_size = len(mask)
                leakage = calculate_leakage_str(hyperedges, all_paths, mask, attr, initial_known_for_path_finding, edge_weights)
                active_paths = _count_active_paths(hyperedges, all_paths, mask, initial_known_for_path_finding)
                paths_blocked = total_paths - active_paths

                with open(data_file_name, mode='a') as csv_file:
                    csv_file.write(
                        f"{attr},{total_time},{init_time},{model_time},{del_time},"
                        f"{leakage},{paths_blocked},{mask_size},{total_paths},{memory},{num_instantiated}\n"
                    )

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/100 iterations for {dataset}")

            except Exception as e:
                print(f"Error in baseline 3, dataset {dataset}, iteration {i+1}: {e}")
                continue
        print(f"Completed baseline 3 for {dataset}\n")

def collect_exponential_deletion_data():
    """
    Collect data for the string-based exponential deletion algorithm.
    """
    data_file_name = "delexp_data_v12.csv"
    datasets = [
        #"airport", "hospital", "ncvoter", "tax", 'flights',
        'Onlineretail']
    attributes = ["InvoiceNo"]


    with open(data_file_name, mode='w', newline='') as csv_file: pass

    for dataset, attr in zip(datasets, attributes):
        print(f"Processing Exponential Deletion for dataset: {dataset}, attribute: {attr}")
        with open(data_file_name, mode='a', newline='') as csv_file:
            csv_file.write(f"-----{dataset}-----\n")
            header = "target_attribute,total_time,init_time,model_time,del_time,leakage,utility,mask_size,num_paths,memory_overhead_bytes,num_instantiated_cells\n"
            csv_file.write(header)
        for i in range(100):
            try:
                chosen_row = get_random_key(dataset)
                if chosen_row is None: continue

                results = exponential_deletion.exponential_deletion_main(dataset=dataset, key=chosen_row, target_cell=attr)
                
                if results:
                    total_time = results['init_time'] + results['model_time'] + results['del_time']
                    with open(data_file_name, mode='a', newline='') as csv_file:
                        csv_row = (
                            f"{attr},{total_time},{results['init_time']},{results['model_time']},"
                            f"{results['del_time']},{results['leakage']},{results['utility']},"
                            f"{results['mask_size']},{results['num_paths']},{results['memory_overhead_bytes']},{results['num_instantiated_cells']}\n"
                        )
                        csv_file.write(csv_row)
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/100 iterations for {dataset}")
            except Exception as e:
                print(f"Error in exponential deletion experiment, dataset {dataset}, iteration {i+1}: {e}")
                continue
        print(f"Completed exponential deletion for {dataset}\n")


def collect_2phase_deletion_data():
    """
    Collect data for the 2-Phase deletion algorithm.
    """
    data_file_name = "2phase_deletion_data_v12.csv"
    datasets = ["airport", "hospital", "ncvoter", "tax", 'flights', 'Onlineretail', 'adult']
    attributes_map = {
        # "airport": "latitude_deg",
        # "hospital": "ProviderNumber",
        # "ncvoter": "voter_reg_num",
        "OnlineRetail": "InvoiceNo",
        # "adult": "education"
    }

    # --- Perform Offline Phase for all datasets and attributes first ---
    all_templates = {}
    for dataset, attr in attributes_map.items():
        # This will build the template if it doesn't exist, and load it.
        templates = two_phase_deletion.offline_precomputation(dataset, [attr], force_rebuild=True)
        all_templates.update(templates)

    with open(data_file_name, mode='w', newline='') as csv_file: pass  # Clear file

    for dataset, attr in attributes_map.items():
        print(f"--- Starting 2-Phase Online Experiment for Dataset: {dataset} ---")
        
        with open(data_file_name, mode='a', newline='') as csv_file:
            csv_file.write(f"-----{dataset}-----\n")
            header = "target_attribute,total_time,init_time,model_time,del_time,leakage,utility,paths_blocked,mask_size,num_paths,memory_overhead_bytes,num_instantiated_cells\n"
            csv_file.write(header)

        for i in range(100):
            try:
                chosen_row = get_random_key(dataset)
                if chosen_row is None: continue
                
                start_time = time.time()
                # Pass the pre-loaded templates to the main function
                results = two_phase_deletion.two_phase_deletion_main(
                    dataset=dataset, key=chosen_row, target_cell=attr, templates={attr: all_templates[attr]}
                )
                total_time = time.time() - start_time
                
                if results:
                    paths_blocked = results['paths_blocked']
                    with open(data_file_name, mode='a', newline='') as csv_file:
                        csv_row = (
                            f"{attr},{total_time},{results['init_time']},{results['model_time']},"
                            f"{results['del_time']},{results['leakage']},{results['utility']},{paths_blocked},"
                            f"{results['mask_size']},{results['num_paths']},{results['memory_overhead_bytes']},{results['num_instantiated_cells']}\n"
                        )
                        csv_file.write(csv_row)

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/100 iterations for {dataset}")

            except Exception as e:
                print(f"Error in 2-phase experiment, dataset {dataset}, iteration {i+1}: {e}")
                continue
        print(f"Completed 2-phase experiment for {dataset}\n")

def main():
    """
    Main function to run all data collection.
    """
    print("=" * 60)
    print("P2E2 Baseline Data Collection")
    print("=" * 60)
    print()
    #
    # --- Baseline 3 (ILP) ---
    print("=" * 60)
    print("Starting Baseline 3 (ILP - Java Style)...")
    setup_database_copies()
    collect_baseline_deletion_data_3()
    cleanup_database_copies()
    print("Finished Baseline 3.\n")

    # # --- Exponential Deletion (Our Method) ---
    # print("=" * 60)
    # print("Starting Exponential Deletion (String-Based)...")
    # setup_database_copies()
    # collect_exponential_deletion_data()
    # cleanup_database_copies()
    # print("Finished Exponential Deletion.\n")
    # #
    # # --- Greedy Gumbel (Our Method) ---
    # print("=" * 60)
    # print("Starting Greedy Gumbel (String-Based)...")
    # setup_database_copies()
    # collect_greedy_gumbel_data()
    # cleanup_database_copies()
    # print("Finished Greedy Gumbel.\n")

    # # --- 2-Phase Deletion (Our Method) ---
    # print("=" * 60)
    # print("Starting 2-Phase Deletion...")
    # setup_database_copies()
    # collect_2phase_deletion_data()
    # cleanup_database_copies()
    # print("Finished 2-Phase Deletion.\n")
    #
    # print("=" * 60)
    # print("Data collection completed!")
    # print("=" * 60)

def collect_greedy_gumbel_data(epsilon: float):
    """
    Collect data for the string-based greedy gumbel algorithm.
    """
    data_file_name = f"delgum_data_epsilon_leakage_graph_{epsilon}.csv"
    datasets = ["airport", "hospital", "ncvoter", "Onlineretail", "adult"]
    attributes = ["latitude_deg", "ProviderNumber", "voter_reg_num", "InvoiceNo", "education"]

    with open(data_file_name, mode='w', newline='') as csv_file: pass

    for dataset, attr in zip(datasets, attributes):
        print(f"Processing Greedy Gumbel for dataset: {dataset}, attribute: {attr}")
        with open(data_file_name, mode='a', newline='') as csv_file:
            csv_file.write(f"-----{dataset}-----\n")
            header = "target_attribute,total_time,init_time,model_time,del_time,leakage,utility,mask_size,num_paths,memory_overhead_bytes,num_instantiated_cells\n"
            csv_file.write(header)

        for i in range(100):
            try:
                chosen_row = baseline_deletion_2.get_random_key(dataset)
                if chosen_row is None: continue

                results = greedy_gumbel.gumbel_deletion_main(dataset=dataset, key=chosen_row, target_cell=attr)
                
                if results:
                    total_time = results['init_time'] + results['model_time'] + results['del_time']
                    with open(data_file_name, mode='a', newline='') as csv_file:
                        csv_row = (
                            f"{attr},{total_time},{results['init_time']},{results['model_time']},"
                            f"{results['del_time']},{results['leakage']},{results['utility']},"
                            f"{results['mask_size']},{results['num_paths']},{results['memory_overhead_bytes']},{results['num_instantiated_cells']}\n"
                        )
                        csv_file.write(csv_row)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/100 iterations for {dataset}")

            except Exception as e:
                print(f"Error in gumbel experiment, dataset {dataset}, iteration {i+1}: {e}")
                continue
        
        print(f"Completed greedy gumbel for {dataset}\n")

if __name__ == "__main__":
    main()
