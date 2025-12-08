import random
import baseline_deletion_2
import baseline_deletion_3
import baseline_deletion_1
import os
import psutil

hospital_attributes = ["ProviderNumber", "HospitalName", "City", "State", "ZIPCode",
                           "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
                           "EmergencyService", "Condition", "MeasureCode", "MeasureName", "Sample",
                           "StateAvg"]
tax_attributes = ["fname",
                  "lname", "gender", "area_code", "phone", "city", "state", "zip", "marital_status", "has_child", "salary", "rate", "single_exemp", "married_exemp", "child_exemp"]
ncvoter_attributes = [
    "voter_id",
    "voter_reg_num",
    "name_prefix",
    "first_name",
    "middle_name",
    "last_name",
    "name_suffix",
    "age",
    "gender",
    "race",
    "ethnic",
    "street_address",
    "city",
    "state",
    "zip_code",
    "full_phone_num",
    "birth_place",
    "register_date",
    "download_month"
]
airport_attributes = ['ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
                    'iso_country', 'iso_region', 'municipality', 'scheduled_service']
import time;
sizes = {"airport": 45037, "hospital": 114920, "tax": 99905, "ncvoter": 1001}



# def collect_baseline_2_data_for_all_dbs():
#     data_file_name = "baseline_deletion_2_data_v2.csv"
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
#                 cells_deleted,explanation_time, memory,depth, dependencies = baseline_deletion_2.delete_one_path_dependent_cell(chosen_attr, chosen_row, dataset, 0.5)
#                 end_time = time.time() - start_time - explanation_time
#                 with open(data_file_name, mode = 'a') as csv_file:
#                     csv_file.write(f"{chosen_attr},{end_time},{dependencies},{cells_deleted},{depth},{memory}\n")
# collect_baseline_2_data_for_all_dbs()

def collect_baseline_1_data_for_all_dbs():
    data_file_name = "baseline_deletion_1_data_v2.csv"
    for dataset, attrs, in zip(["airport", "hospital", "ncvoter", "tax"], ["latitude_deg", "ProviderNumber", "voter_reg_num", "marital_status"]):
            with open(data_file_name, mode = 'a') as csv_file:
                csv_file.write(f"-----{dataset}-----\n")
                csv_file.write("attribute,time,dependencies,cells,depth,space_overhead(B)\n")
            for i in range(100):
                chosen_row = baseline_deletion_2.get_random_key(dataset)
                chosen_attr = attrs
                start_time = time.time()
                depth = 1
                dependencies, cells_deleted, memory = baseline_deletion_1.delete_all_dependent_cells(chosen_attr, chosen_row, dataset, 0.8)
                end_time = time.time() - start_time
                with open(data_file_name, mode = 'a') as csv_file:
                    csv_file.write(f"{chosen_attr},{end_time},{dependencies},{cells_deleted},{depth},{memory}\n")
collect_baseline_1_data_for_all_dbs()