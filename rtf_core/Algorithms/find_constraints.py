# rtf_core/multi_level_optimizer.py

import sys
import os
from typing import Set, Dict, Any, List
import time
import random
import csv

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from cell import Cell, Attribute
from rtf_core.initialization_phase import InitializationManager  # Corrected import statement
from rtf_core.analysis_phase import OrderedAnalysisPhase
from rtf_core.decision_phase import DecisionPhase


class RTFMultiLevelAlgorithm:
    """
    Corrected RTF algorithm that orchestrates modular components.
    """

    def __init__(self, target_cell_info, dataset = 'RTF25', threshold_alpha = 0.8):
        self.init_manager = InitializationManager(target_cell_info, dataset, threshold_alpha)
        # self.analysis_phase = OrderedAnalysisPhase(self.init_manager)
        # self.decision_phase = DecisionPhase(self.init_manager)
        # print("RTF Algorithm initialized")
        # print(f"Target: {target_cell_info}")
        # print(f"Threshold: {threshold_alpha}")

    def run_complete_algorithm(self):
        """
        Main method implementing your complete multi-level analysis algorithm.
        """

        # Initialization Phase
        self.init_manager.initialize()
        return self.init_manager.constraint_cells

list_of_4k_indices_longitude = []
def RTF_algorithm(dataset):
    """Test the corrected algorithm."""
    hospital_attributes = ["ProviderNumber", "HospitalName", "City", "State", "ZIPCode",
                           "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
                           "EmergencyService", "Condition", "MeasureCode", "MeasureName", "Sample",
                           "StateAvg"]
    tax_attributes = ["fname", "lname", "gender", "area_code", "phone", "city", "state", "zip", "marital_status", "has_child", "salary", "rate", "single_exemp", "married_exemp", "child_exemp"]


    airport_attributes = ['ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
                    'iso_country', 'iso_region', 'municipality', 'scheduled_service']
    time_data = []
    cell_data = []
    cell_data_whole = []
    constraints_whole = []
    constraint_data = []
    for i in range(1):
        start_time = time.time()
        target_info = {'key': random.randint(1, 2), 'attribute': 'type'}  # Try key=1 instead of 2
        algorithm = RTFMultiLevelAlgorithm(target_info, dataset, 0.8)
        cellss = algorithm.run_complete_algorithm()
        #print(type(cellss))
        print(algorithm.init_manager.current_deletion_set)
        constraint_data.append(len(algorithm.init_manager.target_denial_constraints))
        constraints_whole.append(algorithm.init_manager.target_denial_constraints)
        end_time = time.time()
        cell_data.append(len(cellss))
        cell_data_whole.append(cellss)
        time_data.append(end_time - start_time)
    with open("hospital_100_random_cell_data.csv", "w", newline= '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "Cells", "Constraints", "Whole Constraints", "Whole Cells"])
        writer.writerows(zip(time_data, cell_data, constraint_data, constraints_whole, cell_data_whole))
    return time_data, cell_data, constraint_data, cell_data_whole, constraints_whole
if __name__ == '__main__':
    airport_atts = ['ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft', 'iso_country', 'iso_region', 'municipality', 'scheduled_service']
    data = RTF_algorithm('airport')

    #print(data)
#
# if __name__ == '__main__':
#     voter_attributes = [
#         "voter_id",
#         "voter_reg_num",
#         "name_prefix",
#         "first_name",
#         "middle_name",
#         "last_name",
#         "name_suffix",
#         "age",
#         "gender",
#         "race",
#         "ethnic",
#         "street_address",
#         "city",
#         "state",
#         "zip_code",
#         "full_phone_num",
#         "birth_place",
#         "register_date",
#         "download_month"
#     ]
#
#     hospital_attributes = ["ProviderNumber", "HospitalName", "City", "State", "ZIPCode",
#                            "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
#                            "EmergencyService", "Condition", "MeasureCode", "MeasureName", "Sample",
#                            "StateAvg"]
#     tax_attributes = ["fname", "lname", "gender", "area_code", "phone", "city", "state", "zip",
#                       "marital_status", "has_child", "salary", "rate", "single_exemp",
#                       "married_exemp", "child_exemp"]
#
#     airport_attributes = ['ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
#                           'iso_country', 'iso_region', 'municipality', 'scheduled_service']
#     data = RTF_algorithm("airport", airport_attributes, 0)
#     print(data)
#
#     # for i in range(45000, 46000, 1000):
#     #     RTF_algorithm("airport", "type", 0)

'''
def RTF_algorithm(dataset):
    """Test the corrected algorithm."""
    hospital_attributes = ["ProviderNumber", "HospitalName", "City", "State", "ZIPCode",
                           "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
                           "EmergencyService", "Condition", "MeasureCode", "MeasureName", "Sample",
                           "StateAvg"]
    tax_attributes = ["fname", "lname", "gender", "area_code", "phone", "city", "state", "zip", "marital_status", "has_child", "salary", "rate", "single_exemp", "married_exemp", "child_exemp"]


    airport_attributes = ['ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
                    'iso_country', 'iso_region', 'municipality', 'scheduled_service']
    time_data = []
    cell_data = []
    cell_data_whole = []
    constraints_whole = []
    constraint_data = []
    for i in range(100):
        start_time = time.time()
        target_info = {'key': random.randint(1, 3500), 'attribute': hospital_attributes[random.randint(1, len(hospital_attributes)-1)]}  # Try key=1 instead of 2
        algorithm = RTFMultiLevelAlgorithm(target_info, dataset, 0.8)
        cellss = algorithm.run_complete_algorithm()
        constraint_data.append(len(algorithm.init_manager.target_denial_constraints))
        constraints_whole.append(algorithm.init_manager.target_denial_constraints)
        end_time = time.time()
        cell_data.append(len(cellss))
        cell_data_whole.append(cellss)
        time_data.append(end_time - start_time)
    with open("hospital_100_random_cell_data.csv", "w", newline= '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "Cells", "Constraints", "Whole Constraints", "Whole Cells"])
        writer.writerows(zip(time_data, cell_data, constraint_data, constraints_whole, cell_data_whole))
    return time_data, cell_data, constraint_data, cell_data_whole, constraints_whole
if __name__ == '__main__':
    airport_atts = ['ident', 'type', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft', 'iso_country', 'iso_region', 'municipality', 'scheduled_service']
    data = RTF_algorithm(dataset)
    print("CELL STUFF")
    for row in data[0]:
        print(row)
    print("TIME DATA")
    for row in data[1]:
        print(row)
        
def RTF_algorithm(dataset, attribute, rand_ints):
    """Test the corrected algorithm."""
    #for att in attribute_list:
    # tax_attributes = ["fname", "lname", "gender", "area_code", "phone", "city", "state", "zip",
    #                   "marital_status", "has_child", "salary", "rate", "single_exemp",
    #                   "married_exemp", "child_exemp"]
    #airports size: 45036, hospital: 114919, ncvoter: 1000, tax: 99904
    time_data = []
    db_data = []
    cell_data = []
    # cell_data_whole = []
    # constraints_whole = []
    # constraint_data = []
    filename = f"{dataset}_cell_data.csv"
    file_exists = os.path.exists(filename)

    for i in range(1):
        start_time = time.time()
        key = 5
        target_info = {'key': key, 'attribute': attribute}  # Try key=1 instead of 2
        list_of_4k_indices_longitude.append(key)
        algorithm = RTFMultiLevelAlgorithm(target_info, dataset, 0.8)
        cellss = algorithm.run_complete_algorithm()
        print(cellss)
        #constraint_data.append(len(algorithm.init_manager.target_denial_constraints))
        #constraints_whole.append(algorithm.init_manager.target_denial_constraints)
        end_time = time.time()
        cell_data.append(len(cellss))
        #cell_data_whole.append(cellss)
        time_data.append(end_time - start_time)
    with open(filename, "a", newline = '') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(cell_data)
        writer.writerows(zip(attribute, cell_data))
'''