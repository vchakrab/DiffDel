import mysql.connector
from mysql.connector import Error
import config  # Your configuration file
import math
import numpy as np
from typing import Dict, Tuple, Any

MARGINAL_HISTOGRAM_QUERY_TEMPLATE = """
SELECT COUNT(*) AS count
FROM {primary_table}
WHERE {target_attribute} = '{value}';
"""
JOINT_HISTOGRAM_QUERY_TEMPLATE = """
SELECT
COUNT(*) AS count
FROM
    {primary_table}
WHERE 
    {attribute_1} = '{value1}' AND {attribute_2} = '{value2}';"""


def calculate_minimum_entropy(joint_probs):
    min_entropy = joint_probs[0]*np.log(1/joint_probs[0])
    for jp in joint_probs:
        curr_entropy = -1 * jp*math.log(jp)
        if curr_entropy < min_entropy:
            min_entropy = curr_entropy
    return min_entropy

def calculate_naive_bayes_entropy(joint_probs, marginal_probs):
    entropy = -1 * marginal_probs[0] * np.log(marginal_probs[0])
    total_mi = 0
    for i in range(1, len(joint_probs)):
        total_mi += joint_probs[i] * math.log(joint_probs[i]/(marginal_probs[0]*marginal_probs[i]))
    return entropy - total_mi

def calculate_entropy_from_sql(dataset_name: str, rule: dict, target_attribute: dict):
    """
    Connects to the database, executes the histogram query, calculates probabilities,
    and returns the Shannon Entropy for the target attribute.
    """
    conn = None

    try:
        # 1. Get Configuration and Query Details
        db_details = config.get_database_config(dataset_name)
        primary_table = config.get_primary_table(dataset_name)

        print(f"--- 🔌 Connecting to DB: {db_details['database']} ---")

        # 2. Establish Connection
        conn = mysql.connector.connect(
            host = db_details['host'],
            user = db_details['user'],
            password = db_details['password'],
            database = db_details['database'],
            ssl_disabled = db_details['ssl_disabled']
        )

        if not conn.is_connected():
            print("Connection failed.")
            return 0.0
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {primary_table}")
        total_records = cursor.fetchone()[0]  # Fetch the single count value
        print(f"Total Records: {total_records}")

        marginal_probs = []
        joint_probs = []

        # 3. Fetch Marginal and Joint Counts for Rule Attributes
        for attr, value in rule.items():

            query_m = MARGINAL_HISTOGRAM_QUERY_TEMPLATE.format(primary_table = primary_table,
                                                               target_attribute = attr,
                                                               value = value)
            cursor.execute(query_m)
            count_m = cursor.fetchone()
            if count_m:
                marginal_probs.append(count_m[0] / total_records)

            query_j = JOINT_HISTOGRAM_QUERY_TEMPLATE.format(primary_table = primary_table,
                                                            attribute_1 = attr, value1 = value,
                                                            attribute_2 = list(target_attribute.keys())[0],
                                                            value2 = list(target_attribute.values())[0])
            cursor.execute(query_j)
            count_j = cursor.fetchone()
            if count_j:
                joint_probs.append(count_j[0] / total_records)

        query_target_m = MARGINAL_HISTOGRAM_QUERY_TEMPLATE.format(
            primary_table = primary_table,
            target_attribute = list(target_attribute.keys())[0],
            value = list(target_attribute.values())[0]
        )
        cursor.execute(query_target_m)
        count_target_m = cursor.fetchone()

        if count_target_m:
            marginal_probs.append(count_target_m[0] / total_records)
        return calculate_minimum_entropy(joint_probs), calculate_naive_bayes_entropy(joint_probs, marginal_probs)

    except Error as e:
        print(f"\n[FATAL ERROR] MySQL Error: {e}")
        return 0.0

    except ValueError as e:
        print(f"[CONFIGURATION ERROR] {e}")
        return 0.0

    finally:
        # 6. Close Connection
        if 'cursor' in locals() and cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("\nConnection closed.")


initalized_rule = {'iso_country': 'US', 'elevation_ft': 237}
target = {'type': 'small_airport'}
dataset_name = 'airport'
print(calculate_entropy_from_sql(dataset_name, initalized_rule, target))
# ============================================================================
# EXECUTION BLOCK
# ============================================================================

# TARGET_DATASET = 'adult'
# # Using 'education' attribute based on ALGORITHM_DEFAULTS in config.py
# TARGET_ATTRIBUTE = 'education'
#
# final_entropy = calculate_entropy_from_sql(TARGET_DATASET, TARGET_ATTRIBUTE)
#
# if final_entropy > 0:
#     print(
#         f"\n✅ Calculated Entropy (H) for '{TARGET_ATTRIBUTE}' in '{TARGET_DATASET}': {final_entropy:.4f} bits")