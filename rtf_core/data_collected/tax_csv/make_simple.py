import os
import pandas as pd


def process_csv_files():
    """
    Finds all CSV files in the current directory, extracts specific columns,
    and saves them to a new file in a 'simple_ones_voter' subfolder.
    """
    # Define the name for the output subfolder
    output_folder = 'simple_ones_voter'

    # Create the subfolder if it doesn't already exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # List all files in the current directory
    files_in_directory = os.listdir('')

    # Loop through each file in the directory
    for filename in files_in_directory:
        # Check if the file is a CSV and not one of our generated simple files
        if filename.endswith('.csv') and not filename.endswith('_simple.csv'):
            print(f"Processing '{filename}'...")

            # Construct the full path to the input file
            input_filepath = os.path.join('', filename)

            try:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(input_filepath)

                # Define the columns to keep
                columns_to_keep = ['Time', 'Cells', 'Constraints']

                # Check if all required columns are present
                if all(col in df.columns for col in columns_to_keep):
                    # Select the desired columns
                    simple_df = df[columns_to_keep]

                    # Create the new filename
                    base_name = os.path.splitext(filename)[0]
                    new_filename = f"{base_name}_simple.csv"

                    # Construct the full path for the output file
                    output_filepath = os.path.join(output_folder, new_filename)

                    # Save the new DataFrame to a CSV file without the index
                    simple_df.to_csv(output_filepath, index = False)
                    print(f"  -> Successfully created '{output_filepath}'")
                else:
                    print(
                        f"  -> Skipping '{filename}': does not contain all required columns (Time, Cells, Constraints).")

            except Exception as e:
                # Catch any other errors during file processing
                print(f"  -> An error occurred while processing '{filename}': {e}")


if __name__ == "__main__":
    process_csv_files()
    print("\nProcessing complete.")
