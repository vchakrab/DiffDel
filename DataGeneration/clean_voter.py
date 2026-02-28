import pandas as pd

input_file = '../csv_files/ncvoter.csv'
output_file = '../csv_files/cleaned_ncvoter.csv'

# Read CSV without automatic dtype inference, keep everything as string
df = pd.read_csv(input_file, dtype=str)

# Clean column names: remove quotes and type suffixes
df.columns = [col.replace('"', '').split()[0] for col in df.columns]

# Remove quotes from all data values
df = df.apply(lambda col: col.str.replace('"', '', regex=False) if col.dtype == "object" else col)

# Define schema for casting
schema = {
    "voter_id": str,
    "voter_reg_num": str,
    "name_prefix": str,
    "first_name": str,
    "middle_name": str,
    "last_name": str,
    "name_suffix": str,
    "age": int,
    "gender": str,
    "race": str,
    "ethnic": str,
    "street_address": str,
    "city": str,
    "state": str,
    "zip_code": str,
    "full_phone_num": str,
    "birth_place": str,
    "register_date": str,
    "download_month": str
}

# Cast columns according to schema
for col, col_type in schema.items():
    if col_type == int:
        df[col] = df[col].astype(int)
    else:
        df[col] = df[col].astype(str)

# Save cleaned CSV
df.to_csv(output_file, index=False)
print(f"Cleaned CSV saved to {output_file}")
