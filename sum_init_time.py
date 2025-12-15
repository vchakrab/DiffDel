import csv

file_path = 'baseline_deletion_1_data_v4.csv'
data = {}
current_dataset = None
header_map = {}

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue

        if row[0].startswith('-----') and row[0].endswith('-----'):
            if current_dataset:
                # Store aggregated data for the previous dataset
                pass
            current_dataset = row[0].strip('-')
            data[current_dataset] = []
            header_map = {}
            continue

        if not current_dataset:
            continue

        if not header_map:
            header = [h.strip() for h in row]
            header_map = {name: idx for idx, name in enumerate(header)}
            continue

        try:
            init_time = float(row[header_map['init_time']])
            data[current_dataset].append(init_time)
        except (ValueError, IndexError, KeyError):
            continue

for dataset_name, init_times in data.items():
    print(f"Dataset: {dataset_name.strip().title()}, Sum of Init Time: {sum(init_times):.0f}")
