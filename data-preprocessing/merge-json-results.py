import os
import json
from glob import glob

# Define paths
input_dir = '../results/gpt-oss'
output_file = '../results/results_gpt-oss.json'

# Find all JSON files in the input directory
json_files = glob(os.path.join(input_dir, '*.json'))

merged_data = []

for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)

# Write merged data to output file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"Merged {len(merged_data)} records into {output_file}")