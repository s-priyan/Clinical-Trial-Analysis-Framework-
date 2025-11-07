import pandas as pd

# Load the two CSV files
df1 = pd.read_csv('../complete_dataset/CSV_files/nli_test_data.csv')
# df2 = pd.read_csv('../complete_dataset/CSV_files/results_deepseek.csv')
df2 = pd.read_csv('../complete_dataset/CSV_files/results_gpt-oss.csv')

# Merge on 'id'
merged_df = pd.merge(df1, df2, on="id", how="outer")

print(f"Number of records after merging: {len(merged_df)}")

# Normalize and map string labels to integers in label_file1
merged_df["ground_truth"] = merged_df["ground_truth"].str.lower().map({
    "contradiction": 0,
    "entailment": 1
})

# Normalize and map string labels to integers in label_file2
merged_df["prediction"] = merged_df["prediction"].str.lower().map({
    "contradiction": 0,
    "entailment": 1
})

invalid_indices = merged_df[merged_df["prediction"].isna()].index

# Print the indices and the corresponding invalid labels (optional)
print("Invalid label indices and values:")
print(merged_df.loc[invalid_indices, ["prediction"]])

merged_df = merged_df.dropna(subset=["prediction"])

print(f"Number of records after normalizing ground_truth: {len(merged_df)}")

# Save or display the result
merged_df.to_csv('../complete_dataset/CSV_files/merged_output_gpt-oss.csv', index=False)

