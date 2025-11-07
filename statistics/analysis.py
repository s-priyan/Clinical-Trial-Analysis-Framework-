from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

# def list_files_in_subfolders(parent_folder):
#     all_files = []
#     # List all entries in the parent folder
#     for entry in os.listdir(parent_folder):
#         entry_path = os.path.join(parent_folder, entry)
#         all_files.append(entry_path)  
#     return all_files

# parent_folder = "./results"
# all_files = list_files_in_subfolders(parent_folder)
# print(all_files)


# results = []
# for file_path in all_files:
#     #read test set JSON 
#     with open(file_path, encoding="utf-8") as file:
#         nli_test_data = json.load(file)

#     for val in nli_test_data:
#         result = {"id": val["id"], "label": val["label"]}
#         results.append(result)  

# with open("./results/combined_results.json", "w", encoding="utf-8") as file:
#     json.dump(results, file, indent=4, ensure_ascii=False)


# Load the merged CSV 
df = pd.read_csv('../complete_dataset/CSV_files/merged_output_gpt-oss.csv')
print(len(df))

# Drop rows with missing labels 
df = df.dropna(subset=["ground_truth", "prediction"])

print(len(df))

# set labels true and prediction
y_true = df["ground_truth"]
y_pred = df["prediction"]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)


print(cm)


# Calculate precision, recall, F1 score, and accuracy
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)    
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
accuracy = accuracy_score(y_true, y_pred)
# Print the results
print(f"Precision: {precision:.4f}")            
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")        
print(f"Accuracy: {accuracy:.4f}")