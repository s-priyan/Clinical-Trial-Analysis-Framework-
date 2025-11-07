import csv
import json


# with open('../nli_test_data/nli_test_data.json', encoding="utf-8") as file:
#     nli_test_data = json.load(file)

# with open('../results/results_deepseek.json', encoding="utf-8") as file:
#     nli_predict_data_deepseek = json.load(file)

# with open('../results/results_gpt4o.json', encoding="utf-8") as file:
#     nli_predict_data_gpt4o = json.load(file)

# with open('../results/results_qwen.json', encoding="utf-8") as file:
#     nli_predict_data_gpt4o = json.load(file)

with open('../results/results_gpt-oss.json', encoding="utf-8") as file:
    nli_predict_data_gpt4o = json.load(file)

# with open('../complete_dataset/CSV_files/nli_test_data.csv', 'w', newline='') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=['id', 'Section_id', 'ground_truth'])
#     writer.writeheader()
#     filtered_data = [{'id': item['id'], 'Section_id': item['Section_id'], 'ground_truth': item['label']} for item in nli_test_data]
#     writer.writerows(filtered_data)

# with open('../complete_dataset/CSV_files/results_deepseek.csv', 'w', newline='') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=['id', 'prediction'])
#     writer.writeheader()
#     filtered_data = [{'id': item['id'], 'prediction': item['label']} for item in nli_predict_data_deepseek ]
#     writer.writerows(filtered_data)

# with open('../complete_dataset/CSV_files/results_gpt4o.csv', 'w', newline='') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=['id', 'prediction'])
#     writer.writeheader()
#     filtered_data = [{'id': item['id'], 'prediction': item['label']} for item in nli_predict_data_gpt4o ]
#     writer.writerows(filtered_data)

with open('../complete_dataset/CSV_files/results_gpt-oss.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['id', 'prediction'])
    writer.writeheader()
    filtered_data = [{'id': item['id'], 'prediction': item['label']} for item in nli_predict_data_gpt4o ]
    writer.writerows(filtered_data)


