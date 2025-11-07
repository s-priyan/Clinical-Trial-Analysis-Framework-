import csv
import json



with open('../results/retrival_results/rereival_results_ada.json', encoding="utf-8") as file:
    nli_predict_data_ada = json.load(file)

with open('../complete_dataset/CSV_files/results_ada.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['id', 'evidence', 'label'])
    writer.writeheader()
    filtered_data = [{'id': item['id'], 'evidence': item['evidence'], 'label': item['label']} for item in nli_predict_data_ada ]
    writer.writerows(filtered_data)


