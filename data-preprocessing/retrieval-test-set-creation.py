import json 
import csv

with open ("../nli_test_data/nli_test_data.json", "r") as file:
    data = json.load(file)


with open ("../complete_dataset/Gold_test.json", "r") as file:
    gold_data = json.load(file)

retrival_test_data = []

for item in data:
    id = item["id"]
    primary_evidence = [i for i in range(len(item["primary_evidence"]))]
    primary_evidence_gold = gold_data[id]["Primary_evidence_index"]
    print(primary_evidence)
    print(primary_evidence_gold)
    for idx in primary_evidence:
        if idx in primary_evidence_gold:
            data_point = {"id": id, "evidence" : item["primary_evidence"][idx], "label": 1}
            retrival_test_data.append(data_point)
        else:
            data_point = {"id": id, "evidence" : item["primary_evidence"][idx], "label": 0}
            retrival_test_data.append(data_point)
       
    if gold_data[id]["Type"] == "Comparison":
        secondary_evidence = [i for i in range(len(item["secondary_evidence"]))]
        secondary_evidence_gold = gold_data[id]["Secondary_evidence_index"]
        print(secondary_evidence)
        print(secondary_evidence_gold) 
        for idx in secondary_evidence:
            if idx in secondary_evidence_gold:
                data_point = {"id": id, "evidence" : item["secondary_evidence"][idx], "label": 1}
                retrival_test_data.append(data_point)
            else:
                data_point = {"id": id, "evidence" : item["secondary_evidence"][idx], "label": 0}
                retrival_test_data.append(data_point)


with open('../complete_dataset/CSV_files/nli_retrive_test_data.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['id', 'evidence', 'label' ])
    writer.writeheader()
    writer.writerows(retrival_test_data)     
     