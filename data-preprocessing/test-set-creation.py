import json

#read test set JSON 
with open('../complete_dataset/Gold_test.json', encoding="utf-8") as file:
    test_data = json.load(file)

# print(test_data)
nli_test_data = []
for id, val in test_data.items():
    
    if val["Type"] == "Comparison":
        primary_doc = val["Primary_id"]
        secondary_doc = val["Secondary_id"]
        with open(f'../complete_dataset/CT json/{primary_doc}.json', encoding="utf-8") as file:
            primary_data = json.load(file)
        with open(f'../complete_dataset/CT json/{secondary_doc}.json', encoding="utf-8") as file:
            secondary_data = json.load(file)
        primary_evidence = primary_data[val["Section_id"]]
        secondary_evidence = secondary_data[val["Section_id"]]
        statement = val["Statement"]
        label = val["Label"]

        sample = {'id': id,
                'Section_id': val["Section_id"],
                'primary_evidence':primary_evidence, 
                'secondary_evidence':secondary_evidence,
                'statement':statement,
                'label':label}
        
    else:
        primary_doc = val["Primary_id"]
        with open(f'../complete_dataset/CT json/{primary_doc}.json', encoding="utf-8") as file:
            primary_data = json.load(file)

        primary_evidence = primary_data[val["Section_id"]]

        statement = val["Statement"]
        label = val["Label"]

        sample = {'id': id,
                'Section_id': val["Section_id"],
                'primary_evidence':primary_evidence, 
                'secondary_evidence':'',
                'statement':statement,
                'label':label} 
    
    # print(sample)
    nli_test_data.append(sample)

    with open("../nli_test_data/nli_test_data.json", "w", encoding="utf-8") as file:
        json.dump(nli_test_data, file, indent=4, ensure_ascii=False)
