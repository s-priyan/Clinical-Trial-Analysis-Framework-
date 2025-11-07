import json

#read test set JSON 
with open('../results/rereival_results_large.json', encoding="utf-8") as file:
    retrieval_results = json.load(file)

# # #read test set JSON 
# with open('../results/rereival_results_ada.json', encoding="utf-8") as file:
#     retrieval_results = json.load(file)


# Extract the average_precision values
precisions = [v["average_precision"] for v in retrieval_results.values()]

# Compute mean average precision
mean_ap = sum(precisions) / len(precisions)

print("Mean Average Precision (mAP):", mean_ap)