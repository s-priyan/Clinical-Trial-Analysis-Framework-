import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)

def get_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings

# Read test set JSON
with open('../nli_test_data/nli_test_data.json', encoding="utf-8") as file:
    nli_test_data = json.load(file)

with open("../complete_dataset/Gold_test.json") as json_file:
    gold = json.load(json_file)

results = {}
for record in nli_test_data:
    uuid = record["id"]
    primary_scores = []
    secondary_scores = []

    primary_evidence = record["primary_evidence"]
    statement = record["statement"]

    primary_combined_docs = primary_evidence + [statement]
    primary_combined_vector = get_embeddings(primary_combined_docs)
    primary_keyword_vectors = primary_combined_vector[:-1]
    query_vector = primary_combined_vector[-1].reshape(1, -1)

    for idx, keyword_vector in enumerate(primary_keyword_vectors):
        score = cosine_similarity(query_vector, keyword_vector.reshape(1, -1))[0][0]
        primary_scores.append({"idx": idx, "source": "primary", "score": score})

    if record["secondary_evidence"] != "":
        secondary_evidence = record["secondary_evidence"]
        secondary_combined_docs = secondary_evidence + [statement]
        secondary_combined_vector = get_embeddings(secondary_combined_docs)
        secondary_keyword_vectors = secondary_combined_vector[:-1]
        secondary_query_vector = secondary_combined_vector[-1].reshape(1, -1)

        for idx, keyword_vector in enumerate(secondary_keyword_vectors):
            score = cosine_similarity(secondary_query_vector, keyword_vector.reshape(1, -1))[0][0]
            secondary_scores.append({"idx": idx, "source": "secondary", "score": score})

    precision_iteration_range = (
        len(gold[uuid]["Primary_evidence_index"]) + len(gold[uuid]["Secondary_evidence_index"])
        if gold[uuid]["Type"] == "Comparison"
        else len(gold[uuid]["Primary_evidence_index"])
    )

    combined_scores = primary_scores + secondary_scores
    combined_scores_sorted = sorted(combined_scores, key=lambda x: x['score'], reverse=True)[:precision_iteration_range]

    sum_p_score = 0
    for iteration in range(precision_iteration_range):
        primary_evidence_index = []
        secondary_evidence_index = []
        gold_p = gold[uuid]["Primary_evidence_index"]
        gold_s = gold[uuid].get("Secondary_evidence_index", [])
        TP = FP = FN = 0
        step_combined_scores_sorted = combined_scores_sorted[:iteration + 1]
        for score in step_combined_scores_sorted:
            if score["source"] == "primary":
                primary_evidence_index.append(score["idx"])
            else:
                secondary_evidence_index.append(score["idx"])

        for idx in primary_evidence_index:
            if idx in gold_p:
                TP += 1
            else:
                FP += 1
        for idx in gold_p:
            if idx not in primary_evidence_index:
                FN += 1

        for idx in secondary_evidence_index:
            if idx in gold_s:
                TP += 1
            else:
                FP += 1
        for idx in gold_s:
            if idx not in secondary_evidence_index:
                FN += 1

        p_score = TP / (TP + FP) if (TP + FP) > 0 else 0
        sum_p_score += p_score

    print(f"Average Precision: {sum_p_score / precision_iteration_range if precision_iteration_range else 0}")
    results[uuid] = {
        "average_precision": sum_p_score / precision_iteration_range if precision_iteration_range else 0
    }

with open('../results/rereival_results_nomic-embed.json', 'w') as file:
    json.dump(results, file, indent=4)