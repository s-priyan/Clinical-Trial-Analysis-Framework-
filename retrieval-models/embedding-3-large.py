import json
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity


embedding_model = AzureOpenAIEmbeddings()

#read test set JSON 
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

    primary_combined_vector = embedding_model.embed_documents(primary_combined_docs)

    primary_keyword_vectors = primary_combined_vector[:-1]

    query_vector = primary_combined_vector[-1]


    for keyword_vector in primary_keyword_vectors:

        score = cosine_similarity([query_vector], [keyword_vector])[0][0]
        
        primary_scores.append({"idx": primary_keyword_vectors.index(keyword_vector), "source" : "primary", "score" : score})


    # for score in primary_scores:
    #     if score > 0.8:
    #         primary_evidence_index.append(primary_scores.index(score))
    
    if record["secondary_evidence"] != "":

        secondary_evidence = record["secondary_evidence"]
        
        secondary_combined_docs = secondary_evidence + [statement]

        secondary_combined_vector = embedding_model.embed_documents(secondary_combined_docs)

        secondary_keyword_vectors = secondary_combined_vector[:-1]

        secondary_query_vector = secondary_combined_vector[-1]

        for keyword_vector in secondary_keyword_vectors:

            score = cosine_similarity([secondary_query_vector], [keyword_vector])[0][0]
            
            secondary_scores.append({"idx": secondary_keyword_vectors.index(keyword_vector), "source" : "secondary", "score" : score})         

        # for score in secondary_scores:
        #     if score > 0.8:
        #         secondary_evidence_index.append(secondary_scores.index(score))   
        # 
    precision_iteration_range =  len(gold[uuid]["Primary_evidence_index"]) + len(gold[uuid]["Secondary_evidence_index"]) if gold[uuid]["Type"] == "Comparison" else len(gold[uuid]["Primary_evidence_index"]) 
    
    combined_scores = primary_scores + secondary_scores

    combined_scores_sorted = sorted(combined_scores, key=lambda x: x['score'], reverse=True)[:precision_iteration_range]
    
    # print(primary_scores)
    print(combined_scores_sorted)

    sum_p_score = 0
    for iteration in range(precision_iteration_range):
         primary_evidence_index = []
         secondary_evidence_index = []
         gold_p =gold[uuid]["Primary_evidence_index"]
         if gold[uuid]["Type"]=="Comparison":
            gold_s =gold[uuid]["Secondary_evidence_index"] 
         TP = 0
         FP = 0
         FN = 0
         step_combined_scores_sorted = combined_scores_sorted[:iteration + 1]
         for score in step_combined_scores_sorted:
             if score["source"] == "primary":
                 primary_evidence_index.append(score["idx"])
             else:
                 secondary_evidence_index.append(score["idx"])
         
         for j in range(len(primary_evidence_index)):
             if primary_evidence_index[j] in gold_p:
                    TP += 1
             if primary_evidence_index[j] not in gold_p:
                    FP += 1
             for j in range(len(gold_p)):
                if gold_p[j] not in primary_evidence_index:
                    FN += 1
         for j in range(len(secondary_evidence_index)):
             if secondary_evidence_index[j] in gold_s:
                    TP += 1
             if secondary_evidence_index[j] not in gold_s:
                    FP += 1
             for j in range(len(gold_s)):
                if gold_s[j] not in secondary_evidence_index:
                    FN += 1

         p_score = TP / (TP + FP)
         r_score = TP / (TP + FN)
        #  f1_score = 2*(p_score * r_score)/(p_score + r_score)
         sum_p_score += p_score
        #  print(f"Iteration {iteration + 1}: Precision: {p_score}")
    print(f"Average Precision: {sum_p_score / precision_iteration_range}")   
    results[uuid] = {
        "average_precision": sum_p_score / precision_iteration_range
    }

with open('../results/rereival_results_large.json', 'w') as file:
    json.dump(results, file, indent=4)

