import json
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity

embedding_model = AzureOpenAIEmbeddings()


#read test set JSON 
with open('../demo-data/evidence_retrival/input/demo-record.json', encoding="utf-8") as file:
    nli_test_data = json.load(file)


retrival_test_data = []
for record in nli_test_data:
    uuid = record["id"]
    primary_scores = []
    secondary_scores = []

    primary_evidence_index_pred = []
    secondary_evidence_index_pred = []
    
    primary_evidence = record["primary_evidence"]

    statement = record["statement"]

    primary_combined_docs = primary_evidence + [statement]

    primary_combined_vector = embedding_model.embed_documents(primary_combined_docs)

    primary_keyword_vectors = primary_combined_vector[:-1]

    query_vector = primary_combined_vector[-1]


    for keyword_vector in primary_keyword_vectors:

        score = cosine_similarity([query_vector], [keyword_vector])[0][0]
        
        primary_scores.append({"idx": primary_keyword_vectors.index(keyword_vector), "source" : "primary", "score" : score})


    for score in primary_scores:
        if score["score"] > 0.8:
            primary_evidence_index_pred.append(primary_scores.index(score))

    for idx in range(len(primary_evidence)):
        if idx in primary_evidence_index_pred:
            data_point = {"id": uuid, "CTR": "Primary", "evidence" : record["primary_evidence"][idx]}
            retrival_test_data.append(data_point)

    if record["secondary_evidence"] != "":

        secondary_evidence = record["secondary_evidence"]
        
        secondary_combined_docs = secondary_evidence + [statement]

        secondary_combined_vector = embedding_model.embed_documents(secondary_combined_docs)

        secondary_keyword_vectors = secondary_combined_vector[:-1]

        secondary_query_vector = secondary_combined_vector[-1]

        for keyword_vector in secondary_keyword_vectors:

            score = cosine_similarity([secondary_query_vector], [keyword_vector])[0][0]
            
            secondary_scores.append({"idx": secondary_keyword_vectors.index(keyword_vector), "source" : "secondary", "score" : score})         

        for score in secondary_scores:
            if score["score"] > 0.8:
                secondary_evidence_index_pred.append(secondary_scores.index(score))   

        for idx in range(len(secondary_evidence)):
            if idx in secondary_evidence_index_pred:
                data_point = {"id": uuid, "CTR": "Secondary", "evidence" : record["secondary_evidence"][idx]}
                retrival_test_data.append(data_point)

    print(retrival_test_data)

    with open('../demo-data/evidence_retrival/output/rereival_results_ada.json', 'w') as file:
        json.dump(retrival_test_data, file, indent=4)
