#!/usr/bin/env python3

import json

def main():

    gold_filename = "../complete_dataset/Gold_test.json"
    pred_filename = "../results/rereival_results_ada.json"

    with open(pred_filename) as json_file:
        results= json.load(json_file)

    with open(gold_filename) as json_file:
        gold = json.load(json_file)

    uuid_list = list(results.keys())

    results_p = []
    gold_p =[]
    results_s = []
    gold_s =[]

    for i in range(len(uuid_list)):
        gold_p.append(gold[uuid_list[i]]["Primary_evidence_index"])
        results_p.append(results[uuid_list[i]]["Primary_evidence_index"])
        if gold[uuid_list[i]]["Type"]=="Comparison":
            gold_s.append(gold[uuid_list[i]]["Secondary_evidence_index"])
            results_s.append(results[uuid_list[i]]["Secondary_evidence_index"])


    TP = 0
    FP = 0
    FN = 0

    for i in range(len(gold_p)):
        for j in range(len(results_p[i])):
            if results_p[i][j] in gold_p[i]:
                TP += 1
            if results_p[i][j] not in gold_p[i]:
                FP += 1
        for j in range(len(gold_p[i])):
            if gold_p[i][j] not in results_p[i]:
                FN += 1

    for i in range(len(gold_s)):
        for j in range(len(results_s[i])):
            if results_s[i][j] in gold_s[i]:
                TP += 1
            if results_s[i][j] not in gold_s[i]:
                FP += 1
        for j in range(len(gold_s[i])):
            if gold_s[i][j] not in results_s[i]:
                FN += 1

    p_score = TP / (TP + FP)
    r_score = TP / (TP + FN)
    score = 2*(p_score * r_score)/(p_score + r_score)

    output_filename = 'scores.txt'
    with open(output_filename, 'w') as f:
        print('F1:{:f}'.format(score), file=f)
        print('precision_score:{:f}'.format(p_score), file=f)
        print('recall_score:{:f}'.format(r_score), file=f)
        
if '__main__' == __name__:
    main()










