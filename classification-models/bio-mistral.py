# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


#read test set JSON 
with open('../nli_test_data/nli_test_data.json', encoding="utf-8") as file:
    nli_test_data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B")


results = []
for val in nli_test_data:

    primary_evidence = val["primary_evidence"]
    secondary_evidence = val["secondary_evidence"]

    statement = val["statement"]

    content = f'''
                System instuction:
                "You are an expert clinical NLI judge. "
                "Decide whether the given Statement is supported by the provided clinical trial evidence. "
                "Use ONLY the evidence given. Do NOT use outside knowledge. "
                "Output must be strictly one of: Entailment, Contradiction."
                
                Primary evidence:
                {primary_evidence}

                Secondary evidence:
                {secondary_evidence}

                Statement: 
                {statement}
            
                INSTRUCTIONS:
                - "Entailment": The statement is directly supported by or logically follows from the evidence
                - "Contradiction": The statement directly opposes or conflicts with the evidence
                - Focus on logical relationships, not just keyword matching
                - Consider numerical data, statistical significance, and clinical outcomes carefully
            
                Based on the combined evidence above, classify the relationship between the evidence and the statement.
                Output only one word: Entailment OR Contradiction      
               '''
    messages = [
        {"role": "user", "content": content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)

    result = {"id" : val["id"], "label" : tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])}
    results.append(result)

    with open("../results/bio-mistral/results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)