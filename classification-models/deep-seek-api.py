from openai import OpenAI
import json

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
)

#read test set JSON 
with open('../nli_test_data/nli_test_data.json', encoding="utf-8") as file:
    nli_test_data = json.load(file)

results = []
for val in nli_test_data[450:500]:

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

  completion = client.chat.completions.create(
    extra_body={}, 
    model="deepseek/deepseek-r1:free",
    messages=[
      {
        "role": "user", 
        "content": content
      }
    ]
  )

  print(completion.choices[0].message.content)

  result = {"id" : val["id"], "label" : completion.choices[0].message.content}

  results.append(result)

with open("../results/results10.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)