from openai import OpenAI
import json

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
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
              Primary evidence:
              {primary_evidence}

              Secondary evidence:
              {secondary_evidence}

              Statement: 
              {statement}

              based on the evidences , tell me the given Statement is Entailment / Contradiction
              only return the label with out any explanation from this list ["Entailment", "Contradiction"]
              
              '''
  
  print(content)

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