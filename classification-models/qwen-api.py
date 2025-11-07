from openai import OpenAI
import json
import time
import re

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
)

def extract_final_label(text):
    # Remove <think>...</think> and any whitespace/newlines after it
    text_no_think = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Remove leading/trailing whitespace
    text_no_think = text_no_think.strip()
    # Only keep the last word if it's Contradiction or Entailment
    for label in ["Contradiction", "Entailment"]:
        if text_no_think.endswith(label):
            return label

#read test set JSON 
with open('../nli_test_data/nli_test_data.json', encoding="utf-8") as file:
    nli_test_data = json.load(file)

results = []
for val in nli_test_data[490:500]:

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
    extra_headers={},
    model="qwen/qwq-32b:free",
    messages=[
        {
        "role": "user",
        "content": content
        }
    ]
    )
    print(completion.choices[0].message.content)

    result = {"id" : val["id"], "label" : extract_final_label(completion.choices[0].message.content)}

    results.append(result)

    with open("../results/Qwen/results15.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    time.sleep(60) 