from typing import Any
import json

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


def gpt4_api_call(
    llm: AzureChatOpenAI, prompt: PromptTemplate
) -> Any | str:
    """
    Calls the GPT-4 API using a specified LLM, and prompt.

    Parameters:
    - llm: The large language model instance (e.g., AzureChatOpenAI).
    - prompt: The prompt template to be used.

    Returns:
    - The parsed output if successful, or an error message if an exception occurs.
    """
    try:
        chain = prompt | llm 
        return chain
    except Exception as e:
        # Log or handle the exception 
        return f"An unexpected error occurred: {str(e)}"

llm = AzureChatOpenAI()

prompt = PromptTemplate(
    template='''
              Primary evidence:
              {primary_evidence}

              Secondary evidence:
              {secondary_evidence}

              Statement: 
              {statement}

              based on the evidences , tell me the given Statement is Entailment / Contradiction
              only return the label with out any explanation from this list ["Entailment", "Contradiction"]
              
              ''',  # noqa: E501
    input_variables=["primary_evidence", "secondary_evidence", "statement"]
)

chain = gpt4_api_call(
    llm=llm,
    prompt=prompt,
)

#read test set JSON 
with open('../nli_test_data/nli_test_data.json', encoding="utf-8") as file:
    nli_test_data = json.load(file)


results = []
for val in nli_test_data:
    primary_evidence = val["primary_evidence"]
    secondary_evidence = val["secondary_evidence"]

    statement = val["statement"]

    response = chain.invoke(
            {
                "primary_evidence": primary_evidence,
                "secondary_evidence": secondary_evidence,
                "statement": statement,
            }
    ) 

    result = {"id" : val["id"], "label" : response.content.strip()}

    print(f'''doc count : {len(results)+1}
              result : {result}''')

    results.append(result)

# print(results)
with open("../results/results_gpt4o.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)