from typing import Any
import json
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class Response(BaseModel):
    label: str

def llama_api_call(
    llm: ChatOllama, prompt: ChatPromptTemplate
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
    
llm = ChatOllama(model="qwen3:8b", temperature=0)

parser = PydanticOutputParser(pydantic_object=Response)

prompt = ChatPromptTemplate(
    [
    ("system", 
        "You are an expert clinical NLI judge. "
        "Decide whether the given Statement is supported by the provided clinical trial evidence. "
        "Use ONLY the evidence given. Do NOT use outside knowledge. "
        "Output must be strictly one of: Entailment, Contradiction."),
    ("human", '''
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
        ),
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = llama_api_call(
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

    result = {"id" : val["id"], "label" : response.label}

    print(f'''doc count : {len(results)+1}
              result : {result}''')

    results.append(result)

# print(results)
with open("../results/results_qwen3.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)