from langchain.schema import SystemMessage, HumanMessage
import json
from langchain_ollama import ChatOllama

local_llm = "llama3.2:1b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0.7)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

hallucination_grader_instructions = """
You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria and can lack of specific details and two sentences to respond are okay. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys: 'binary_score', which is 'yes' or 'no' to indicate whether the STUDENT ANSWER is grounded in the FACTS, and 'explanation', which contains an explanation of the score."""

def check_hallucination(llm_json_mode, documents, generation):
    """
    Evalúa si la respuesta contiene información no fundamentada en los documentos.

    Parameters:
    llm_json_mode (llm instance): El modelo de lenguaje utilizado.
    documents (str): Los documentos usados como base.
    generation (str): La respuesta generada.

    Returns:
    dict: JSON con 'binary_score' ('yes' o 'no') y una explicación del puntaje.
    """
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=documents, generation=generation
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    return json.loads(result.content)
