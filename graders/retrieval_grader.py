import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

local_llm = "llama3.2:1b-instruct-fp16"
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

doc_grader_instructions = """You are a grader assessing the relevance of a retrieved document to a user question.

Your task is to evaluate whether the document contains keyword(s) or semantic meaning related to the question. Please consider the context and main ideas in both the document and the question.

If the document contains at least some information that is relevant to the question, grade it as relevant. If it does not contain relevant information, grade it as not relevant."""

doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
Please assess whether the document contains any relevant information related to the question.
Return JSON with a single key, binary_score, that is 'yes' if the document is relevant, or 'no' if it is not."""

def grade_document_relevance(llm_json_mode, document, question):
    """
    Evalúa la relevancia de un documento en relación a una pregunta.

    Parameters:
    llm_json_mode (llm instance): El modelo de lenguaje utilizado.
    document (str): El contenido del documento recuperado.
    question (str): La consulta del usuario.

    Returns:
    dict: JSON con una clave 'binary_score' que indica si el documento es relevante ('yes') o no ('no').
    """
    doc_grader_prompt_formatted = doc_grader_prompt.format(
        document=document, question=question
    )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=doc_grader_instructions)]
        + [HumanMessage(content=doc_grader_prompt_formatted)]
    )
    try:
        response_json = json.loads(result.content)
        return response_json
    except json.JSONDecodeError as e:
        return {'binary_score': 'no'}

