import json
from langchain_core.messages import HumanMessage, SystemMessage
from utils.init_llama import llm_json_mode

doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""


doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

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
    return json.loads(result.content)
