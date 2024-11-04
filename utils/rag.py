import json
from langchain_core.messages import HumanMessage
from init_llama import llm_json_mode

rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""


def format_docs(docs):
    """Formatea los documentos para ser usados en el prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(relevant_docs, question):
    """Genera una respuesta utilizando los documentos relevantes y la pregunta del usuario."""
    docs_txt = format_docs(relevant_docs)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)

    generation = llm_json_mode.invoke([HumanMessage(content=rag_prompt_formatted)])
    return generation.content
