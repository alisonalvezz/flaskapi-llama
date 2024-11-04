import json
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

local_llm = "llama3.2:1b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0.7)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

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
