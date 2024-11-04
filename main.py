import json
from routing.router import route_question
from graders.retrieval_grader import grade_document_relevance
from graders.hallucination_grader import check_hallucination
from graders.answer_grader import evaluate_answer
from vectorstore.document_loader import retriever as vector_retriever
from utils.rag import generate_answer
from init_llama import llm_json_mode
from tavily import TavilyClient

tavily_client = TavilyClient(api_key="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")

def main(query):
    print(f"Consulta recibida: {query}")
    routing_decision = route_question(llm_json_mode, {"question": query})
    datasource = routing_decision.get("datasource")

    if datasource == "vectorstore":
        documents = vector_retriever.get_relevant_documents(query)
        print(f"Documentos recuperados: {len(documents)}")

        relevant_docs = []
        for doc in documents:
            relevance = grade_document_relevance(llm_json_mode, doc.page_content, query)
            print(f"Relevancia del documento: {relevance}")
            if relevance.get("binary_score") == "yes":
                relevant_docs.append(doc)

        print(f"Documentos relevantes: {len(relevant_docs)}")
        if not relevant_docs:
            print("No se encontraron documentos relevantes, realizando búsqueda web.")
            answer = tavily_client.qna_search(query=query)
            return answer
    else:
        print("No se encontró un datasource adecuado, realizando búsqueda web.")
        answer = tavily_client.qna_search(query=query)
        return answer

    answer = generate_answer(relevant_docs, query)
    print(f"Respuesta generada: {answer}")

    if not answer or answer == None or answer == "{}":
        print("No se generó ninguna respuesta válida, realizando búsqueda web.")
        return tavily_client.qna_search(query=query)

    hallucination_check = check_hallucination(llm_json_mode, relevant_docs, answer)
    print(f"Chequeo de alucinaciones: {hallucination_check}")

    if hallucination_check.get("binary_score") == "yes":
        answer_evaluation = evaluate_answer(query, answer)
        print(f"Evaluación de la respuesta: {answer_evaluation}")
        print("Respuesta final:", answer)
        return answer
    else:
        print("Alucinación detectada, realizando búsqueda web.")
        web_results = tavily_client.qna_search(query=query)
        return web_results

