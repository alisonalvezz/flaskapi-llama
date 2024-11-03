import json
from routing.router import route_query
from graders.retrieval_grader import grade_document_relevance
from graders.hallucination_grader import check_hallucination
from graders.answer_grader import evaluate_answer
from vectorstore.document_loader import retriever as vector_retriever
from utils.web_search import web_search_tool
from utils.rag import generate_answer
from utils.init_llama import llm_json_mode
import os
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3, tavily_api_key='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')


def main(query):
    routing_decision = route_query(llm_json_mode, query)
    datasource = routing_decision.get("datasource")


    if datasource == "vectorstore":
        documents = vector_retriever(query)

        relevant_docs = []
        for doc in documents:
            relevance = grade_document_relevance(llm_json_mode, doc.page_content, query)
            if relevance.get("binary_score") == "yes":
                relevant_docs.append(doc)

        if not relevant_docs:
            web_results = web_search_tool.run(query)
            return web_results
    else:
        web_results = web_search_tool.run(query)
        return web_results
    
    answer = generate_answer(relevant_docs, query)
    hallucination_check = check_hallucination(llm_json_mode, relevant_docs, answer)

    while hallucination_check.get("binary_score") == "no":
        answer = generate_answer(relevant_docs, query)
        hallucination_check = check_hallucination(llm_json_mode, relevant_docs, answer)

    answer_evaluation = evaluate_answer(llm_json_mode, query, answer)

    if answer_evaluation.get("binary_score") == "no":
        web_results = web_search_tool.run(query)
        return web_results
    return answer

if __name__ == "__main__":
    user_query = input("que es una direccion ip")
    response = main(user_query)
    print("Respuesta:", response)
