import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

local_llm = "llama3.2:1b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0.7)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to the following topics: 
- IP address
- Sentence BERT (sbert)
- Sentence transformers
- Sorting algorithms in Python

For questions specifically about these topics, respond with 'vectorstore'. 
For all other questions, including general inquiries and current events, respond with 'websearch'.

Examples of websearch questions:
- "What is a language model?"
- "What are the latest trends in technology?"
- "What is the latest news about AI?"
Everything that isnt related to ipaddress, sentence bert, sentence transformers or sorting algorithms should be answered with 'websearch'.

Return only a JSON object with a single key, 'datasource', containing either 'websearch' or 'vectorstore', with no additional text."""

def route_question(llm_json_mode, question):
    print("---ROUTE QUESTION---")
    
    route_question_response = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=question["question"])]
    )
    
    try:
        response_content = json.loads(route_question_response.content)
        source = response_content["datasource"]
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error al procesar la respuesta del modelo: {e}")
        return {"datasource": "error"}
    
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return {"datasource": "websearch"}
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return {"datasource": "vectorstore"}
    else:
        print(f"Fuente desconocida: {source}")
        return {"datasource": "error"}
