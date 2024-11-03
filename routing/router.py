import json
from langchain_core.messages import HumanMessage, SystemMessage


router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.
Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

def route_query(llm_json_mode, query):
    """
    Función para evaluar la consulta y decidir el datasource.

    Parameters:
    llm_json_mode (llm instance): El modelo de lenguaje utilizado.
    query (str): La consulta del usuario.

    Returns:
    dict: JSON con la clave 'datasource' que indica si usar 'websearch' o 'vectorstore'.
    """
    response = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=query)]
    )
    return json.loads(response.content)
