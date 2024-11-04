import unittest
from unittest.mock import Mock
from langchain_core.messages import HumanMessage, SystemMessage
import json

router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to IP addresses, Python sorting, and SBERT documentation.
Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.
Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

def route_query(llm_json_mode, query):
    response = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=query)]
    )
    return json.loads(response.content)

class TestRouteQuery(unittest.TestCase):
    def setUp(self):
        self.llm_json_mode = Mock()

    def test_vectorstore_ip_topic(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({"datasource": "vectorstore"})
        query = "What are common issues with IP addresses?"
        result = route_query(self.llm_json_mode, query)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["datasource"], "vectorstore")

    def test_vectorstore_python_sorting(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({"datasource": "vectorstore"})
        query = "How can I sort a list in Python?"
        result = route_query(self.llm_json_mode, query)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["datasource"], "vectorstore")

    def test_vectorstore_sbert_topic(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({"datasource": "vectorstore"})
        query = "What is SBERT used for?"
        result = route_query(self.llm_json_mode, query)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["datasource"], "vectorstore")

    def test_websearch_non_vectorstore_topic(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({"datasource": "websearch"})
        query = "What is the latest trend in AI development?"
        result = route_query(self.llm_json_mode, query)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["datasource"], "websearch")

    def test_websearch_current_events(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({"datasource": "websearch"})
        query = "What are the recent updates in machine learning?"
        result = route_query(self.llm_json_mode, query)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["datasource"], "websearch")

if __name__ == "__main__":
    unittest.main()
