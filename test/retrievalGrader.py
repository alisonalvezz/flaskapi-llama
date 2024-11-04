import unittest
from unittest.mock import Mock
from langchain_core.messages import HumanMessage, SystemMessage
import json

doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

def grade_document_relevance(llm_json_mode, document, question):
    doc_grader_prompt_formatted = doc_grader_prompt.format(
        document=document, question=question
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=doc_grader_instructions)]
        + [HumanMessage(content=doc_grader_prompt_formatted)]
    )
    return json.loads(result.content)

class TestGradeDocumentRelevance(unittest.TestCase):
    def setUp(self):
        self.llm_json_mode = Mock()

    def test_relevant_document(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({"binary_score": "yes"})
        document = "This document contains information about Python sorting algorithms."
        question = "How do I sort a list in Python?"
        result = grade_document_relevance(self.llm_json_mode, document, question)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["binary_score"], "yes")

    def test_irrelevant_document(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({"binary_score": "no"})
        document = "This document contains historical information about ancient civilizations."
        question = "How do I sort a list in Python?"
        result = grade_document_relevance(self.llm_json_mode, document, question)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["binary_score"], "no")

    def test_partial_relevance(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({"binary_score": "yes"})
        document = "Python sorting and IP addresses are important topics in computer science."
        question = "What are common Python sorting techniques?"
        result = grade_document_relevance(self.llm_json_mode, document, question)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["binary_score"], "yes")

if __name__ == "__main__":
    unittest.main()
