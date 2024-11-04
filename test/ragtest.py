import unittest
from unittest.mock import Mock
from langchain_core.messages import HumanMessage
from dataclasses import dataclass

@dataclass
class Document:
    page_content: str

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

def generate_answer(llm_json_mode, relevant_docs, question):
    docs_txt = format_docs(relevant_docs)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm_json_mode.invoke([HumanMessage(content=rag_prompt_formatted)])
    return generation.content

class TestGenerateAnswer(unittest.TestCase):
    def setUp(self):
        self.llm_json_mode = Mock()

    def test_generate_answer_with_relevant_docs(self):
        self.llm_json_mode.invoke.return_value.content = "Python sorting allows you to organize lists and dictionaries."
        relevant_docs = [Document("Python sorting techniques include bubble sort and quicksort.")]
        question = "What are common sorting methods in Python?"
        
        result = generate_answer(self.llm_json_mode, relevant_docs, question)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result, "Python sorting allows you to organize lists and dictionaries.")

    def test_generate_answer_with_irrelevant_docs(self):
        self.llm_json_mode.invoke.return_value.content = "No relevant information found."
        irrelevant_docs = [Document("This document discusses ancient history and archaeology.")]
        question = "What are common sorting methods in Python?"
        
        result = generate_answer(self.llm_json_mode, irrelevant_docs, question)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result, "No relevant information found.")

    def test_generate_answer_with_multiple_docs(self):
        self.llm_json_mode.invoke.return_value.content = "Python sorting methods include quicksort and mergesort."
        multiple_docs = [
            Document("Python sorting techniques include bubble sort and quicksort."),
            Document("Mergesort is a more advanced sorting algorithm in Python.")
        ]
        question = "What are common sorting methods in Python?"
        
        result = generate_answer(self.llm_json_mode, multiple_docs, question)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result, "Python sorting methods include quicksort and mergesort.")

    def test_generate_answer_with_empty_docs(self):
        self.llm_json_mode.invoke.return_value.content = "No context provided."
        empty_docs = []
        question = "What are common sorting methods in Python?"
        
        result = generate_answer(self.llm_json_mode, empty_docs, question)
        
        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result, "No context provided.")

    def test_format_docs(self):
        docs = [
            Document("Document content 1."),
            Document("Document content 2.")
        ]
        formatted_docs = format_docs(docs)
        self.assertEqual(formatted_docs, "Document content 1.\n\nDocument content 2.")

if __name__ == "__main__":
    unittest.main()
