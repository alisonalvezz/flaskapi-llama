import unittest
from unittest.mock import Mock
from langchain.schema import SystemMessage, HumanMessage
import json

hallucination_grader_instructions = """
You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys: 'binary_score', which is 'yes' or 'no' to indicate whether the STUDENT ANSWER is grounded in the FACTS, and 'explanation', which contains an explanation of the score."""

def check_hallucination(llm_json_mode, documents, generation):
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=documents, generation=generation
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    return json.loads(result.content)

class TestCheckHallucination(unittest.TestCase):
    def setUp(self):
        self.llm_json_mode = Mock()

    def test_check_hallucination_relevant_answer(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "yes",
            "explanation": "The answer is grounded in the provided facts."
        })
        documents = "The Earth revolves around the Sun."
        generation = "The Earth orbits the Sun."

        result = check_hallucination(self.llm_json_mode, documents, generation)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["binary_score"], "yes")
        self.assertEqual(result["explanation"], "The answer is grounded in the provided facts.")

    def test_check_hallucination_irrelevant_answer(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "no",
            "explanation": "The answer mentions that the Earth is flat, which is not grounded in the provided facts."
        })
        documents = "The Earth is round and revolves around the Sun."
        generation = "The Earth is flat and does not move."

        result = check_hallucination(self.llm_json_mode, documents, generation)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["binary_score"], "no")
        self.assertEqual(result["explanation"], "The answer mentions that the Earth is flat, which is not grounded in the provided facts.")

    def test_check_hallucination_partial_relevance(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "no",
            "explanation": "The answer is partially relevant but includes unsupported claims."
        })
        documents = "Python is widely used for data analysis and web development."
        generation = "Python is used for data analysis, web development, and controlling nuclear reactors."

        result = check_hallucination(self.llm_json_mode, documents, generation)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["binary_score"], "no")
        self.assertEqual(result["explanation"], "The answer is partially relevant but includes unsupported claims.")

    def test_check_hallucination_empty_documents(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "no",
            "explanation": "No facts were provided to assess the answer."
        })
        documents = ""
        generation = "Python is a popular programming language."

        result = check_hallucination(self.llm_json_mode, documents, generation)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result["binary_score"], "no")
        self.assertEqual(result["explanation"], "No facts were provided to assess the answer.")

if __name__ == "__main__":
    unittest.main()
