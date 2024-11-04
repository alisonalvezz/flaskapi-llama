import unittest
from unittest.mock import Mock
from langchain.schema import SystemMessage, HumanMessage
import json

answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

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

def evaluate_answer(llm_json_mode, question, answer):
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        question=question, generation=answer
    )

    result = llm_json_mode.invoke(
        [SystemMessage(content=answer_grader_instructions)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )

    result_content = json.loads(result.content)

    if 'binary_score' in result_content and 'explanation' in result_content:
        binary_score = result_content['binary_score']
        explanation = result_content['explanation']
        return binary_score, explanation
    else:
        return None, None

def check_hallucination(llm_json_mode, documents, generation):
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=documents, generation=generation
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    result_content = json.loads(result.content)
    return result_content

class TestRAGFunctions(unittest.TestCase):

    def setUp(self):
        self.llm_json_mode = Mock()

    def test_evaluate_answer_relevant(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "yes",
            "explanation": "The answer directly addresses the question and is accurate."
        })
        question = "What is the capital of France?"
        answer = "The capital of France is Paris."

        binary_score, explanation = evaluate_answer(self.llm_json_mode, question, answer)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(binary_score, "yes")
        self.assertEqual(explanation, "The answer directly addresses the question and is accurate.")

    def test_evaluate_answer_irrelevant(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "no",
            "explanation": "The answer does not address the question."
        })
        question = "What is the capital of France?"
        answer = "France is a country in Europe."

        binary_score, explanation = evaluate_answer(self.llm_json_mode, question, answer)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(binary_score, "no")
        self.assertEqual(explanation, "The answer does not address the question.")

    def test_evaluate_answer_with_extra_information(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "yes",
            "explanation": "The answer provides additional context but still directly answers the question."
        })
        question = "What is the capital of France?"
        answer = "The capital of France is Paris, which is also known for landmarks like the Eiffel Tower."

        binary_score, explanation = evaluate_answer(self.llm_json_mode, question, answer)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(binary_score, "yes")
        self.assertEqual(explanation, "The answer provides additional context but still directly answers the question.")

    def test_evaluate_answer_incomplete_response(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "score": "yes"
        })
        question = "What is the capital of France?"
        answer = "The capital of France is Paris."

        binary_score, explanation = evaluate_answer(self.llm_json_mode, question, answer)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertIsNone(binary_score)
        self.assertIsNone(explanation)

    def test_check_hallucination_with_empty_docs(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "no",
            "explanation": "The answer contains information not grounded in the provided documents."
        })
        documents = ""
        generation = "This information is not found in the documents."

        result = check_hallucination(self.llm_json_mode, documents, generation)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result['binary_score'], "no")
        self.assertIn("not grounded", result['explanation'])

    def test_check_hallucination_with_irrelevant_docs(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "no",
            "explanation": "The answer contains hallucinated information."
        })
        documents = "These are relevant documents."
        generation = "This is hallucinated information."

        result = check_hallucination(self.llm_json_mode, documents, generation)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result['binary_score'], "no")

    def test_check_hallucination_with_relevant_docs(self):
        self.llm_json_mode.invoke.return_value.content = json.dumps({
            "binary_score": "yes",
            "explanation": "The answer is grounded in the provided documents."
        })
        documents = "These are relevant documents."
        generation = "This information is included in the documents."

        result = check_hallucination(self.llm_json_mode, documents, generation)

        self.llm_json_mode.invoke.assert_called_once()
        self.assertEqual(result['binary_score'], "yes")

if __name__ == "__main__":
    unittest.main()
