from langchain_core.messages import HumanMessage, SystemMessage
from utils.init_llama import llm_json_mode

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

def evaluate_answer(question, answer):
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        question=question, generation=answer
    )

    result = llm_json_mode.invoke(
        [SystemMessage(content=answer_grader_instructions)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )

    print("Result:", result)

    if 'binary_score' in result and 'explanation' in result:
        binary_score = result['binary_score']
        explanation = result['explanation']
        return binary_score, explanation
    else:
        print("El resultado no contiene las claves esperadas.")
        return None, None

