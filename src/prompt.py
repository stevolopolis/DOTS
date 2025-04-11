INIT_DIRECT_ANSWERING_INSTRUCTION = "In this step, you need to give the final answer to this question based on the previous thoughts. Follow the mentioned format in the query."

INIT_CoT_INSTRUCTION = "In this step, you need to think step by step with words, solve the problem and get the answer."

INIT_PLANNING_INSTRUCTION = """In this step, you need to reflect on the problem, and describe it in your own words. Analyze how you can decompose the problem into smaller, more manageable sub-tasks. Pay attention to small details, nuances, notes and examples in the problem description."""

INIT_QUERY_REWRITING_INSTRUCTION = """In this step, you need to reveal the Core Question with only a simple sentence and useful information. The output follows the format:
core question:...
Note: Please extract the question-solving information related to the problem, and list them one by one.
useful information:...
"""

INIT_INNER_KNOWLEDGE_RETRIEVAL_INSTRUCTION = """In this step, you need to pinpoint the main ideas and concepts within the question and perform knowledge brainstorm. Review the most relevant theories, principles, or established knowledge relevant to the topic. Think about if the question relates to a specific domain of knowledge. Identify specific terms or keywords that are central to the question. These could be technical terms, jargon, or common phrases associated with the topic.
Present the analysis in a concise, and organized manner. Ensure that the response logically follows from the question and the related knowledge identified. The output format is:
1. The relevant theories are $THEORY_1, $THEORY_2...(where $THEORY_1 is some theory in words);
2. The possible mistakes for solving the question is $Mistake_1, $Mistake_2...
"""

INIT_PROGRAMMING_VERIFIER_INSTRUCTION = """In this step, write and execute a Python code to verify the current results. All the calculation must leverage codes. Before executing the program, you have no idea of the final answer. Don't show it in your comment or code. And don't use the plot function. In this step, start with "```python\n# Now write Python codes to verify the previous answers and use print() to print out the verification result\n# Assume the final result is $ANSWER" """

INIT_PROGRAMMING_SOLVER_INSTRUCTION = """In this step, you need to write and execute a Python code to solve the query. Use the simplest and most straightforward programming methods to solve the problem. For instance, if a query can be efficiently solved using a brute force method, prefer it over heuristic or more complex methods. Utilize any available and commonly-used libraries that can simplify the task or improve code maintainability. All the calculation must leverage codes. Print out the results with print() function. Before executing the program, you have no idea of the final answer. Don't show it in your comment or code. And don't use the plot function.
In this step, start with "```python\n# Now write Python codes to answer this question and use print() to print out the result" """

INIT_VERIFIER_INSTRUCTION = """In this step, you need to carefully verify the correctness of the previous thoughts with natural language. You need to formulate a new verification question, not the same question as before, based on the assumption that the final answer is correct. Then try to find if there is any contradiction. If the results are incorrect, the last line should end up with "The answer is: incorrect". Otherwise, the last line should end with "The answer is: correct"
"""

INIT_VERIFIER_INSTRUCTION_v2 = """In this step, you need to formulate a new verification question, not the same question as before, to verify the result and get the final result. If the verification shows the answer is incorrect, please re-generate an answer with natural language. Otherwise, use the privious answer. Follow the mentioned format in the query in answering the question."""

INIT_AI_TESTER_INSTRUCTION = """Think about if the answer provided is correct, is there any contradiction in conditions.
In this step, start with
"Assume the answer is $ANSWER, we could reconstruct the question to "
The last line of your response could be of the form:
'The answer would cause contradiction in conditions, we should go back to the step 0 to resolve it.\nLet's resolve the question from the very beginning.\n#### STEP 0 ####\n' (without quotes).
OR
'The answer would not cause contradiction in conditions, we should go to the next step.' (without quotes)."""

SUMMARIZATION_INSTRUCTION = """As a clever assistant, your task is to summarize content previously generated. Specifically, for sections involving complex reasoning, you should provide a concise summary highlighting the main takeaway message. Additionally, include a brief explanation to support the main message.
1. Identify Key Points: Focus on the critical elements of the content to determine the main message.
2. Be Concise: Ensure the summary and reasoning are short and to the point.
3. Use Simple Language: Avoid jargon or complex language to make the summary easily understandable.
4. Double-Check Accuracy: Ensure that the summary and reasoning accurately reflect the original content.
5. Highlight Connections: Point out any connections to other relevant content or concepts if applicable.
Please follow the format "Takeaway Message:..., Concise Supporting Reasons:...".
"""

INIT_PROGRAMMING_VERIFIER_DECISION_INSTRUCTION = """Given the execution results, briefly decide the correctness of previous thoughts. The last line should be "After verification, the answer from previous thoughts is Correct/Incorrect".
"""

INIT_RE_ANSWERING_INSTRUCTION = "Since you have found that the previous solution is not correct, try to solve the question step by step with words again and don't make the same mistake. The final answer should follow the mentioned format in the query."

ATOMIC_COMPONENTS_DESCRIPTION = {
    "direct_answering": {
        "description": """If you are very confident that you know the answer to the initial question based on your thoughts and analysis, you may provide the answer directly. However, refrain from choosing this option unless you are absolutely certain of your response. Unless you are highly confident in your response, it may be better to use other modules to explore a more detailed, step-by-step thoughts.""",
        "instruction": INIT_DIRECT_ANSWERING_INSTRUCTION},
    "CoT": {"description": "Answer the question with words step-by-step",
            "instruction": INIT_CoT_INSTRUCTION},
    "planning": {"description": "Make a general plan for the query.",
                 "instruction": INIT_PLANNING_INSTRUCTION},
    "inner_knowledge_retrieve": {
        "description": "Perform knowledge brainstorm to review theories or keywords to the topic.",
        "instruction": INIT_INNER_KNOWLEDGE_RETRIEVAL_INSTRUCTION},
    "query_rewriting": {"description": "Perform knowledge brainstorm to review theories or keywords to the topic.",
                        "instruction": INIT_QUERY_REWRITING_INSTRUCTION},
    "programming_verifier": {
        "description": "Write python code to verify the current thoughts or answers.",
        "instruction": INIT_PROGRAMMING_VERIFIER_INSTRUCTION},
    "programming_solver": {
        "description": "Write python code to solve the initial query.",
        "instruction": INIT_PROGRAMMING_SOLVER_INSTRUCTION},
    "verifier": {
        "description": "Verify the current conditions and solutions.",
        "instruction": INIT_VERIFIER_INSTRUCTION},
    "AI_tester": {
        "description": "Generate more test cases to help solve the problem.",
        "instruction": INIT_AI_TESTER_INSTRUCTION},
    "programming_verifier_decision": {
        "description": "Decide whether to step back to the solution part.",
        "instruction": INIT_PROGRAMMING_VERIFIER_DECISION_INSTRUCTION},
}

INIT_ROUTER_INSTRUCTION = r"""Now you serve as a router for Possible reasoning modules. 
As a router, your task is to select the most appropriate reasoning module from the available options.
1. Evaluate the current thoughts and decide on a module that can best extend the exploration from this point.
2. Provide concise and clear reasons for your selection, ensuring you do not offer specific solutions to the query.
3. Consider the context and potential next steps in the reasoning process to ensure a logical progression.
4. Ensure the selected module aligns with the overarching goals and constraints of the task.
5. Strive for efficiency by selecting a module that balances depth of exploration with the need to reach a timely conclusion.
6. Maintain flexibility in your selections, allowing for adjustments based on new information or changes in the task.
And the last line must be "Next Reasoning Step: $NAME_OF_REASONING_MODULE" (without quote), where $NAME_OF_REASONING_MODULE is one element from {component_lst} and keep the "_" in the name.
"""