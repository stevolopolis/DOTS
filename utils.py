import json
import pickle
import re
from src.eval_utils import *
from src.safe_execute import safe_exec
import sympy
from common import EQUALITY_TEMPLATE
from src.model_api import select_llm_model


def save_json(filename, data):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def read_pickle(file):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    return data

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def extract_model_answer(model_answer):
    things_to_open_and_close = {
        "\\frac{": "}",
        "\\boxed{": "}",
        "\\[": "\\]",
        "$$": "$$",
        "$": "$",
        "{": "}",
        "(": ")",
    }

    position = 0
    math_parts = []
    things_to_close = []
    while position < len(model_answer):
        remaining = model_answer[position:]
        if len(things_to_close) != 0:
            thing_to_close = things_to_close[-1]
            if remaining.startswith(thing_to_close):
                things_to_close.pop()
                position += len(thing_to_close)
                math_parts[-1] += thing_to_close
                continue
        next = False
        for special_symbol_start in things_to_open_and_close.keys():
            if remaining.startswith(special_symbol_start):
                if len(things_to_close) == 0:
                    math_parts.append(special_symbol_start)
                else:
                    math_parts[-1] += special_symbol_start
                things_to_close.append(things_to_open_and_close[special_symbol_start])
                position += len(special_symbol_start)
                next = True
                break
        if next:
            continue
        if len(things_to_close) == 0:
            if remaining.startswith(" "):
                position += 1
                continue
            part_until_next_space = remaining.split(" ")[0]
            if len(part_until_next_space) != 1 and part_until_next_space.endswith("."):
                part_until_next_space = part_until_next_space[:-1]
            try:
                math_parts.append(str(int(part_until_next_space)))
            except:
                try:
                    math_parts.append(str(float(part_until_next_space)))
                except:
                    if re.match("^[0-9]", part_until_next_space):
                        math_parts.append(part_until_next_space)
                    pass
            position += len(part_until_next_space)
            continue
        else:
            math_parts[-1] += remaining[0]
            position += 1

    if len(math_parts) == 0:
        return ""
    else:
        answer = math_parts[-1]

    if answer.endswith("."):
        answer = answer[:-1]

    if answer.startswith("$$") and answer.endswith("$$"):
        answer = answer[2:-2]
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1]
    if answer.startswith("\\[") and answer.endswith("\\]"):
        answer = answer[2:-2]

    if answer.startswith("\\boxed{"):
        answer = answer[7:-1]

    if "=" in answer:
        answer = answer.split("=")[1]

    answer = answer.replace("∞", "\\infty")

    return answer


def is_math_correct(model_answer, correct_answer):
    model_answer_lines = reversed(model_answer.split("\n"))
    for line in model_answer_lines:
        model_answer_extracted = extract_model_answer(line)
        if len(model_answer_extracted.strip()) == 0:
            continue
        try:
            model_answer_extracted = strip_string(model_answer_extracted)
        except:
            pass
        correct_answer = strip_string(
            remove_boxed(last_boxed_only_string(correct_answer))
        )
        return model_answer_extracted == correct_answer
    return None

def save_list_to_txt(file_path, data_list):
    """
    Save a list to a .txt file, with each element on a new line.

    :param file_path: The path to the .txt file.
    :param data_list: The list of elements to save.
    """
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")


def read_txt_to_list(file_path):
    """
    Read a .txt file into a list, with each line as an element.

    :param file_path: The path to the .txt file.
    :return: A list of elements read from the file.
    """
    with open(file_path, 'r') as file:
        data_list = file.readlines()
    return [line.strip() for line in data_list]


MATH_QUESTION_TEMPLATE = """Solve the following math problem. The last line of your response should be of the form 'Answer: \\boxed{{$ANSWER}}' (without quotes) where $ANSWER is the answer to the problem. If the answer is a fraction, do not convert it to a decimal.
Question: {Question}
""".strip()

BBH_QUESTION_TEMPLATE = """Solve the following problem. The last line of your response should be of the form 'Answer: $ANSWER' (without quotes) where $ANSWER is the final answer to the problem. The $ANSWER could be a choice (e.g., (A)), a word (e.g., invalid, yes, no), a number (e.g., -35), or a symbol (e.g, ]]), etc.
Question: {Question}
""".strip()

Game24_QUESTION_TEMPLATE = """Use numbers and basic arithmetic operations (+ - * / ( )) to obtain 24. It uses each input number exactly once and no other numbers to reach 24. You could change the order of the numbers. The final answer should be a math expression that uses the four input numbers to reach 24. The last line of your response should be of the form 'Answer: $ANSWER' (without quotes) where $ANSWER is the number and symbol to reach 24, e.g, 'Answer: (1-1)+4*6'.
Question: {Question}
""".strip()

mmlu_pro_QUESTION_TEMPLATE = """Answer the following multiple choice question. Only one choice is correct. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is only one of A,B,C,D,E..., depending how many options you have.
Question: {Question}
{Choices}
""".strip()

StrategyQA_QUESTION_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is only one of A,B.
Question: {Question}
{Choices}
""".strip()


def math_answer_extract(x):
    if isinstance(x, list):
        return x[-1]['content']
    else:
        return x


def math_answer_checking(correct_ans, predicted_ans):
    answer = extract_math_answer(predicted_ans, False)
    if isinstance(correct_ans, list):
        return 1 if compare_answer_with_groundtruth(answer, *correct_ans) else 0
    else:
        value = number_it(correct_ans)
        return 1 if compare_answer_with_groundtruth(groundtruth_str=correct_ans, answer=answer,
                                                    groundtruth_num=value) else 0


def check_equality(expr1: str, expr2: str):
    llm = select_llm_model(llm_model='gpt-4o-mini', host='')
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = llm.invoke(prompt=prompt, temperature=0)
    return response.lower().strip() == "yes"


def math_answer_checking_simple_eval(correct_ans, predicted_ans):
    # if 'Answer:' in predicted_ans:
    #     match = re.search(ANSWER_PATTERN, predicted_ans)
    #     extracted_answer = match.group(1) if match else None
    # else:
    #     extracted_answer = extract_math_answer(predicted_ans, False)
    extracted_answer = extract_math_answer(predicted_ans, False)
    correct_ans1 = correct_ans[0]
    correct_ans2 = correct_ans[1]
    return 1 if check_equality(extracted_answer, correct_ans1) or check_equality(extracted_answer, correct_ans2) else 0


def bbh_answer_extract(x):
    if isinstance(x, list):
        x = x[-1]['content']
    x = x.replace("**", "")
    x = x.replace("$", "")
    x = x.split('Answer:')[-1].replace(" ", "")
    if ")" in x:
        x = x.split(')')[0] + ")"
    if "(" and ")" in x and len(x.split('(')[-1].replace(')', '')) == 1:
        x = x.split('(')[-1].replace(')', '')
    return x


def bbh_answer_extract_dyck(x):
    x = x.split('Answer:')[-1].replace(" ", "")
    return x


def bbh_answer_checking(correct_ans, predicted_ans):
    if "(" and ")" in correct_ans and len(correct_ans.split('(')[-1].replace(')', '')) == 1:
        correct_ans = correct_ans.split('(')[-1].replace(')', '')
    predicted_ans = bbh_answer_extract(predicted_ans)
    try:
        if correct_ans.lower() == predicted_ans.lower():
            return 1
        else:
            return 0
    except:
        return 0


def game24_check_answer(pred, question):
    x = pred.lower().replace("**", "")
    predicted_ans = x.split('answer:')[-1].replace(" ", "").replace("=24", "")
    four_number = question.split("Here are the four numbers to use: ")[-1]
    four_number_lst = four_number.split(" ")
    for num in four_number_lst:
        if num not in predicted_ans:
            return 0
    try:
        return 1 if sympy.simplify(predicted_ans) == 24 else 0
    except:
        return 0


def GPQA_answer_extract(x):
    x = x.replace("**", "")
    x = x.split('Answer:')[-1].replace(" ", "")
    x = x.split(')')[0]
    return x


def GPQA_answer_checking(correct_ans, predicted_ans):
    predicted_ans = GPQA_answer_extract(predicted_ans)
    try:
        return 1 if correct_ans == predicted_ans else 0
    except:
        return 0


def mmlu_pro_answer_checking(correct_ans, predicted_ans):
    predicted_ans = predicted_ans.replace("**", "")
    predicted_ans = predicted_ans.split('Answer: ')[-1]
    predicted_ans = predicted_ans.split('Final answer: ')[-1]
    if ")" in predicted_ans:
        predicted_ans = predicted_ans.split(')')[0][-1]
    try:
        return 1 if correct_ans == predicted_ans else 0
    except:
        return 0


def strategyQA_answer_checking(correct_ans, predicted_ans):
    predicted_ans = predicted_ans.replace("**", "")
    predicted_ans = predicted_ans.replace('$', '')
    predicted_ans = predicted_ans.replace('\n', '')
    predicted_ans = predicted_ans.split('Answer: ')[-1]
    predicted_ans = predicted_ans.split('answer is:')[-1]
    predicted_ans = predicted_ans.split('####')[0]
    predicted_ans = predicted_ans.replace(' ', '')
    try:
        return 1 if str(correct_ans).lower() == predicted_ans.lower() else 0
    except:
        return 0


def folio_answer_checking(correct_ans, predicted_ans):
    predicted_ans = predicted_ans.replace("**", "")
    predicted_ans = predicted_ans.replace('$', '')
    predicted_ans = predicted_ans.split('Answer: ')[-1]
    predicted_ans = predicted_ans.split('Final answer: ')[-1]
    predicted_ans = predicted_ans.split('####')[0]
    predicted_ans = predicted_ans.replace(' ', '')
    predicted_ans = predicted_ans.replace('\n', '')

    try:
        return 1 if correct_ans.lower() == predicted_ans.lower() else 0
    except:
        return 0


def drop_answer_checking(correct_ans, predicted_ans):
    predicted_ans = predicted_ans.replace("**", "")
    predicted_ans = predicted_ans.replace('$', '')
    predicted_ans = predicted_ans.split('Answer: ')[-1]
    predicted_ans = predicted_ans.replace('\n', '')
    predicted_ans = predicted_ans.replace(',', '')
    predicted_ans = predicted_ans.replace('%', '')
    try:
        return 1 if correct_ans.lower() == predicted_ans.lower() else 0
    except:
        return 0


def clean_string(s):
    cleaned = re.sub(r'[^a-zA-Z]', '', s)
    return cleaned.lower()


def get_correctness_word_sorting(correct_ans, predicted_ans):
    predicted_ans = predicted_ans.split("Answer: ")[-1]
    predicted_ans = predicted_ans.split('\n####Step 2')[0]
    predicted_ans = predicted_ans.split(':')[-1]
    predicted_ans = clean_string(predicted_ans)
    correct_answer = clean_string(correct_ans)
    return 1 if predicted_ans == correct_answer else 0


def get_correctness_dyck_language(correct_ans, predicted_ans):
    pred = predicted_ans.split("Answer: ")[-1]
    pred = pred.replace("$", "")
    pred = pred.replace("`", "")
    pred = pred.replace(" ", "")
    correct_answers = correct_ans.replace(" ", "")
    pred = pred[-len(correct_answers):]
    return 1 if pred == correct_answers else 0


def get_salient_translation_error_detection(correct_ans, predicted_ans):
    correct_ans = correct_ans.replace('(', '').replace(')', '')
    predicted_ans = bbh_answer_extract(predicted_ans)
    predicted_ans = predicted_ans.replace('(', '').replace(')', '')
    try:
        if correct_ans.lower() == predicted_ans.lower():
            return 1
        else:
            return 0
    except:
        return 0

def read_GameOf24_dataset():
    cache_dir = "dataset/Game24"
    csv_data = pd.read_csv(f"{cache_dir}/GameOf24.csv")
    print(csv_data.columns)
    print(csv_data.shape)
    print(csv_data.head())

    dataset = {}
    subtasks = ["game_of_24_level1", "game_of_24_level2", "game_of_24_level3"]
    for subtask in subtasks:
        dataset[subtask] = {
            "description": "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. It uses each input number exactly once and no other numbers to reach 24.",
            "format": "The final answer should be a math expression that uses the four input numbers to reach 24.",
            "instances": []
        }

    rank_list = csv_data["Rank"].tolist()
    puzzles_list = csv_data["Puzzles"].tolist()
    amt_list = csv_data["AMT (s)"].tolist()
    solved_rate_list = csv_data["Solved rate"].tolist()

    for i in range(len(rank_list)):
        if 0 <= i < 500:
            subtask = subtasks[0]
        elif 500 <= i < 1000:
            subtask = subtasks[1]
        else:
            subtask = subtasks[2]

        instance = {}
        instance["Id"] = subtask + "@" + str(i)
        instance["input"] = f"Here are the four numbers to use: {puzzles_list[i]}"
        instance["target"] = "24"
        instance["meta_info"] = {
            "rank": rank_list[i],
            "amt": amt_list[i],
            "solved_rate": solved_rate_list[i]
        }
        dataset[subtask]["instances"].append(instance)
    return dataset


def clean_GameOf24_expression(answer):
    if answer.endswith("."):
        answer = answer[:-1]
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    if answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    if answer.startswith("`") and answer.endswith("`"):
        answer = answer[1:-1]
    if answer.startswith("$$") and answer.endswith("$$"):
        answer = answer[2:-2]
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1]
    if answer.startswith("\\[") and answer.endswith("\\]"):
        answer = answer[2:-2]
    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2]
    if answer.startswith("\\text{") and answer.endswith("}"):
        answer = answer[6:-1]

    return answer

def Game24_answer_extract(x):
    x = x.lower().replace("**", "")
    return x.split('answer:')[-1].replace(" ", "").replace("=24", "")


def Game24_answer_checking(correct_ans, predicted_ans):
    try:
        return 1 if sympy.simplify(predicted_ans) == 24 else 0
    except:
        return 0

def mmlu_pro_answer_extract(x):
    x = x.replace("**", "")
    x = x.split('Answer: ')[-1]
    x = x.split(')')[0]
    return x


def math_id(d):
    id = d['id']
    query = MATH_QUESTION_TEMPLATE.format(Question=d['problem'])
    answer = d['answer']
    return id, query, answer


def bbh_id(d):
    id = d['id']
    query = BBH_QUESTION_TEMPLATE.format(Question=d['problem'])
    correct_answer = d['answer']
    return id, query, correct_answer


def Game24_id(d):
    id = d['Id']
    query = Game24_QUESTION_TEMPLATE.format(Question=d['input'])
    correct_answer = 24
    return id, query, correct_answer


def theoremQA_id(d):
    query = d['Question']
    correct_answer = d['Answer']
    id = d['Question']
    return id, query, correct_answer


def mmlu_id(d):
    id = d['question_id']
    choices = ''
    for j, o in enumerate(d['options']):
        choices += f"{chr(65 + (j % 26))}) {o}\n"
    query = mmlu_pro_QUESTION_TEMPLATE.format(Question=d['question'], Choices=choices)
    correct_answer = d['answer']
    return id, query, correct_answer


def strategyQA_id(d):
    id = d["qid"]
    question = StrategyQA_QUESTION_TEMPLATE.format(Question=d['question'], Choices="A) True\nB) False")
    correct_answer = 'A' if d['answer'] else 'B'
    return id, question, correct_answer


def strategyQA_answer_extract(x):
    x = x.replace("**", "")
    x = x.split('Answer: ')[-1]
    x = x.split(')')[0]
    return x

def python_code_execute(code):
    import re
    def extract_python_code(text):
        if "```python" in code:
            pattern = r'```python(.*?)```'
        else:
            pattern = r'python(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    if "```python" in code or "```\npython" in code:
        if 'pip install' in code:
            return "An error occurred: You cannot install any Python packages."
        code_matches = extract_python_code(code)
        if code_matches:
            code = code_matches[0]
        else:
            return "An error occurred: No valid Python code block with ```python found."
    try:
        output = safe_exec(code)
    except:
        output = ""
    return output

def construct_explain_prompt(query, action, id=None):
    prompt4 = f"""Action Categories:
1. Understanding process:
query_rewriting: Rewrite the question and answer it.
planning: Make a plan to solve the given question.
2. Solving process:
chain of thought: For step-by-step reasoning with language.
programming: For programming solver.
3. Verification process:
verifier: To check the correctness of the solution.

Task Instruction: For the given question, explain why the above Required actions are necessary based on the Analysis Rubric within 50 words. Don't need to solve the question. If the Required Action is 'chain_of_thought', analyze why we shouldn't use the code. 

Query1: {{
Find $2 \cdot 5^{{-1}} + 8 \cdot 11^{{-1}} \pmod{{56}}$.\n\nExpress your answer as an integer from $0$ to $55$, inclusive.
}}
Required Action1: programming, verifier
Explanation1:
1. Question Type: Modular arithmetic problem;
2. Understanding process: No query rewriting or planning needed as the problem is clearly stated;
3. Solving process: Use programming_solver because:
    Modular arithmetic can be efficiently implemented using programming libraries like sympy
    The calculation involves multiple steps that are prone to human error
    Not using chain_of_thought because the mathematical operations are well-defined and don't require elaborate reasoning
4. Verification Process: Use verifier because:
    The result can be easily checked by substituting back into the original equation
    The answer must be within a specific range (0 to 55), which needs verification
5. Justification:
    Programming_solver is more efficient and accurate for this type of calculation
    Verifier ensures the correctness of the result and adherence to the given constraints

Query2: {{
What is the largest four-digit number whose digits add up to 16?
}}
Required Action2: programming, verifier
Explanation2:
1. Question Type: Digit counting and optimization problem;
2. Understanding process: No query rewriting or planning needed as the problem is straightforward;
3. Solving process: Use programming_solver because:
    It can efficiently generate and check all four-digit numbers
    The problem has a clear, systematic approach that can be easily coded
    Not using chain_of_thought because the solution doesn't require complex logical reasoning
4. Verification Process: Use verifier because:
    The result needs to be checked for being the largest possible number
    The sum of digits must be verified to equal 16
5. Justification:
    Programming_solver is faster and more accurate for exhaustive searches
    Verifier ensures the solution meets all criteria (largest number, sum of digits)


Query3: {{
Let $f(x) = x^2-3x$. For what values of $x$ is $f(f(x)) = f(x)$? Enter all the solutions, separated by commas.
}}
Required Action3: chain of thought, verifier
Explanation3:
1. Question Type: Functional equation solving problem;
2. Understanding process: No query rewriting or planning needed as the problem is clearly defined;
3. Solving process: Use chain_of_thought because:
    The problem requires algebraic manipulation and reasoning
    It's not easily solvable through simple programming without advanced symbolic manipulation
    Not using programming_solver because program cannot solve the equation well.
4. Verification Process: Use verifier because:
    Each solution needs to be checked by substituting back into the original equation
    Multiple solutions may exist, and all need to be verified
5. Justification:
    Chain_of_thought allows for clear demonstration of the problem-solving steps
    Verifier ensures all solutions are correct and no solutions are missed

Query4: {{
What is the value of $a$ for which $\\frac{{1}}{{\\text{{log}}_2a}} + \\frac{{1}}{{\\text{{log}}_3a}} + \\frac{{1}}{{\\text{{log}}_4a}} = 1$?
}}
Required Action4: chain of thought
Explanation3:
1. Question Type: Equation solving problem;
2. Understanding process: No query rewriting or planning needed as the problem is clearly defined;
3. Solving process: Use chain_of_thought because:
    The problem requires algebraic manipulation and reasoning
    Not using programming_solver because program cannot solve the equation well
4. Verification Process: Use verifier because:
    Verification is not required because the question is easy to solve and don't need to be verified
5. Justification:
    Chain_of_thought allows for clear demonstration of the problem-solving steps
    Verifier ensures all solutions are correct and no solutions are missed


Query5:{{
{query}
}}
Required Action5: {action}
Explanation5:"""
    return prompt4



