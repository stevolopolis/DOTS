import random

import common
from utils import *
import argparse
import sympy
from src.eval_utils import *
from datetime import datetime
from tqdm import tqdm
from common import ANSWER_PATTERN, EQUALITY_TEMPLATE
from src.model_api import select_llm_model

now = datetime.now()
formatted_time = now.strftime("%m%d%H%M%S")


def math_answer_extract(x):
    if isinstance(x, list):
        return x[-1]['content']
    else:
        return x


def math_answer_checking(correct_ans, predicted_ans):
    answer = extract_math_answer(predicted_ans, False)
    if isinstance(correct_ans, list):
        pass
        # return 1 if compare_answer_with_groundtruth(answer, *correct_ans) else 0
    elif isinstance(correct_ans, str):
        correct_ans = [str(correct_ans), number_it(correct_ans)]
    elif isinstance(correct_ans, int) or isinstance(correct_ans, float):
        correct_ans = [str(correct_ans), correct_ans]
        # value = number_it(correct_ans)
    return 1 if compare_answer_with_groundtruth(answer, *correct_ans) else 0


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--save_file", type=str, default="")
    parser.add_argument("--debug", type=str, default="true")
    args = parser.parse_args()
    file = args.file
    save_file = args.save_file
    data = read_json(file)
    if args.debug == 'true':
        random.shuffle(data)
        data = data[:100]

    data_dict = {}
    new_data = []


    def fn(d):
        global data_dict
        id = d['id']
        # d['pred'] = d['output']
        task = id.split("=")[0]
        if task == 'bbh':
            sub_task = "_".join(id.split("=")[-1].split("_")[:-1])
            if sub_task == 'word_sorting':
                correctness = get_correctness_word_sorting(predicted_ans=d['pred'], correct_ans=d['answer'])
            elif sub_task == 'dyck_languages':
                correctness = get_correctness_dyck_language(predicted_ans=d['pred'], correct_ans=d['answer'])
            elif sub_task == 'salient_translation_error_detection' or sub_task == 'penguins_in_a_table':
                correctness = get_salient_translation_error_detection(predicted_ans=d['pred'], correct_ans=d['answer'])
            else:
                correctness = bbh_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
        elif task == 'game24':
            correctness = game24_check_answer(d['pred'], d['query'])
        elif task == 'MATH':
            correctness = math_answer_checking_simple_eval(predicted_ans=d['pred'], correct_ans=d['answer'])
        elif task == "theoremqa" or task == "deepmind" or task == "gpqa":
            correctness = math_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
        elif task == "mmlu_pro":
            correctness = mmlu_pro_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
        elif task == "sqa":
            correctness = strategyQA_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
        elif task == "drop":
            correctness = drop_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
        else:
            return
        d['correctness'] = correctness
        new_data.append(d)
        if task not in data_dict:
            data_dict[task] = []
        data_dict[task].append(correctness)
    common.map_with_progress(f=fn, xs=data, num_threads=100)

    for d in new_data:
        id = d['id']
        correctness = d['correctness']
        task = id.split("=")[0]
        if task not in data_dict:
            data_dict[task] = []
        data_dict[task].append(correctness)

    for key in data_dict:
        print(f"{key}: {sum(data_dict[key]) / len(data_dict[key])}")
    if len(args.save_file) > 0:
        save_json(filename=save_file, data=data)
