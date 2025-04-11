import random
import common
from utils import *
import argparse
from datetime import datetime
from src.model_api import select_llm_model
now = datetime.now()
formatted_time = now.strftime("%m%d%H%M%S")

def get_internalized_result(d):
    prompt = d['query']
    response = llm.invoke(prompt=prompt, temperature=0)
    dialogue = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    if "```python" in response or "```\npython" in response:
        response += '```'
        exec_result = python_code_execute(response)
        if "An error occurred:" in exec_result:
            new_prompt = "After execution, we get an error:\n" + exec_result + f"Please refine the code and try again."
            dialogue.append({"role": "user", "content": new_prompt})
            new_response = llm.invoke(prompt='', final_messages=dialogue, temperature=0)
            dialogue.append({"role": "assistant", "content": new_response})
            exec_result = python_code_execute(new_response)
        new_prompt = "After execution, we get:\n" + exec_result + f"In this step, you need to give the final answer to this question based on the previous thoughts. Follow the mentioned format in the query."
        dialogue.append({"role": "user", "content": new_prompt})
        new_response = llm.invoke(prompt='', final_messages=dialogue, temperature=0)
        dialogue.append({"role": "assistant", "content": new_response})
    d['dialogue'] = dialogue
    d['pred'] = dialogue[-1]['content']


def eval_instance(d):
    global data_dict
    id = d['id']
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
    elif task == "theoremqa" or task == "deepmind":
        correctness = math_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
    elif task == "mmlu_pro":
        correctness = mmlu_pro_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
    elif task == "sqa":
        correctness = strategyQA_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
    elif task == "drop":
        correctness = drop_answer_checking(predicted_ans=d['pred'], correct_ans=d['answer'])
    else:
        raise Exception(f"Not supported task {task}.")
    d['correctness'] = correctness
    new_data.append(d)
    if task not in data_dict:
        data_dict[task] = []
    data_dict[task].append(correctness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model", type=str, default="")
    parser.add_argument("--host", type=str, default="")
    parser.add_argument("--debug", type=lambda x: x.lower() == 'true', default="true")
    parser.add_argument("--input_data_path", type=str, default="data/mixture_test.json")
    args = parser.parse_args()
    data = read_json(args.input_data_path)
    if args.debug:
        random.shuffle(data)
        data = data[:50]
        debug = 'debug_'
    else:
        debug = ''
    save_file = f'{args.input_data_path}_eval_{debug}{formatted_time}.json'

    data_dict = {}
    new_data = []
    llm = select_llm_model(llm_model=args.api_model, host=args.host)
    print('inferencing...')
    common.map_with_progress(f=get_internalized_result, xs=data, num_threads=100)
    print('evaluating')
    common.map_with_progress(f=eval_instance, xs=data, num_threads=100)

    for d in new_data:
        id = d['id']
        correctness = d['correctness']
        task = id.split("=")[0]
        if task not in data_dict:
            data_dict[task] = []
        data_dict[task].append(correctness)

    for key in data_dict:
        print(f"{key}: {sum(data_dict[key]) / len(data_dict[key])}")
    save_json(filename=save_file, data=data)
