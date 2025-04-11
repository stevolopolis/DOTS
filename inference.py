from utils import *
from common import map_with_progress
from src.model_apiV2 import select_llm_model
import argparse
from datetime import datetime
from src.reasoning_modules_general import ReasoningModulesGeneral
import random
# from training_data_generation_v3 import input_instruction
# from MATH_performance_zero_shot_5_for_instruction_tuning import python_code_execute

now = datetime.now()
formatted_time = now.strftime("%m%d%H%M%S")


def extract_actions(paragraph):
    if "query_rewriting" in paragraph:
        action1 = "query_rewriting"
    else:
        action1 = ""
    if "programming_solver" in paragraph:
        action2 = "programming_solver"
    else:
        action2 = "CoT"
    if "verifier" in paragraph:
        action3 = "verifier"
    else:
        action3 = ""
    return [action1, action2, action3, "direct_answering"]


def fn(d):
    task_type = d['id'].split("=")[0]
    # if task_type in ["bbh"]:
    #     return
    if len(args.continual_running_file) > 0:
        if d['id'] in id_set:
            return
    question = d['query']
    if args.pre_defined_traj != 'direct_answering':
        if args.pre_defined_traj == 'dynamic':
            if 'predict_action' in d:
                traj_lst = extract_actions(d['predict_action'])
            else:
                traj_prompt = question
                traj_response = traj_llm.invoke(traj_prompt, temperature=0.0)
                if args.debug == "true":
                    print(traj_response)
                traj_lst = extract_actions(traj_response.split("Required Action: ")[-1])
                d['trajectory_prompt'] = traj_prompt
                d['trajectory llm response'] = traj_response
            if traj_lst is None:
                traj_lst = ["", "CoT", "", "direct_answering"]
        elif args.pre_defined_traj == 'CoT':
            traj_lst = ["", "CoT", "", "direct_answering"]
        elif args.pre_defined_traj == 'PoT':
            traj_lst = ["", "programming_solver", "", "direct_answering"]
        elif args.pre_defined_traj == 'PoT_verifier':
            traj_lst = ["", "programming_solver", "verifier", "direct_answering"]
        else:
            traj_lst = ["", "CoT", "", "direct_answering"]
        prompt = question
        _, dialogue = reasoning_modules.think_reply(query=prompt, modules=traj_lst, temperature=0.0, id='')
        d['pred_action_trajectory'] = traj_lst
    else:
        prompt = question
        response = llm.invoke(prompt=prompt, temperature=0)
        dialogue = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        if "```python" in response or "```\npython" in response:
            response += '```'
            exec_result = python_code_execute(response)
            new_prompt = "After execution, we get:\n" + exec_result + f"In this step, you need to give the final answer to this question based on the previous thoughts. Follow the mentioned format in the query."
            dialogue.append({"role": "user", "content": new_prompt})
            new_response = llm.invoke(prompt='', final_messages=dialogue, temperature=0)
            dialogue.append({"role": "user", "content": new_response})
    d['dialogue'] = dialogue
    d['pred'] = dialogue[-1]['content']
    new_data.append(d)
    global save_step
    if len(new_data) > save_step:
        save_step += 500
        save_json(args.output_file, new_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--pre_defined_traj", type=str, default="direct_answering")
    parser.add_argument("--input_file", type=str,
                        default="")
    parser.add_argument("--output_file", type=str,
                        default="")
    parser.add_argument("--trajectory_api_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--trajectory_host", type=str, default="localhost")
    parser.add_argument("--continual_running_file", type=str, default="")
    parser.add_argument("--debug", type=str, default="false")
    args = parser.parse_args()
    data = read_json(args.input_file)

    if args.debug == "true":
        debug = "debug_"
        random.shuffle(data)
        data = data[:50]
    else:
        debug = ""

    if len(args.continual_running_file) > 0:
        continual_data = read_json(args.continual_running_file)
        id_set = set()
        for cd in continual_data:
            id_set.add(cd['id'])
        new_data = continual_data
    else:
        new_data = []

    model_name = args.api_model if '/' not in args.api_model else args.api_model.split('/')[-2]
    if args.pre_defined_traj == '':
        traj_llm = select_llm_model(llm_model=args.trajectory_api_model, host=args.trajectory_host)
    llm = select_llm_model(llm_model=args.api_model, host=args.host)
    reasoning_modules = ReasoningModulesGeneral(llm_name=args.api_model, host=args.host)
    save_step = 0
    map_with_progress(f=fn, xs=data, num_threads=100)
    save_json(
        args.output_file,
        new_data)
    correct_num = 0