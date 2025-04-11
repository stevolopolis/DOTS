import random
import sympy
from src.LLM_reasoning_workflow_general import LLMReasoningWorkflowGeneral
from datetime import datetime
import argparse
import common
from utils import *

now = datetime.now()
formatted_time = now.strftime("%m%d%H%M%S")


def eval_dataset(dataset, json_file, llm, layers, answer_extract_func, checking_func, id_func, data=None, DEBUG=True):
    if data is None:
        data = read_json(json_file)
    if DEBUG:
        random.shuffle(data)
        data = data[:5]
        debug = "debug_"
        examine_time = 4
    else:
        debug = ''
        examine_time = 2
    reasoning_workflow = LLMReasoningWorkflowGeneral(llm_name=llm, layers=layers, host=args.host)
    traj_dict = {
        "query_rewriting": ["query_rewriting", "CoT", "", "direct_answering"],
        "CoT": ["", "CoT", "", "direct_answering"],
        "verifier": ["", "CoT", "verifier", "direct_answering"],
        "query_rewriting_programming": ["query_rewriting", "programming_solver", "", "direct_answering"],
        "programming_solver_verifier": ["", "programming_solver", "verifier", "direct_answering"],
        'planning_programming': ["planning", "programming_solver", "", "direct_answering"],
        'planning': ["planning", "CoT", "", "direct_answering"],
        "query_rewriting_verifier": ["query_rewriting", "CoT", "verifier", "direct_answering"],
        "planning_verifier": ["planning", "CoT", "verifier", "direct_answering"],
        "programming_solver": ["", "programming_solver", "", "direct_answering"],
        "query_rewriting_programming_verifier": ["query_rewriting", "programming_solver", "verifier",
                                                 "direct_answering"],
        "planning_programming_verifier": ["planning", "programming_solver", "verifier", "direct_answering"],
    }


    def fn(d):
        id, query, correct_answer = id_func(d)
        if DEBUG:
            info = reasoning_workflow.explore(meta_info=d, query=query, correct_answer=correct_answer,
                                              answer_extract_func=answer_extract_func,
                                              checking_func=checking_func, examine_times=examine_time,
                                              traj_dict=traj_dict)
        else:
            try:
                info = reasoning_workflow.explore(meta_info=d, query=query, correct_answer=correct_answer,
                                                  answer_extract_func=answer_extract_func,
                                                  checking_func=checking_func, examine_times=examine_time,
                                                  traj_dict=traj_dict)
            except:
                return
        global res_data
        global save_step
        res_data.append(info)
        if len(res_data) > save_step:
            save_step += args.save_step
            save_json(
                f"{args.save_folder}/{dataset}_trajectory_{debug}{formatted_time}.json",
                data=res_data)

    common.map_with_progress(f=fn, xs=data, num_threads=100)
    global res_data
    save_json(f"{args.save_folder}/{dataset}_trajectory_{debug}{formatted_time}.json",
              data=res_data)


def main(dataset):
    layers = [
        ["query_rewriting", "planning", ""],
        ["CoT", "programming_solver"],
        ["verifier", ""],
        ["direct_answering"]
    ]
    llm = args.api_model
    DEBUG = args.debug

    if dataset == "MATH":
        json_file = args.file
        eval_dataset(dataset=dataset, json_file=json_file, llm=llm, layers=layers,
                     answer_extract_func=math_answer_extract, checking_func=math_answer_checking,
                     id_func=math_id,
                     DEBUG=DEBUG)
    elif dataset == "bbh":
        json_file = args.file
        eval_dataset(dataset=dataset, json_file=json_file, llm=llm, layers=layers,
                     answer_extract_func=bbh_answer_extract, checking_func=bbh_answer_checking,
                     id_func=bbh_id,
                     DEBUG=DEBUG)
    elif dataset == "Game24":
        game24data = read_GameOf24_dataset()
        data = []
        for key in game24data:
            for inst in game24data[key]['instances']:
                data.append(inst)
        random.Random(42).shuffle(data)
        eval_dataset(dataset=dataset, json_file="", data=data, llm=llm, layers=layers,
                     answer_extract_func=Game24_answer_extract, checking_func=Game24_answer_checking,
                     id_func=Game24_id,
                     DEBUG=DEBUG)
    elif dataset == "theoremQA":
        json_file = args.file
        eval_dataset(dataset=dataset, json_file=json_file, llm=llm, layers=layers,
                     answer_extract_func=extract_theoremqa_answer, checking_func=math_answer_checking,
                     id_func=theoremQA_id,
                     DEBUG=DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model", type=str, default="gpt-4o-mini-2024-07-18_azure")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--file", type=str, default="data/math_train.json")
    parser.add_argument("--save_folder", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument('--debug', type=lambda x: x.lower() == 'true', default="True")
    args = parser.parse_args()
    res_data = []
    save_step = args.save_step
    for dataset in args.dataset.split(","):
        main(dataset)