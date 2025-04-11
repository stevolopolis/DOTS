from utils import *
from common import map_with_progress
from src.model_api import select_llm_model
import argparse
from datetime import datetime
from src.reasoning_modules_general import ReasoningModulesGeneral
import random
# from training_data_generation_v3 import input_instruction
# from MATH_performance_zero_shot_5_for_instruction_tuning import python_code_execute

now = datetime.now()
formatted_time = now.strftime("%m%d%H%M%S")

def fn(d):
    question = d['query']
    prompt = question
    response = llm.invoke(prompt=prompt, temperature=0)
    dialogue = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    if "```python" in response or "```\npython" in response:
        response += '```'
        exec_result = python_code_execute(response)
        if "An error occurred:" in exec_result or len(exec_result) == 0:
            continual_instruction = "Please refine the code. Use print() to print out the result. Start with ```python\n"
            dialogue.append({"role": "user", "content": "\nAfter execution, we get:\n" + exec_result + f"\n{continual_instruction}"})
            new_response = llm.invoke(prompt='', final_messages=dialogue, temperature=0)
            dialogue.append({"role": "assistant", "content": new_response})
        new_prompt = "After execution, we get:\n" + exec_result + f"In this step, you need to give the final answer to this question based on the previous thoughts. Follow the mentioned format in the query."
        dialogue.append({"role": "user", "content": new_prompt})
        new_response = llm.invoke(prompt='', final_messages=dialogue, temperature=0)
        dialogue.append({"role": "assistant", "content": new_response})
    if 'veri' in response.split("Required Action: ")[-1].split('\n')[0]:
        new_prompt = "In this step, you need to carefully verify the correctness of the previous thoughts with natural language. You need to formulate a new verification question, not the same question as before, based on the assumption that the final answer is correct. Then try to find if there is any contradiction. If the results are incorrect, the last line should end up with \"The answer is: incorrect\". Otherwise, the last line should end with \"The answer is: correct\""
        dialogue.append({"role": "user", "content": new_prompt})
        new_response = llm.invoke(prompt='', final_messages=dialogue, temperature=0)
        dialogue.append({"role": "user", "content": new_response})
        new_prompt = "In this step, you need to give the final answer to this question based on the previous thoughts. Follow the mentioned format in the query"
        dialogue.append({"role": "user", "content": new_prompt})
        new_response = llm.invoke(prompt='', final_messages=dialogue, temperature=0)
        dialogue.append({"role": "assistant", "content": new_response})
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
    parser.add_argument("--input_file", type=str,
                        default="")
    parser.add_argument("--output_file", type=str,
                        default="")
    parser.add_argument("--debug", type=str, default="false")
    args = parser.parse_args()
    data = read_json(args.input_file)

    if args.debug == "true":
        debug = "debug_"
        random.shuffle(data)
        data = data[:50]
    else:
        debug = ""
    new_data = []
    llm = select_llm_model(llm_model=args.api_model, host=args.host)
    save_step = 0
    map_with_progress(f=fn, xs=data, num_threads=100)
    save_json(
        args.output_file,
        new_data)
    correct_num = 0