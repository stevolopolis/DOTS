from utils import *
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/math_explanation.json")
    parser.add_argument("--output_file", type=str, default="data/sft_training_new.json")
    parser.add_argument("--dataset", type=str, default="MATH")
    args = parser.parse_args()
    data = read_json(args.input_file)
    input_output_data = []
    repeat_time = 8
    for d in data:
        action = d['action']
        if args.dataset=='MATH':
            question = MATH_QUESTION_TEMPLATE.format(Question=d['query'])
        else:
            question = d['query']
        explain = "First, let me analyze the question and determine the reasoning actions I will follow.\n" + d[
            'explanation'] + f"\nRequired Action: {', '.join(action)}\nNow, I will begin solving the question with the required actions:\n"
        d['solution'][1]['content'] = explain + d['solution'][1]['content']
        input_output_data.append(
            {"dataset": args.dataset, "instruction": question, "input": "",
             "output": d['solution'][1]['content']})
        if  args.dataset != 'MATH':
            for i in range(repeat_time):
                input_output_data.append(
                    {"dataset": args.dataset, "instruction": question, "input": "",
                     "output": d['solution'][1]['content']})
    random.shuffle(input_output_data)
    save_json(filename=args.output_file, data=input_output_data)
