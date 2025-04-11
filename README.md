# DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search

This repository provides code implementation for our paper [DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search](https://arxiv.org/abs/2410.03864).


```
@article{yue2024dotslearningreasondynamically,
      title={DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search}, 
      author={Murong Yue and Wenlin Yao and Haitao Mi and Dian Yu and Ziyu Yao and Dong Yu},
      year={2024},
      eprint={2410.03864},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.03864}, 
}
```

## Data
If you only need the searched result and final SFT data, please refer to [Google Drive](https://drive.google.com/drive/folders/1aFIvCa1yxU9DHNbaMeffdbVLGW9ygygV?usp=sharing).

## Setup
```
pip install -r requirements.txt
touch .env
echo "openai_api_key=<YOUR_OPENAI_KEY>" >> .env
```
We host local models using vllm (https://github.com/vllm-project/vllm). 

```sh
python -m vllm.entrypoints.openai.api_server --model $OPEN_SOURCED_LLM --tensor-parallel-size 1 --max-num-batched-tokens 8192 --dtype bfloat16 --port $PORT
```

[//]: # ()
[//]: # (```sh)

[//]: # (python search_path.py --api_model gpt-4o-mini --file dataset/math_train.json --dataset MATH --debug)

[//]: # (```)


## Code implementation
#### 1. Search the Best Trajectory

[//]: # (We host our fine-tuned model models using vllm &#40;https://github.com/vllm-project/vllm&#41;. )

[//]: # (```sh)

[//]: # (python -m vllm.entrypoints.openai.api_server --model $LLM_PATH --tensor-parallel-size 1 --max-num-batched-tokens 8192 --dtype bfloat16 --port $PORT)

[//]: # (```)
For searching the path,
```sh
python search_path.py --api_model $OPEN_SOURCED_LLM:$PORT --host $HOST --file $FIlE_NAME --dataset $DATASET
```
Arguments:
- ``--api_model``: gpt-4o-mini or $OPEN_SOURCED_LLM:$PORT
- ``--host``: Server IP. Host for remote LLM.
- ``--file``: data file to load, e.g., "data/math_train.json"
- ``--dataset``: data file to select, e.g., 'MATH'

<!-- The searched results are in https://huggingface.co/datasets/MurongYue/DOTS_searching_results -->
You can download it from [Hugging Face](https://huggingface.co/datasets/MurongYue/DOTS_searching_results) or [Google Drive](https://drive.google.com/drive/folders/1aFIvCa1yxU9DHNbaMeffdbVLGW9ygygV?usp=sharing).


#### 2. Explaination Generation
If you use the data generated from step 1, run the code:
```sh
python local_searched_data_processing.py --input_file $serached_file --output_file $output_file --api_model $MODEL
```
- ``--input_file``: data file to load, e.g., "./data/searching/llama3_8b_math_result.json"
- ``--output_file``: data file to save, e.g., "./data/training_data/llama3_8b_math_training.json"
- ``--api_model``: gpt-4o, gpt-4o-mini or $OPEN_SOURCED_LLM:$PORT



#### 3. Training
Before starting training, we need to process the data.
```sh
python training_instance_construction.py --input_file $serached_file --output_file $output_file --api_model $MODEL
```

After processing, we get the training data. We use the litgpt (https://github.com/Lightning-AI/litgpt/tree/main) to finetune the LLM. Please follow the instruction for environment setup.

```sh
litgpt finetune_full --config ..
```
The config files are in ./config and training files in training_config.

#### 4. Inference and Evaluation
After training, please also host the fine-tuned model models using vllm.
<!-- For external planner inference, run
```sh
python external_planner_performance.py 
```
- ``--api_model``: For solver LLM. gpt-4o-mini or .../Meta-Llama-3-70B-Instruct/:$PORT
-  ``--host``: Host for remote solver LLM
-  ``--trajectory_api_model``: .../$Planner_LLM/:$PORT
-  ``--trajectory_host``: Host for remote planner LLM
-  ``--debug``: whether to activate the debug mode
Then run
```sh
python datasets_eval.py
``` -->

<!-- For internalized planner inference, run -->
```sh
python internalized_planner_performance.py --api_model $OPEN_SOURCED_LLM:$PORT --host $HOST --test_file $test_file
```
- ``--api_model``: For solver LLM. gpt-4o-mini or .../Meta-Llama-3-70B-Instruct/:$PORT
-  ``--host``: Host for remote solver LLM
