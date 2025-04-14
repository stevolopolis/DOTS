from dotenv import load_dotenv
import os
from openai import OpenAI,AzureOpenAI

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
azure_api_key = os.getenv('AZURE_OPENAI_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_version = os.getenv('AZURE_OPENAI_VERSION')

try:
    import openai
except ImportError:
    print("OpenAI API not installed.")
    pass
try:
    import anthropic
except ImportError:
    print("")
    pass
import time
import random
import requests
import json
import sys


def select_llm_model(llm_model, host):
    if "llama-3" in llm_model.lower() or "llama3" in llm_model.lower():
        llm_chain = ChatVLLM(model_name=llm_model, host=host)
    elif "claude" in llm_model:
        llm_chain = ClaudeAthropic(model_name=llm_model)
    elif "gpt" in llm_model:
        llm_chain = ChatGPTOpenAI(model_name=llm_model, api_key_type="openai")
    return llm_chain


class ChatGPTOpenAI:
    def __init__(self, model_name: str, api_key_type: str):
        self.model_name = model_name
        if "gpt-3.5" in model_name or "gpt-35" in model_name:
            self.system_prompt = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture."
        elif "gpt-4" in model_name:
            self.system_prompt = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."

        self.cot_prompt = "You need to think step by step with words, solve the problem and get the answer."
        if "cot" in model_name:
            self.system_prompt += f"\n{self.cot_prompt}"
            model_name = model_name.replace("-cot", "")

        if "_azure" in model_name:
            # openai.api_key = azure_api_key
            # openai.api_base = azure_endpoint
            self.client = AzureOpenAI(api_key=azure_api_key, azure_endpoint=azure_endpoint, api_version=azure_version)
        else:
            openai.api_key = openai_api_key
            self.client = OpenAI()

        if "_azure" in model_name:
            self.api_model = model_name.replace("_azure", "")
        else:
            self.api_model = model_name

    def invoke(self, prompt: str, stop: str = '```\n', temperature: float = 0.0, frequency_penalty: float = 0.0,
               final_messages=None,
               n: int = 1) -> str:
        if final_messages is None:
            final_messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        else:
            final_messages = [{"role": "system", "content": self.system_prompt}] + final_messages
        if "_azure" in self.model_name:
            RateLimitError_max = 50
            APIError_max = 5
            InvalidRequestError_max = 5
            sleep_time = 2
        else:
            RateLimitError_max = 50
            APIError_max = 5
            InvalidRequestError_max = 3
            sleep_time = 5

        RateLimitError_count = 0
        APIError_count = 0
        InvalidRequestError_count = 0
        while True:
            try:
                openai_response = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=final_messages,
                    max_tokens=2048,
                    temperature=temperature,
                    frequency_penalty=frequency_penalty,
                    n=n,
                )

            except Exception as e:
                print('Error occurred' + ', retrying. Error type: ', type(e).__name__)
                if RateLimitError_count >= RateLimitError_max:
                    print(f'RateLimitError_count exceeded {RateLimitError_max}, exiting...')
                    response = None
                    break
                elif APIError_count >= APIError_max:
                    print(f'APIError_count exceeded {APIError_max}, exiting...')
                    response = None
                    break
                elif InvalidRequestError_count >= InvalidRequestError_max:
                    print(f'InvalidRequestError_count exceeded {InvalidRequestError_max}, exiting...')
                    response = None
                    break
                elif type(e).__name__ == 'RateLimitError':
                    time.sleep(random.uniform(sleep_time - 0.2, sleep_time + 0.2))
                    RateLimitError_count += 1
                elif type(e).__name__ == 'APIError':
                    time.sleep(random.uniform(sleep_time - 0.2, sleep_time + 0.2))
                    APIError_count += 1
                elif type(e).__name__ in ['InvalidRequestError', 'BadRequestError', 'AttributeError', 'IndexError']:
                    time.sleep(random.uniform(sleep_time - 0.2, sleep_time + 0.2))
                    InvalidRequestError_count += 1
            else:
                response = openai_response
                break

        if response is None or len(response.choices) == 0:
            resp = ""
            finish_reason = "Error during OpenAI inference"
            return resp
        if n == 1:
            resp = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
        else:
            resp = [response.choices[i].message.content for i in range(len(response.choices))]
            finish_reason = response.choices[0].finish_reason
        return resp

    def assistant_invoke(self, prompt: str, temperature: float = 0.0) -> str:
        if "_azure" in self.model_name:
            RateLimitError_max = 300
            APIError_max = 300
            InvalidRequestError_max = 5
            sleep_time = 2
        else:
            RateLimitError_max = 50
            APIError_max = 5
            InvalidRequestError_max = 3
            sleep_time = 5

        RateLimitError_count = 0
        APIError_count = 0
        InvalidRequestError_count = 0

        while True:
            try:
                assistant = self.client.beta.assistants.create(model=self.api_model, temperature=temperature,
                                                               tools=[{"type": "code_interpreter"},
                                                                      {"type": "file_search"}])
                thread = self.client.beta.threads.create()
                self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=prompt
                )
                run = self.client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                )
                if run.status == 'completed':
                    messages = self.client.beta.threads.messages.list(
                        thread_id=thread.id
                    )
                    resp = '\n'.join(
                        [messages.data[k].content[0].text.value for k in range(len(messages.data) - 2, -1, -1)])
                else:
                    resp = ""
            except Exception as e:
                # print('Error occurred' + ', retrying. Error type: ', type(e).__name__)
                if RateLimitError_count >= RateLimitError_max:
                    print(f'RateLimitError_count exceeded {RateLimitError_max}, exiting...')
                    response = None
                    break
                elif APIError_count >= APIError_max:
                    print(f'APIError_count exceeded {APIError_max}, exiting...')
                    response = None
                    break
                elif InvalidRequestError_count >= InvalidRequestError_max:
                    print(f'InvalidRequestError_count exceeded {InvalidRequestError_max}, exiting...')
                    response = None
                    break
                elif type(e).__name__ == 'RateLimitError':
                    time.sleep(random.uniform(sleep_time - 0.2, sleep_time + 0.2))
                    RateLimitError_count += 1
                elif type(e).__name__ == 'APIError':
                    time.sleep(random.uniform(sleep_time - 0.2, sleep_time + 0.2))
                    APIError_count += 1
                elif type(e).__name__ in ['InvalidRequestError', 'BadRequestError', 'AttributeError', 'IndexError']:
                    time.sleep(random.uniform(sleep_time - 0.2, sleep_time + 0.2))
                    InvalidRequestError_count += 1
            else:
                response = resp
                break

        return response


class ClaudeAthropic:
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic(api_key="xxxxxxxxxxx")
        # "claude-3-opus-20240229",
        # "claude-3-sonnet-20240229",
        # "claude-3-haiku-20240307",
        # "claude-2.1",
        # "claude-2.0",
        # "claude-instant-1.2",
        self.system_prompt = "The assistant is Claude, created by Anthropic."
        self.api_model = model_name

    def invoke(self, prompt: str, temperature: float = 0.2, frequency_penalty: float = 0.0) -> str:

        final_messages = [
            {"role": "user", "content": prompt}
        ]

        RateLimitError_max = 20
        APIError_max = 5
        sleep_time = 5

        RateLimitError_count = 0
        APIError_count = 0
        while True:
            try:
                claude_response = self.client.messages.create(
                    model=self.api_model,
                    system=self.system_prompt,
                    max_tokens=4096,
                    temperature=temperature,
                    messages=final_messages
                )
            except Exception as e:
                """
                # https://github.com/anthropics/anthropic-sdk-python
                * 400	BadRequestError
                401	AuthenticationError
                403	PermissionDeniedError
                404	NotFoundError
                * 422	UnprocessableEntityError
                * 429	RateLimitError
                * >=500	InternalServerError
                * N/A	APIConnectionError
                """
                print('Error occurred' + ', retrying. Error type: ', type(e).__name__)
                if RateLimitError_count >= RateLimitError_max:
                    print(f'RateLimitError_count exceeded {RateLimitError_max}, exiting...')
                    response = None
                    break
                elif APIError_count >= APIError_max:
                    print(f'APIError_count exceeded {APIError_max}, exiting...')
                    response = None
                    break
                elif type(e).__name__ in ['RateLimitError', 'InternalServerError']:
                    time.sleep(sleep_time)
                    RateLimitError_count += 1
                elif type(e).__name__ == 'APIConnectionError':
                    time.sleep(sleep_time)
                    APIError_count += 1
                elif type(e).__name__ in ['BadRequestError', 'UnprocessableEntityError']:
                    response = None
                    break
                else:
                    response = None
                    break
            else:
                response = claude_response
                break

        if response is None or len(response.content) == 0:
            resp = ""
            finish_reason = "Error during Claude inference"
            return resp
        resp = response.content[0].text
        finish_reason = response.stop_reason
        return resp


class ChatVLLM:
    def __init__(self, model_name: str, host):
        if ":" in model_name:
            self.host = host
            self.port = model_name.split(":")[-1]
            self.model_name = model_name.split(":")[0]
        else:
            self.host = 'localhost'
            self.port = 4231
            self.model_name = model_name

        self.cot_prompt = "You need to think step by step with words, solve the problem and get the answer."

        if "llama-3" in self.model_name.lower():
            template = [
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                f"You are a helpful assistant.<|eot_id|>\n",  # The system prompt is optional
                # f"You are a helpful assistant. {self.cot_prompt}<|eot_id|>\n",  # CoT prompt
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "{prompt}<|eot_id|>\n",
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            ]

            if "cotsys" in model_name:
                template[1] = f"You are a helpful assistant. {self.cot_prompt}<|eot_id|>\n"
                self.model_name = self.model_name.replace("-cotsys", "")
            elif "cotasst" in model_name:
                template.append("Let's think step by step.\n")
                self.model_name = self.model_name.replace("-cotasst", "")

            self.template = "".join(template)


    def message2prompt(self, message):
        message_lst = [
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            "You are a helpful assistant.<|eot_id|>\n",  # The system prompt is optional
        ]
        for m in message:
            if m['role'] == 'user':
                message_lst.append("<|start_header_id|>user<|end_header_id|>\n\n")
                message_lst.append(f"{m['content']}<|eot_id|>\n")
            elif m['role'] == 'assistant':
                message_lst.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                message_lst.append(f"{m['content']}<|eot_id|>\n")
        message_lst.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(message_lst)

    def invoke(self, prompt: str, temperature: float = 0.4, frequency_penalty: float = 0.0, n: int = 1, stop=None,
               final_messages=None):
        if stop is None:
            stop = []
        if final_messages is None:
            new_prompt = self.template.format(prompt=prompt)
        else:
            new_prompt = self.message2prompt(message=final_messages)
        json_data = {
            "n": n,
            "model": self.model_name,
            "prompt": new_prompt,
            "max_tokens": 2048,
            "top_p": 1,
            "stop": ["<|end_of_text|>", "<|eot_id|>"] + stop,
            "temperature": temperature,
        }
        if n == 1:
            try:
                response = requests.post(f'http://{self.host}:{self.port}/v1/completions', json=json_data)
                response_content_string = response.text
                completion = json.loads(response_content_string)['choices'][0]['text']
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting...")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}")
                completion = "ERROR"
            return completion
        else:
            try:
                response = requests.post(f'http://{self.host}:{self.port}/v1/completions', json=json_data)
                response_content_string = response.text
                json_content = json.loads(response_content_string)
                # resp = [response.choices[i].message.content for i in range(len(response.choices))]
                # completion = json.loads(response_content_string)['choices'][0]['text']
                completion_lst = [json_content['choices'][i]['text'] for i in range(len(json_content['choices']))]
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting...")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}")
                completion = "ERROR"
                completion_lst = [completion]
            return completion_lst