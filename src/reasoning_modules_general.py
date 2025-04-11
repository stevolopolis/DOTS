from src.model_api import select_llm_model
from src.prompt import *
from src.safe_execute import safe_exec
from common import map_with_progress, map_without_progress


class ReasoningModulesGeneral:
    def __init__(self, llm_name, host):
        self.llm_name = llm_name
        self.llm = select_llm_model(llm_name, host)
        self.REASONING_INSTRUCTION = {}
        for key in ATOMIC_COMPONENTS_DESCRIPTION:
            self.REASONING_INSTRUCTION[key] = ATOMIC_COMPONENTS_DESCRIPTION[key]['instruction']
        if "direct_answering" not in self.REASONING_INSTRUCTION:
            self.REASONING_INSTRUCTION['direct_answering'] = ATOMIC_COMPONENTS_DESCRIPTION['direct_answering'][
                "instruction"]

    def _prompt_create(self, query, reasoning_modules):
        instruction = ""
        index = 0
        for module in reasoning_modules:
            if module == "" or module.lower() == "empty" or module == "verifier":
                continue
            instruction += f"Step {index}: {self.REASONING_INSTRUCTION[module]}\n"
            index += 1
        prompt = f"""{{
{query}
}}
Instruction:{{
{instruction}
}}
For the i-th step, the output should start with "#### Step i ####".
"""
        return prompt

    def _python_code_execute(self, code):
        import re
        def extract_python_code(text):
            if "```python" in code:
                pattern = r'```python(.*?)```'
            else:
                pattern = r'python(.*?)```'
            matches = re.findall(pattern, text, re.DOTALL)
            return matches

        if "```python" in code or "```\npython" in code:
            # code += '\n```\n'
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

    def _post_processing(self, text):
        text = text.replace("**", "")
        return text

    def dialogue_update_with_code_exec(self, dialogue_history, continual_instruction):
        exec_result = self._python_code_execute(dialogue_history[-1]["content"])
        exec_final_result = "\nAfter execution, we get:\n" + exec_result + f"\n{continual_instruction}"
        dialogue_history += [{"role": "user", "content": exec_final_result}]
        response = self.llm.invoke(prompt='', final_messages=dialogue_history)
        dialogue_history += [{"role": "assistant", "content": response}]
        return response, dialogue_history

    def think_reply(self, query: str, modules: list, id='', n=1, temperature=1.0):
        """
        :param id:
        :param temperature:
        :param query:
        :param layer_input:
        :param module_name:
        :param n: The number for self-consistency
        :return:
        """
        prompt = self._prompt_create(query, modules)
        messages = [{"role": "user", "content": prompt}]
        # logging.info(f"{id}: Prompt: {prompt}")
        if n == 1:
            dialogue_history = messages
            response = self.llm.invoke(prompt='', final_messages=dialogue_history,
                                       temperature=temperature)
            dialogue_history += [{"role": "assistant", "content": response}]
            if 'programming_solver' in modules:
                if "verifier" in modules:
                    continual_instruction = INIT_VERIFIER_INSTRUCTION
                    response, cur_dialogue_history = self.dialogue_update_with_code_exec(dialogue_history,
                                                                                         continual_instruction=continual_instruction)
                    if "incorrect" not in response.split('\n')[-1]:
                        dialogue_history += [{"role": "user", "content": INIT_DIRECT_ANSWERING_INSTRUCTION}]
                    else:
                        dialogue_history += [{"role": "user", "content": INIT_RE_ANSWERING_INSTRUCTION}]
                    response = self.llm.invoke(prompt='', final_messages=dialogue_history,
                                               temperature=temperature)
                    dialogue_history += [{"role": "assistant", "content": response}]
                else:
                    continual_instruction = INIT_DIRECT_ANSWERING_INSTRUCTION
                    _, dialogue_history = self.dialogue_update_with_code_exec(dialogue_history,
                                                                              continual_instruction=continual_instruction)
            elif "verifier" in modules:
                dialogue_history += [{"role": "user", "content": INIT_VERIFIER_INSTRUCTION}]
                response = self.llm.invoke(prompt='', final_messages=dialogue_history,
                                           temperature=temperature)
                dialogue_history += [{"role": "assistant", "content": response}]
                if "incorrect" not in response.split('\n')[-1]:
                    dialogue_history += [{"role": "user", "content": INIT_DIRECT_ANSWERING_INSTRUCTION}]
                else:
                    dialogue_history += [{"role": "user",
                                          "content": INIT_RE_ANSWERING_INSTRUCTION}]
                response = self.llm.invoke(prompt='', final_messages=dialogue_history,
                                           temperature=temperature)
                dialogue_history += [{"role": "assistant", "content": response}]
            return prompt, dialogue_history
        else:
            dialogue_history = messages
            dialogue_lst = []
            response_lst = self.llm.invoke(prompt, n=n, temperature=temperature)
            if 'programming_solver' in modules or "verifier" in modules:
                paras = []
                for response in response_lst:
                    paras.append({'response': response, 'dialogue_history': dialogue_history, 'modules': modules,
                                  'temperature': temperature})
                dialogue_lst = map_without_progress(f=self.fn, xs=paras, num_threads=10)
            else:
                for response in response_lst:
                    cur_dialogue_history = dialogue_history + [{"role": "assistant", "content": response}]
                    dialogue_lst.append(cur_dialogue_history)
            return prompt, dialogue_lst

    def fn(self, paras):
        dialogue_history = paras['dialogue_history']
        modules = paras['modules']
        temperature = paras['temperature']
        response = paras['response']
        cur_dialogue_history = dialogue_history + [{"role": "assistant", "content": response}]
        if 'programming_solver' in modules:
            if "verifier" in modules:
                continual_instruction = INIT_VERIFIER_INSTRUCTION
                response, cur_dialogue_history = self.dialogue_update_with_code_exec(cur_dialogue_history,
                                                                                     continual_instruction=continual_instruction)
                if "incorrect" not in response.split('\n')[-1]:
                    cur_dialogue_history += [{"role": "user", "content": INIT_DIRECT_ANSWERING_INSTRUCTION}]
                else:
                    cur_dialogue_history += [{"role": "user", "content": INIT_RE_ANSWERING_INSTRUCTION}]
                response = self.llm.invoke(prompt='', final_messages=cur_dialogue_history,
                                           temperature=temperature)
                cur_dialogue_history += [{"role": "assistant", "content": response}]
            else:
                continual_instruction = INIT_DIRECT_ANSWERING_INSTRUCTION
                _, cur_dialogue_history = self.dialogue_update_with_code_exec(cur_dialogue_history,
                                                                              continual_instruction=continual_instruction)
        elif "verifier" in modules:
            cur_dialogue_history += [{"role": "user", "content": INIT_VERIFIER_INSTRUCTION}]
            response = self.llm.invoke(prompt='', final_messages=cur_dialogue_history,
                                       temperature=temperature)
            cur_dialogue_history += [{"role": "assistant", "content": response}]
            if "incorrect" not in response.split('\n')[-1]:
                cur_dialogue_history += [{"role": "user", "content": INIT_DIRECT_ANSWERING_INSTRUCTION}]
            else:
                cur_dialogue_history += [{"role": "user",
                                          "content": INIT_RE_ANSWERING_INSTRUCTION}]
            response = self.llm.invoke(prompt='', final_messages=cur_dialogue_history,
                                       temperature=temperature)
            cur_dialogue_history += [{"role": "assistant", "content": response}]
        return cur_dialogue_history
