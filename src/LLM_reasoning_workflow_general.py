# import math
# from typing import List, Dict, Optional
# import random
# import common
# from src.reasoning_modules_general import ReasoningModulesGeneral
# from tqdm import tqdm
# import copy
#
#
# class LLMReasoningWorkflowGeneral:
#     def __init__(self, llm_name: str, host: str,
#                  layers: List[list] = None,
#                  temperature: Optional[float] = 0.8,
#                  ):
#         """
#         :param llm_name: LLM name for explore
#         :param layers: the layers to select
#         :param temperature: temperature
#         """
#         self.llm_name = llm_name
#         if layers is None:
#             self.layers = {[""], ["CoT"], [""], ["direct_answering"]}
#         else:
#             self.layers = layers
#         self.reasoning_module = ReasoningModulesGeneral(self.llm_name, host=host)
#         self.temperature = temperature
#         self.total_explored_time = 0
#
#     def explore(self, meta_info, query, correct_answer, answer_extract_func, checking_func, examine_times,
#                 traj_dict):
#         score_dict0 = {}
#         for key in traj_dict:
#             path_length = 0
#             for action in traj_dict[key]:
#                 if action != "":
#                     path_length += 1
#             score_dict0[key] = [0, path_length, traj_dict[key]]
#
#         # Probing for filtering easy questions
#         probing_examine_times = 5
#         probing_res_dict = self.probing_step(meta_info, query, correct_answer, answer_extract_func, checking_func,
#                                              probing_examine_times)
#         if probing_res_dict['meta_info']['programming_solver_accuracy'] < 1 or probing_res_dict['meta_info'][
#             'CoT_accuracy'] < 1:
#             return {"meta_info": meta_info, "probing_result": probing_res_dict["dialogue_history"], "step 1 result": [],
#                     "step 2 result": [],
#                     "step 3 result": [], 'step1 score': {}, "step2 score": {},
#                     'step3 score': {}}
#
#         # exploration 1
#         step_1_res_lst, score_dict1 = self.explore_step(query, correct_answer, answer_extract_func, checking_func,
#                                                         examine_times, score_dict0, step_id="1")
#         # filtering 1
#         try:
#             score_dict1 = dict(sorted(score_dict1.items(), key=lambda x: (-x[1][0], x[1][1]))[:8])
#         except:
#             return {"meta_info": meta_info, "probing_result": probing_res_dict["dialogue_history"],
#                     "step 1 result": step_1_res_lst, "step 2 result": [],
#                     "step 3 result": [], 'step1 score': score_dict1, "step2 score": {},
#                     'step3 score': {}}
#         # filtering questions too easy or too hard
#         too_easy_cases_num = 0
#         too_hard_cases_num = 0
#         for key in score_dict1:
#             if score_dict1[key][0] >= examine_times - 1:
#                 too_easy_cases_num += 1
#             if score_dict1[key][1] <= 1:
#                 too_hard_cases_num += 1
#         if too_easy_cases_num >= len(score_dict1) - 2 or too_hard_cases_num >= len(score_dict1) - 2:
#             return {"meta_info": meta_info, "probing_result": probing_res_dict["dialogue_history"],
#                     "step 1 result": step_1_res_lst, "step 2 result": [],
#                     "step 3 result": [], 'step1 score': score_dict1, "step2 score": {},
#                     'step3 score': {}}
#
#         # exploration 2
#         step_2_res_lst, score_dict2 = self.explore_step(query, correct_answer, answer_extract_func, checking_func,
#                                                         examine_times, score_dict1, step_id="2")
#         # filtering 2
#         try:
#             score_dict2 = dict(sorted(score_dict2.items(), key=lambda x: (-x[1][0], x[1][1]))[:3])
#         except:
#             return {"meta_info": meta_info, "probing_result": probing_res_dict["dialogue_history"],
#                     "step 1 result": step_1_res_lst, "step 2 result": step_2_res_lst,
#                     "step 3 result": [], 'step1 score': score_dict1, "step2 score": score_dict2,
#                     'step3 score': {}}
#
#         # exploration 3
#         step_3_res_lst, score_dict3 = self.explore_step(query, correct_answer, answer_extract_func, checking_func,
#                                                         examine_times, score_dict2, step_id="3")
#         return {"meta_info": meta_info, "probing_result": probing_res_dict["dialogue_history"],
#                 "step 1 result": step_1_res_lst, "step 2 result": step_2_res_lst,
#                 "step 3 result": step_3_res_lst, 'step1 score': score_dict1, "step2 score": score_dict2,
#                 'step3 score': score_dict3}
#
#     def probing_step(self, meta_info, query, correct_answer, answer_extract_func, checking_func, examine_times):
#         traj_dict = {
#             "CoT": ["", "CoT", "", "direct_answering"],
#             "programming_solver": ["", "programming_solver", "", "direct_answering"],
#         }
#         new_meta_info = meta_info.copy()
#         res_lst = []
#         for key in traj_dict.keys():
#             modules = traj_dict[key]
#             _, results = self.reasoning_module.think_reply(query=query, modules=modules,
#                                                            n=examine_times,
#                                                            temperature=self.temperature)
#             c = 0
#             for i in range(examine_times):
#                 x = results[i]
#                 pred_ans = answer_extract_func(x)
#                 score = checking_func(correct_answer, pred_ans)
#                 if score > 0:
#                     c += 1
#                 res_lst.append(
#                     {"id": f"{key}_{i}", "trajectory": modules, "reasoning dialogues": x,
#                      "pred_ans": pred_ans,
#                      "score": score})
#             accuracy = c / examine_times
#             new_meta_info[f'{key}_accuracy'] = accuracy
#         return {"meta_info": new_meta_info, "dialogue_history": res_lst}
#
#     def explore_step(self, query, correct_answer, answer_extract_func, checking_func, examine_times,
#                      score_dict, step_id):
#         res_lst = []
#         for key in score_dict.keys():
#             modules = score_dict[key][-1]
#             _, results = self.reasoning_module.think_reply(query=query, modules=modules,
#                                                            n=examine_times,
#                                                            temperature=self.temperature)
#             c = 0
#             for i in range(examine_times):
#                 x = results[i]
#                 pred_ans = answer_extract_func(x)
#                 score = checking_func(correct_answer, pred_ans)
#                 if score > 0:
#                     c += 1
#                 res_lst.append({"id": f"{key}_step{step_id}_{i}", "trajectory": modules, "reasoning dialogues": x,
#                                 "pred_ans": pred_ans, "score": score})
#             score_dict[key][0] += c
#         return res_lst, copy.deepcopy(score_dict)

import math
from typing import List, Dict, Optional
from src.reasoning_modules_general import ReasoningModulesGeneral
import copy


class LLMReasoningWorkflowGeneral:
    def __init__(self, llm_name: str, host: str, layers: List[list] = None, temperature: Optional[float] = 0.8):
        """
        :param llm_name: LLM name for explore
        :param layers: the layers to select
        :param temperature: temperature
        """
        self.llm_name = llm_name
        self.layers = layers if layers is not None else [[""], ["CoT"], [""], ["direct_answering"]]
        self.reasoning_module = ReasoningModulesGeneral(self.llm_name, host=host)
        self.temperature = temperature
        self.total_explored_time = 0

    def explore(self, meta_info, query, correct_answer, answer_extract_func, checking_func, examine_times, traj_dict):
        score_dict = self.initialize_score_dict(traj_dict)

        probing_res_dict = self.probing_step(meta_info, query, correct_answer, answer_extract_func, checking_func, examine_times)
        if not self.is_probing_successful(probing_res_dict):
            return self.create_exploration_result(meta_info, probing_res_dict["dialogue_history"], [], [], [])

        # Step 1 exploration and filtering
        step_1_res, score_dict1 = self.explore_step(query, correct_answer, answer_extract_func, checking_func,
                                                    examine_times, score_dict, step_id="1")
        score_dict1 = self.filter_explored_cases(score_dict1, 8)

        if self.is_case_too_easy_or_hard(score_dict1, examine_times):
            return self.create_exploration_result(meta_info, probing_res_dict["dialogue_history"], step_1_res, [], [])

        # Step 2 exploration and filtering
        step_2_res, score_dict2 = self.explore_step(query, correct_answer, answer_extract_func, checking_func,
                                                    examine_times, score_dict1, step_id="2")
        score_dict2 = self.filter_explored_cases(score_dict2, 3)

        # Step 3 exploration
        step_3_res, score_dict3 = self.explore_step(query, correct_answer, answer_extract_func, checking_func,
                                                    examine_times, score_dict2, step_id="3")

        return self.create_exploration_result(meta_info, probing_res_dict["dialogue_history"], step_1_res, step_2_res,
                                              step_3_res, score_dict1, score_dict2, score_dict3)

    def probing_step(self, meta_info, query, correct_answer, answer_extract_func, checking_func, examine_times):
        traj_dict = {
            "CoT": ["", "CoT", "", "direct_answering"],
            "programming_solver": ["", "programming_solver", "", "direct_answering"],
        }
        new_meta_info = meta_info.copy()
        res_lst = []

        for key, modules in traj_dict.items():
            c, results = self.run_modules(query, modules, examine_times)
            res_lst.extend(
                self.generate_result_list(results, key, modules, correct_answer, answer_extract_func, checking_func,
                                          examine_times))
            accuracy = c / examine_times
            new_meta_info[f'{key}_accuracy'] = accuracy

        return {"meta_info": new_meta_info, "dialogue_history": res_lst}

    def explore_step(self, query, correct_answer, answer_extract_func, checking_func, examine_times, score_dict,
                     step_id):
        res_lst = []
        for key, (score, path_length, modules) in score_dict.items():
            c, results = self.run_modules(query, modules, examine_times)
            res_lst.extend(
                self.generate_result_list(results, key, modules, correct_answer, answer_extract_func, checking_func,
                                          examine_times, step_id))
            score_dict[key][0] += c
        return res_lst, copy.deepcopy(score_dict)

    # Helper Functions
    def initialize_score_dict(self, traj_dict):
        return {key: [0, len([a for a in actions if a != ""]), actions] for key, actions in traj_dict.items()}

    def is_probing_successful(self, probing_res_dict):
        return probing_res_dict['meta_info']['programming_solver_accuracy'] == 1 and probing_res_dict['meta_info'][
            'CoT_accuracy'] == 1

    def filter_explored_cases(self, score_dict, max_cases):
        try:
            return dict(sorted(score_dict.items(), key=lambda x: (-x[1][0], x[1][1]))[:max_cases])
        except Exception:
            return score_dict

    def is_case_too_easy_or_hard(self, score_dict, examine_times):
        too_easy_cases = sum(1 for score, path_length, _ in score_dict.values() if score >= examine_times - 1)
        too_hard_cases = sum(1 for score, path_length, _ in score_dict.values() if path_length <= 1)
        return too_easy_cases >= len(score_dict) - 2 or too_hard_cases >= len(score_dict) - 2

    def create_exploration_result(self, meta_info, probing_res, step_1_res, step_2_res, step_3_res, score_dict1=None,
                                  score_dict2=None, score_dict3=None):
        return {
            "meta_info": meta_info,
            "searching_results": probing_res + step_1_res + step_2_res + step_3_res,
            "probing_result": probing_res,
            "step 1 result": step_1_res,
            "step 2 result": step_2_res,
            "step 3 result": step_3_res,
            'step1 score': score_dict1 if score_dict1 else {},
            "step2 score": score_dict2 if score_dict2 else {},
            'step3 score': score_dict3 if score_dict3 else {},
        }

    def run_modules(self, query, modules, examine_times):
        _, results = self.reasoning_module.think_reply(query=query, modules=modules, n=examine_times,
                                                       temperature=self.temperature)
        correct_count = sum(1 for x in results if x)
        return correct_count, results

    def generate_result_list(self, results, key, modules, correct_answer, answer_extract_func, checking_func,
                             examine_times, step_id=None):
        res_lst = []
        for i in range(examine_times):
            result = results[i]
            pred_ans = answer_extract_func(result)
            score = checking_func(correct_answer, pred_ans)
            res_lst.append({
                "id": f"{key}_{'step' + step_id + '_' if step_id else ''}{i}",
                "trajectory": modules,
                "reasoning dialogues": result,
                "pred_ans": pred_ans,
                "score": score
            })
        return res_lst
