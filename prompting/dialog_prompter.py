import os
import json
import pickle
import numpy as np
from datetime import datetime
from os.path import join
from typing import List, Tuple, Dict, Union, Optional, Any

from rocobench.subtask_plan import LLMPathPlan
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.envs import MujocoSimEnv, EnvState
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from .llm_api import chat_completion

PATH_PLAN_INSTRUCTION="""
[路径规划指令]
每个 <coord> 是一个 (x,y,z) 元组，表示夹爪位置，按以下步骤规划：
1) 确定目标位置（如你要抓取的物体）和当前夹爪位置。
2) 规划一组从当前夹爪平滑移动到目标位置的 <coord> 列表。
3) 各 <coord> 之间的间距应均匀。
4) 每个 <coord> 不能与其他机器人碰撞，且必须远离桌面和物体。
[如何利用 [环境反馈] 改进计划]
    如果 IK 失败，提出更可行的夹爪移动路径。
    如果检测到碰撞，移动机器人使夹爪和手中物体远离碰撞物体。
    如果在目标步骤检测到碰撞，选择不同的动作。
    使路径间距更均匀：让相邻步骤之间的距离相近。
    如果计划执行失败，重新规划更可行的 PATH 步骤，或选择不同的动作。
"""

class DialogPrompter:
    """
    Each round contains multiple prompts, query LLM once per each agent 
    """
    def __init__(
        self,
        env: MujocoSimEnv,
        parser: LLMResponseParser,
        feedback_manager: FeedbackManager, 
        max_tokens: int = 512,
        debug_mode: bool = False,
        use_waypoints: bool = False,
        robot_name_map: Dict[str, str] = {"panda": "Bob"},
        num_replans: int = 3, 
        max_calls_per_round: int = 10,
        use_history: bool = True,  
        use_feedback: bool = True,
        temperature: float = 0,
        llm_source: str = "gpt-4"
    ):
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.num_replans = num_replans
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []
        self.max_calls_per_round = max_calls_per_round 
        self.temperature = temperature
        self.llm_source = llm_source

    def compose_system_prompt(
        self, 
        obs: EnvState, 
        agent_name: str,
        chat_history: List = [], # chat from previous replan rounds
        current_chat: List = [],  # chat from current round, this comes AFTER env feedback 
        feedback_history: List = []
    ) -> str:
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION
        agent_prompt = self.env.get_agent_prompt(obs, agent_name)
        
        round_history = self.get_round_history() if self.use_history else ""

        execute_feedback = ""
        if len(self.failed_plans) > 0:
            execute_feedback = "以下计划执行失败，请改进以避免碰撞并平稳到达目标：\n"
            execute_feedback += "\n".join(self.failed_plans) + "\n"

        chat_history = "[之前的对话]\n" + "\n".join(chat_history) if len(chat_history) > 0 else ""

        system_prompt = f"{action_desp}\n{round_history}\n{execute_feedback}{agent_prompt}\n{chat_history}\n"

        if self.use_feedback and len(feedback_history) > 0:
            system_prompt += "\n".join(feedback_history)

        if len(current_chat) > 0:
            system_prompt += "[当前对话]\n" + "\n".join(current_chat) + "\n"

        return system_prompt 

    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[历史记录]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== 第{i}轮 ==\n{history}\n"
        ret += f"== 当前轮 ==\n"
        return ret
    
    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        plan_feedbacks = []
        chat_history = [] 
        for i in range(self.num_replans):
            final_agent, final_response, agent_responses = self.prompt_one_dialog_round(
                obs,
                chat_history,
                plan_feedbacks,
                replan_idx=i,
                save_path=save_path,
            )
            chat_history += agent_responses
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, final_response) 

            curr_feedback = "None"
            if not parse_succ:
                curr_feedback = f"""
[{final_agent}]的上一个回复解析失败！: '{final_response}'
{parsed_str} 请严格按照[动作输出格式]重新输出！"""
                ready_to_execute = False  
            
            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        break
            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (final_response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w')) 

            if ready_to_execute: 
                break  
            else:
                print(curr_feedback)
        self.latest_chat_history = chat_history
        return ready_to_execute, llm_plans, plan_feedbacks, chat_history
   
    def prompt_one_dialog_round(
        self, 
        obs, 
        chat_history, 
        feedback_history, 
        replan_idx=0,
        save_path='data/',
        ):
        """
        keep prompting until an EXECUTE is outputted or max_calls_per_round is reached
        """
        
        agent_responses = []
        usages = []
        dialog_done = False 
        num_responses = {agent_name: 0 for agent_name in self.robot_agent_names}
        n_calls = 0

        while n_calls < self.max_calls_per_round:
            for agent_name in self.robot_agent_names:
                system_prompt = self.compose_system_prompt(
                    obs, 
                    agent_name,
                    chat_history=chat_history,
                    current_chat=agent_responses,
                    feedback_history=feedback_history,   
                    ) 
                
                agent_prompt = f"你是{agent_name}，请用中文回复你的想法和计划："
                if n_calls == self.max_calls_per_round - 1:
                    agent_prompt = f"""
你是{agent_name}，这是最后一轮发言，你必须综合之前所有讨论，通过 EXECUTE 输出最终计划。
你的回复是：
                    """
                response, usage = self.query_once(
                    system_prompt, 
                    user_prompt=agent_prompt, 
                    max_query=3,
                    )
                
                tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": agent_prompt,
                    },
                    {
                        "sender": agent_name,
                        "message": response,
                    },
                    usage,
                ]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                fname = f'{save_path}/replan{replan_idx}_call{n_calls}_agent{agent_name}_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))  

                num_responses[agent_name] += 1
                # strip all the repeated \n and blank spaces in response: 
                pruned_response = response.strip()
                # pruned_response = pruned_response.replace("\n", " ")
                agent_responses.append(
                    f"[{agent_name}]:\n{pruned_response}"
                    )
                usages.append(usage)
                n_calls += 1
                if 'EXECUTE' in response:
                    if replan_idx > 0 or all([v > 0 for v in num_responses.values()]):
                        dialog_done = True
                        break
 
                if self.debug_mode:
                    dialog_done = True
                    break
            
            if dialog_done:
                break
 
        # response = "\n".join(response.split("EXECUTE")[1:])
        # print(response)  
        return agent_name, response, agent_responses

    def query_once(self, system_prompt, user_prompt, max_query):
        response = None
        usage = None

        if self.debug_mode:
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()

        for n in range(max_query):
            print('querying {}th time'.format(n))
            response, usage = chat_completion(
                model=self.llm_source,
                messages=[
                    {"role": "user", "content": system_prompt + user_prompt},
                ],
                max_tokens=8192,
                temperature=self.temperature,
            )
            if response is not None:
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            print("API returned None, retrying...")
        return response, usage
    
    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            chats = "\n".join(self.latest_chat_history)
            self.round_history.append(
                f"[Chat History]\n{chats}\n[Executed Action]\n{parsed_plan}"
            )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return 

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []