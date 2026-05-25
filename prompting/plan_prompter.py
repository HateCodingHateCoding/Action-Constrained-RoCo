import os
import json
import pickle
import numpy as np
from rocobench.envs import MujocoSimEnv, EnvState
from datetime import datetime
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from .llm_api import chat_completion
from typing import List, Tuple, Dict, Union, Optional, Any

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
PLACEHOLDER_REMOVED = True


def get_chat_prompt(env: MujocoSimEnv):
    robot_names = env.get_sim_robots().keys()
    talk_order_str = ",".join([f"[{name}]" for name in robot_names])
    chat_prompt = f"""
机器人们互相讨论以找到最佳策略。它们仔细分析其他机器人的回复，并利用[环境反馈]来改进计划。
发言顺序为 {talk_order_str}... 达成一致后，严格按照[动作输出格式]输出最终计划，然后停止讨论。
完整的讨论和最终计划如下：
    """
    return chat_prompt 


def get_plan_prompt(env: MujocoSimEnv):
    return """
请逐步分析任务，找到协调各机器人的最佳策略。为每个机器人提出**恰好一个**动作的计划。
利用[环境反馈]来改进计划。严格按照[动作输出格式]来输出计划。
你的推理过程和最终计划输出如下：
    """
    

class SingleThreadPrompter:
    """
    At each round, queries LLM once for each action plan, 
    query again with environment feedback if the action plan cannot be executed
    """
    def __init__(
        self, 
        env: MujocoSimEnv,
        parser: LLMResponseParser, 
        feedback_manager: FeedbackManager,
        comm_mode: str = "plan", # or chat
        use_waypoints: bool = False,
        use_history: bool = True,
        max_api_queries: int = 3,
        num_replans: int = 3,
        debug_mode: bool = False,   
        temperature: float = 0,
        max_tokens: int = 1000, 
        llm_source: str = "gpt-4",
    ):
        self.env = env 
        self.robot_agent_names = env.get_sim_robots().keys()
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.comm_mode = comm_mode
        self.max_api_queries = max_api_queries
        self.num_replans = num_replans
        self.debug_mode = debug_mode 
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.temperature = temperature
        self.llm_source = llm_source
        self.max_tokens = max_tokens

        self.round_history = [] # [obs_t, action_t] but only if action_t got executed
        self.failed_plans = [] # could inherit from previous round if the final plan failed to execute in env.
        self.response_history = [] # [response_t]
        

    def save_state(self, save_path, fname = 'prompter_state.pkl'):
        state_dict = dict(
            round_history=self.round_history,
            failed_plans=self.failed_plans,
        )
        save_path = os.path.join(save_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, load_path, fname = 'prompter_state.pkl'):
        load_path = os.path.join(load_path, fname)
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        self.round_history = state_dict["round_history"]
        self.failed_plans = state_dict["failed_plans"]

    def compose_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}"
        ret += f"== Current Round ==\n"
        return ret
        
    def compose_system_prompt(
        self,
        obs_desp: str,
        plan_feedbacks: List[str] = [], 
        ):
        
        task_desp = self.env.describe_task_context() # should include task rules
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n" 
        
        if self.use_history:
            history_desp = self.compose_round_history() 
            full_prompt += history_desp + "\n" 
        
        full_prompt += obs_desp + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "以下计划执行失败，请改进以避免碰撞并平稳到达目标：\n"
            execute_feedback += "\n".join(self.failed_plans) 
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "之前的计划需要改进：\n"
            feedback_prompt += "\n".join(plan_feedbacks) + "\n"
            full_prompt += feedback_prompt
        
        if self.comm_mode == "plan":
            comm_prompt = get_plan_prompt(self.env)
        elif self.comm_mode == "chat":
            comm_prompt = get_chat_prompt(self.env) 
        else:
            raise NotImplementedError
        full_prompt += comm_prompt

        return full_prompt 

    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        plan_feedbacks = []
        response_history = []
        obs_desp = self.env.describe_obs(obs)
        for i in range(self.num_replans): 
            system_prompt = self.compose_system_prompt(obs_desp, plan_feedbacks)
            response, usage = self.query_once(
                system_prompt, user_prompt=""
                ) # NOTE: single_thread doesn't use user role
            response_history.append(response)
            
            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": "",
                    },
                    {
                        "sender": "Planner",
                        "message": response,
                    },
                    usage,
                ]
            fname = f'{save_path}/replan{i}_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))  
            
            curr_feedback = "None"
            # try parsing 
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response) 
            if not parse_succ: 
                execute_str = 'EXECUTE' + response.split('EXECUTE')[-1]
                curr_feedback = f"""
解析失败！{parsed_str}
之前的回复：{execute_str}
请严格按照[动作输出格式]重新输出！
                """
                plan_feedbacks.append(curr_feedback)
                ready_to_execute = False  
            # give env. feedback 
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
                    "message": (response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w')) 

            if ready_to_execute:
                plan_str = parsed_str
                break  
        self.response_history = response_history
        return ready_to_execute, llm_plans, plan_feedbacks, response_history


    def query_once(self, system_prompt, user_prompt=""):
        response = None
        usage = None
        if self.debug_mode:
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()

        for n in range(self.max_api_queries):
            print('querying {}th time'.format(n))
            response, usage = chat_completion(
                model=self.llm_source,
                messages=[
                    {"role": "user", "content": system_prompt},
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
        return response, usage

    

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            responses = "\n".join(self.response_history)
            self.round_history.append(
                f"[Response History]\n{responses}\n{obs_desp}\n[Executed Action]\n{parsed_plan}"
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
        self.response_history = []