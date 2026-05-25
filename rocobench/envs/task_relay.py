import os
import copy
import time
import cv2
import random
import numpy as np
from pydantic import dataclasses, validator
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dm_control
from dm_control.utils.transformations import mat_to_quat
from pyquaternion import Quaternion
from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS, UR5E_SUCTION_CONSTANTS, PANDA_CONSTANTS

RELAY_ALL_OBJECTS = [
    "panel2",
    "panel4",
    "panel6",
    "blue_square",
    "pink_polygon",
    "yellow_trapezoid",
]
RELAY_CUBE_NAMES = [
    "blue_square",
    "pink_polygon",
    "yellow_trapezoid",
]

RELAY_TASK_CONTEXT = """
[任务描述：流水线装配]
桌面上有7个面板（panel1~panel7），从左到右排列，组成一条流水线。
3个零件（方块）初始散落在左侧，需要通过流水线传递到右侧终点 panel6。

**目标**：将所有3个零件运送到 panel6：
  blue_square → panel6
  pink_polygon → panel6
  yellow_trapezoid → panel6

**机器人分工**（流水线三站）：
  Alice（上料站）：负责 panel1, panel2, panel3，把零件送到 panel3 交接区
  Bob（中转站）：负责 panel3, panel4, panel5，从 panel3 取件送到 panel5 交接区
  Chad（下料站）：负责 panel5, panel6, panel7，从 panel5 取件放到 panel6 终点

**关键规则**：
  - Alice 只能操作 panel1/2/3
  - Bob 只能操作 panel3/4/5
  - Chad 只能操作 panel5/6/7
  - panel3 是 Alice→Bob 交接区
  - panel5 是 Bob→Chad 交接区
  - 每个面板只能放一个零件！

**流水线节奏**：
  - 如果零件在 Alice 区域（panel1/2），Alice 送到 panel3
  - 如果零件在 panel3，Bob 取走送到 panel5
  - 如果零件在 panel5，Chad 取走送到 panel6
  - 交接区被占时，先等或用临时面板暂存
"""

RELAY_ACTION_SPACE = """
[可用动作]
1) PICK <物体名> PLACE <目标面板>
2) WAIT
只有当夹爪为空时才能 PICK。只能 PICK 你能够到的面板上的方块，只能 PLACE 到你能够到的面板上。
**重要**：PLACE 之前检查目标面板是否已有方块！如果有，必须先搬走！

[动作输出格式 - 严格遵守！]
EXECUTE
NAME Alice ACTION <单个动作>
NAME Bob ACTION <单个动作>
NAME Chad ACTION <单个动作>

**关键规则**：
- 只输出3行，每个机器人只输出1行，总共恰好3个 NAME 行
- 不要输出多轮计划（一个机器人只能出现1次）
- 不要加```代码块符号
- 其他解释、思考过程不要出现在 EXECUTE 之后

**正确例子**：
EXECUTE
NAME Alice ACTION PICK blue_square PLACE panel2
NAME Bob ACTION WAIT
NAME Chad ACTION PICK yellow_trapezoid PLACE panel6

**错误例子（绝对不要这样）**：
EXECUTE
NAME Alice ACTION PICK X PLACE Y   ← Alice 第一次
NAME Bob ACTION PICK X PLACE Y
NAME Chad ACTION PICK X PLACE Y
NAME Alice ACTION PICK X PLACE Y   ← ❌ Alice 不能出现第二次！
"""

RELAY_CHAT_PROMPT = """机器人们讨论流水线装配策略。每个机器人发言时：
1) 检查自己区域内有没有零件需要传递
2) 检查交接区（panel3/panel5）是否空闲
3) 提出传递方案
发言顺序为 [Alice],[Bob],[Chad],[Alice] ...
达成一致后，为每个机器人提出**恰好一个** ACTION，然后停止讨论。
"""

RELAY_PLAN_PROMPT = """
分析当前流水线状态，为**当前这一轮**制定计划：
1) 哪些零件需要向右传递？
2) 交接区是否空闲？
3) 每个机器人该做什么？

输出格式（不要加```，不要输出多轮计划）：
EXECUTE
NAME Alice ACTION ...
NAME Bob ACTION ...
NAME Chad ACTION ...
"""


class BlockRelayTask(MujocoSimEnv):
    def __init__(
        self,
        filepath: str = "rocobench/envs/task_sort.xml",
        **kwargs,
    ):
        self.robot_names = ["ur5e_robotiq", "panda", "ur5e_suction"]
        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob",
            "ur5e_suction": "Chad",
        }
        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda",
            "Chad": "ur5e_suction",
        }
        self.robots = dict()
        self.cube_names = RELAY_CUBE_NAMES

        self.cube_to_target = dict(
            yellow_trapezoid="panel6",
            pink_polygon="panel6",
            blue_square="panel6",
        )

        super(BlockRelayTask, self).__init__(
            filepath=filepath,
            task_objects=RELAY_ALL_OBJECTS,
            agent_configs=dict(
                ur5e_robotiq=UR5E_ROBOTIQ_CONSTANTS,
                panda=PANDA_CONSTANTS,
                ur5e_suction=UR5E_SUCTION_CONSTANTS,
            ),
            **kwargs
        )

        self.panel_coords = dict()
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'panel' in geom.name:
                self.panel_coords[geom.name] = self.physics.data.geom(geom.name).xpos.copy()

        self.robots[self.robot_name_map["ur5e_robotiq"]] = SimRobot(
            physics=self.physics, use_ee_rest_quat=False, **UR5E_ROBOTIQ_CONSTANTS)
        self.robots[self.robot_name_map["panda"]] = SimRobot(
            physics=self.physics, use_ee_rest_quat=False, **PANDA_CONSTANTS)
        self.robots[self.robot_name_map["ur5e_suction"]] = SimRobot(
            physics=self.physics, use_ee_rest_quat=False, **UR5E_SUCTION_CONSTANTS)

        self.align_threshold = 0.1
        self.bin_slot_pos = dict()
        for bin_name in ["panel2", "panel4", "panel6"]:
            for slot in ["middle"]:
                self.bin_slot_pos[f"{bin_name}_{slot}"] = self.physics.named.data.site_xpos[f"{bin_name}_{slot}"]

        self.reachable_panels = dict(
            Alice=["panel1", "panel2", "panel3"],
            Bob=["panel3", "panel4", "panel5"],
            Chad=["panel5", "panel6", "panel7"],
        )

    @property
    def use_preplace(self):
        return True

    @property
    def waypoint_std_threshold(self):
        return 0.15

    def get_contact(self):
        contacts = super().get_contact()
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c in self.cube_names]
        contacts["panda"] = [c for c in contacts["panda"] if c in self.cube_names]
        contacts["ur5e_suction"] = [c for c in contacts["ur5e_suction"] if c in self.cube_names]
        return contacts

    def get_obs(self):
        obs = super().get_obs()
        return obs

    def get_allowed_collision_pairs(self):
        ret = []
        cube_ids = [self.physics.model.body(cube).id for cube in self.cube_names]
        table_id = self.physics.model.body("table").id
        panel_ids = []
        for pname in [f"panel{i}" for i in range(1, 8)]:
            try:
                panel_ids.append(self.physics.model.body(pname).id)
            except:
                pass
        for link_id in (self.robots["Alice"].all_link_body_ids +
                        self.robots["Bob"].all_link_body_ids +
                        self.robots["Chad"].all_link_body_ids):
            for cube_id in cube_ids:
                ret.append((link_id, cube_id))
            for pid in panel_ids:
                ret.append((link_id, pid))
            ret.append((link_id, table_id))
        for cube_id in cube_ids:
            ret.append((cube_id, table_id))
            for cube_id2 in cube_ids:
                if cube_id != cube_id2:
                    ret.append((cube_id, cube_id2))
            for pid in panel_ids:
                ret.append((cube_id, pid))
        return ret

    def get_target_pos(self, agent_name, target_name, target_type: str = 'site'):
        ret = None
        robot_name = self.robot_name_map_inv[agent_name]
        if 'panel' in target_name:
            try:
                ret = self.physics.data.geom(target_name).xpos.copy()
            except KeyError:
                return None

            # Apply offsets like in SortTask
            if target_name == 'panel3':
                if 'panda' in robot_name:
                    ret[0] -= 0.12
                    ret[1] -= 0.1
                else:
                    ret[0] += 0.12
                    ret[1] += 0.1
            if target_name == 'panel5':
                if 'panda' in robot_name:
                    ret[0] += 0.12
                    ret[1] -= 0.1
                else:
                    ret[0] -= 0.12
                    ret[1] += 0.1

            ret[2] = 0.5
        elif target_name in self.cube_names:
            sname = f"{target_name}_top"
            ret = self.physics.data.site(sname).xpos.copy()
        else:
            ret = None
        return ret

    def get_closest_panel(self, pos):
        min_dist = float('inf')
        closest = "unknown"
        for pname, ppos in self.panel_coords.items():
            dist = abs(pos[0] - ppos[0])
            if dist < min_dist:
                min_dist = dist
                closest = pname
        return closest

    def describe_cube_state(self, obs, cube_name):
        cube_state = obs.objects[cube_name]
        panel = self.get_closest_panel(cube_state.xpos)
        if panel == "panel6":
            return f"  {cube_name} 在 {panel} ✅ 已到达终点！"
        stage = "上料区" if panel in ["panel1", "panel2"] else "交接区" if panel in ["panel3", "panel5"] else "中转区" if panel == "panel4" else "下料区"
        return f"  {cube_name} 在 {panel}（{stage}，目标：panel6）"

    def describe_robot_state(self, obs, robot_name):
        agent_name = self.robot_name_map[robot_name]
        robot_state = getattr(obs, robot_name)
        contacts = list(robot_state.contacts)
        reachable = ", ".join(self.reachable_panels[agent_name])
        if len(contacts) > 0:
            return f"  {agent_name} 正拿着 {', '.join(contacts)}，可达面板：{reachable}"
        cube_info = []
        for cn in self.cube_names:
            cp = self.get_closest_panel(obs.objects[cn].xpos)
            if cp in self.reachable_panels[agent_name]:
                cube_info.append(cn)
        reachable_cubes = ", ".join(cube_info) if cube_info else "无"
        return f"  {agent_name} 夹爪为空，可达面板：{reachable}，可抓取：{reachable_cubes}"

    def describe_obs(self, obs):
        desp = "[场景描述]\n零件状态：\n"
        for cn in self.cube_names:
            desp += self.describe_cube_state(obs, cn) + "\n"

        completed = sum(1 for cn in self.cube_names
                        if self.get_closest_panel(obs.objects[cn].xpos) == "panel6")
        desp += f"流水线进度：{completed}/3 个零件已到达终点 panel6\n"

        desp += "机器人状态：\n"
        for rn in self.robot_names:
            desp += self.describe_robot_state(obs, rn) + "\n"

        desp += "\n[流水线分工提醒]\n"
        desp += "  Alice（上料站）: panel1, panel2, panel3 → 把零件送到 panel3\n"
        desp += "  Bob（中转站）: panel3, panel4, panel5 → 从 panel3 取件送到 panel5\n"
        desp += "  Chad（下料站）: panel5, panel6, panel7 → 从 panel5 取件放到 panel6\n"

        occupied = {}
        for cn in self.cube_names:
            p = self.get_closest_panel(obs.objects[cn].xpos)
            occupied[p] = cn
        desp += "面板占用情况："
        for i in range(1, 8):
            pn = f"panel{i}"
            if pn in occupied:
                desp += f" {pn}=[{occupied[pn]}]"
            else:
                desp += f" {pn}=[空]"
        desp += "\n"
        return desp

    def describe_task_context(self):
        return RELAY_TASK_CONTEXT

    def get_action_prompt(self):
        return RELAY_ACTION_SPACE

    def get_chat_prompt(self):
        return RELAY_CHAT_PROMPT

    def get_plan_prompt(self):
        return RELAY_PLAN_PROMPT

    def get_agent_prompt(self, obs, agent_name, include_response_instructions=True):
        reachable = ", ".join(self.reachable_panels[agent_name])
        other_robots = ", ".join([r for r in self.robots.keys() if r != agent_name])
        cube_states = "\n".join([self.describe_cube_state(obs, cn) for cn in self.cube_names])
        robot_state = self.describe_robot_state(obs, self.robot_name_map_inv[agent_name])

        roles = {"Alice": "上料站，把零件送到panel3", "Bob": "中转站，从panel3取件送到panel5", "Chad": "下料站，从panel5取件放到panel6"}
        prompt = f"""
你是 {agent_name}（{roles[agent_name]}），与 {other_robots} 协作完成流水线装配。
你只能操作：{reachable}
当前零件：
{cube_states}
{robot_state}
目标：所有零件到达 panel6
"""
        if include_response_instructions:
            prompt += """请用中文简要分析，提出你这一轮的动作。
如果需要继续讨论输出 PROCEED。如果全部同意，输出 EXECUTE 加最终计划。
"""
        return prompt

    def get_reward_done(self, obs):
        reward = 1
        done = True
        for cube_name, target_panel in self.cube_to_target.items():
            cube_pos = self.physics.data.body(cube_name).xpos
            target_pos = self.panel_coords[target_panel]
            if abs(cube_pos[0] - target_pos[0]) > self.align_threshold:
                done = False
                reward = 0
                break
        return reward, done

    def get_task_feedback(self, llm_plan, pose_dict):
        feedback = ""
        for agent_name, action_str in llm_plan.action_strs.items():
            if 'PICK' in action_str and 'PLACE' in action_str:
                place_target = action_str.split('PLACE')[1].strip().replace(' ', '_')
                reachable = self.reachable_panels.get(agent_name, [])
                if place_target not in reachable:
                    feedback += f"{agent_name} 不能放到 {place_target}（超出可达范围 {reachable}）；"
        return feedback

    def get_target_pos(self, agent_name, target_name):
        robot_name = self.robot_name_map_inv[agent_name]
        if target_name in self.panel_coords:
            ret = self.panel_coords[target_name].copy()

            # Apply offsets for shared panels
            if target_name == 'panel3':
                if 'panda' in robot_name:
                    ret[0] -= 0.12
                    ret[1] -= 0.1
                else:
                    ret[0] += 0.12
                    ret[1] += 0.1
            if target_name == 'panel5':
                if 'panda' in robot_name:
                    ret[0] += 0.12
                    ret[1] -= 0.1
                else:
                    ret[0] -= 0.12
                    ret[1] += 0.1

            ret[2] = 0.5
            return ret
        for cn in self.cube_names:
            if target_name == cn:
                return self.physics.data.body(cn).xpos.copy()
        return None

    def get_target_quat(self, agent_name, target_name):
        return np.array([1, 0, 0, 0])

    def get_grasp_site(self, obj_name):
        if obj_name in self.cube_names:
            return f"{obj_name}_top"
        return None

    def get_object_joint_name(self, obj_name):
        if obj_name in self.cube_names:
            return f"{obj_name}_joint"
        return None

    def get_graspable_objects(self):
        return self.cube_names

    def check_reach_range(self, agent_name, pos):
        reachable = self.reachable_panels[agent_name]
        min_x = min([self.panel_coords[p][0] for p in reachable]) - 0.3
        max_x = max([self.panel_coords[p][0] for p in reachable]) + 0.3
        return min_x <= pos[0] <= max_x
