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

SORT_ALL_OBJECTS=[
    "panel2",
    "panel4",
    "panel6",
    "blue_square",
    "pink_polygon",
    "yellow_trapezoid",
]
ONE_OBJ_EACH=[
    "blue_square",
    "pink_polygon",
    "yellow_trapezoid",
] 
SORTING_BIN_NAMES=[
    "panel2",
    "panel4",
    "panel6",
]

SORT_TASK_CONTEXT="""
桌面上有7个面板，从左到右排列为：panel1,...,panel7。它们组成一条直线流水线，panel1 靠近 panel2，离 panel7 最远。
桌上有3个方块，每个机器人必须把方块放到正确的目标面板上，它们的（方块，目标面板）对应关系为：
Alice: (blue_square, panel2),
Bob: (pink_polygon, panel4),
Chad: (yellow_trapezoid, panel6)。
有3个机器人，每个机器人的臂展有限，只能从特定面板上拿取方块或放置方块。机器人与可达面板的对应关系：
(Alice, [panel1, panel2, panel3])
(Bob, [panel3, panel4, panel5])
(Chad, [panel5, panel6, panel7])
"""
 
SORT_TASK_DIALOG_PROMPT=""

SORT_TASK_CHAT_PROMPT="""机器人们互相讨论以找到最佳策略。每个机器人发言时，先分析自己的能力（例如：我是Bob，我只能够到panel3-5），然后验证其他机器人的判断（例如：我是Alice，这个梯形方块确实离Chad太远了），最后决定如何协作完成任务（例如：因此我先把它移到panel3，这样Chad就能够到了）。
仔细分析[环境反馈]、场景描述和其他机器人的回复来协调合作。发言顺序为 [Alice],[Bob],[Chad],[Alice] ...，达成一致后，为每个机器人提出**恰好一个** ACTION，然后停止讨论。
完整的对话记录和最终计划如下：
"""

SORT_TASK_PLAN_PROMPT="""
请逐步思考，为每个机器人制定最佳策略来完成目标或帮助其他机器人。仔细考虑[环境反馈]和场景描述。
判断每个机器人能够到哪些方块和面板。每一轮为每个机器人规划**恰好一个** ACTION。
"""

SORTING_ACTION_SPACE="""
[可用动作]
1) PICK <物体名> PLACE <目标位置>
2) WAIT
只有当夹爪为空时才能 PICK 物体。PLACE 的目标位置应为面板(panel)。
[动作输出格式]
你必须先输出 'EXECUTE\n'，然后为每个机器人给出**恰好一个**动作，每个动作占一行。
示例: 'EXECUTE\nNAME Alice ACTION PICK red_square PLACE panel3\nNAME Bob ACTION WAIT\nNAME Chad ACTION PICK green_trapezoid PLACE panel6\n'
"""

class SortOneBlockTask(MujocoSimEnv):
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
        self.obj_to_panel = dict()
        self.cube_names = ONE_OBJ_EACH
        self.cube_to_bin = dict(
            blue_square="panel2",
            pink_polygon="panel4",
            yellow_trapezoid="panel6",
        )
        
        super(SortOneBlockTask, self).__init__(
            filepath=filepath, 
            task_objects=SORT_ALL_OBJECTS,
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
                self.panel_coords[geom.name] = geom.pos

        self.robots[
            self.robot_name_map["ur5e_robotiq"]
            ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **UR5E_ROBOTIQ_CONSTANTS,
        )
        self.robots[
            self.robot_name_map["panda"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **PANDA_CONSTANTS,
        )
        self.robots[
            self.robot_name_map["ur5e_suction"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **UR5E_SUCTION_CONSTANTS,
        )

        self.align_threshold = 0.1
        self.bin_slot_pos = dict()
        for bin_name in SORTING_BIN_NAMES: 
            for slot in ["middle"]: # ["left", "middle", "right"]:
                self.bin_slot_pos[f"{bin_name}_{slot}"] = self.physics.named.data.site_xpos[f"{bin_name}_{slot}"]
        
        self.robot_to_bin = dict(
            ur5e_robotiq="panel2",
            panda="panel4",
            ur5e_suction="panel6",
        )
        self.bin_x_coords = {
            'panel2': -0.8,
            'panel4': 0,
            'panel6': 0.8,
        }         

        self.cube_targets = dict(
            Alice=("blue_square", "panel2"),
            Bob=("pink_polygon", "panel4"),
            Chad=("yellow_trapezoid", "panel6"),
        )
        self.reachable_panels = dict(
            Alice=["panel1", "panel2", "panel3"],
            Bob=["panel3", "panel4", "panel5"],
            Chad=["panel5", "panel6", "panel7"],
        )
        
    @property
    def use_preplace(self):
        return True

    def get_agent_prompt(self, obs, agent_name, include_response_instructions=True):
        cube_name, bin_name = self.cube_targets[agent_name]
        other_robots = ", ".join(
            [r for r in self.robots.keys() if r != agent_name]
        )
        robot_name = self.get_robot_name(agent_name)
        agent_state = self.describe_robot_state(obs, robot_name)
        agent_state = agent_state.replace(f"{agent_name}'s", "Your")
        
        cube_states = "\n".join(
            [self.describe_cube_state(obs, cube_name) for cube_name in self.cube_names]
        )
        reachable_panels = ", ".join(self.reachable_panels[agent_name])

        agent_prompt = f"""
7 panels on the table, ordered left to right: panel1,...,panel7. They form a straight assembly line, panel1 is closed to panel2 and farthest from panel7.
You are robot {agent_name} in front of {bin_name}. You are collaborating with {other_robots} to sort cubes into their target panels. The task is NOT done until all three cubes are sorted.
At current round: 
{cube_states}
Your goal is to place {cube_name} on {bin_name}, but you can only reach {reachable_panels}: this means you can only pick cubes from these panels, and can only place cubes on these panels.
{agent_state}
Never forget you are {agent_name}! Never forget you can only reach {reachable_panels}!
Think step-by-step about the task and others' response. Carefully check and correct them if they made a mistake. 
Improve your plans if given [Environment Feedback].
"""
        if include_response_instructions:
            agent_prompt += f"""
When you respond, tell others about your goal and all constraints. Respond very concisely but informatively, and do not repeat what others have said.
Discuss with others to come up with the best plan, e.g. if your cube is out of your reach, ask others for help, and you can do the same for them. 
Propose exactly one action for yourself at the **current** round, select from [Action Options].
End your response by either: 1) output PROCEED, if the plans require further discussion, or 2) If everyone has made proposals and got approved, output EXECUTE and the final plan, must strictly follow [Action Output Instruction]!
In the plan, at least one robot should be acting, you can't all WAIT.
"""
# Example response #1:
# [Reasons] I am {agent_name}, I must put blue_square on panel2, but I can't reach blue_square for now. Since Chad needs yellow_trapezoid, I propose to help Chad move it closer. What does everyone think?
# [Proposal] PICK yellow_trapezoid PLACE panel3
# [Decision] PROCEED
# Example response #2:
# [Reasons] I am Chad, My previous proposal was approved and no need for update. I approve the latest proposals from Alice and Bob.
# [Proposal] WAIT 
# [Decision] 
# EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION PICK blue_square PLACE panel3\nNAME Chad WAIT
        

        # if agent_name == "Alice":
        #     agent_prompt += f"You must put blue_square in panel2" #you can only reach panel2, panel1, panel3. But you can't reach panel5, panel7, or other bins."
        # elif agent_name == "Bob":
        #     agent_prompt += "You must put pink_polygon in panel4" # you can only reach panel4, panel3, panel5. But you can't reach panel1, panel7, or other bins."
        # elif agent_name == "Chad":
        #     agent_prompt += "You must put yellow_trapezoid in panel6" #you can only reach panel6, panel5, panel7. But you can't reach panel1, panel3, or other bins."
 
        return agent_prompt

        
    def get_action_prompt(self) -> str:
        return SORTING_ACTION_SPACE

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]
    
    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name]
    
    def get_robot_config(self) -> Dict[str, Dict[str, Any]]:
        return self.agent_configs
    
    def get_sim_robots(self) -> Dict[str, SimRobot]:
        """NOTE this is indexed by agent name, not actual robot names"""
        return self.robots

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == self.robot_name_map["ur5e_robotiq"]:
            return dict(x=(-1.4, -0.1), y=(0.1, 1.3), z=(0.16, 1))
        
        elif robot_name == "panda" or robot_name == self.robot_name_map["panda"]:
            return dict(x=(-0.7, 0.7), y=(-0.21, 1.3), z=(0.16, 1))
        
        elif robot_name == "ur5e_suction" or robot_name == self.robot_name_map["ur5e_suction"]:
            return dict(x=(0.2, 1.5), y=(0.1, 1.3), z=(0.16, 1))
        
        else:
            raise NotImplementedError
    
    def check_reach_range(self, robot_name, point: Tuple[float, float, float]) -> bool: 
        reach_range = self.get_robot_reach_range(robot_name)
        for i, axis in enumerate(["x", "y", "z"]):
            if point[i] < reach_range[axis][0] or point[i] > reach_range[axis][1]:
                return False
        return True

    def sample_initial_scene(self):
        # find the pre-defined panel positions in the xml
        tosample_panels = []
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'panel' in geom.name:
                tosample_panels.append(
                    (geom.name, geom.pos, geom.size)
                )
        assert len(tosample_panels) >= 3, "Not enough panel positions to sample from"
        
        far_panels = dict()
        far_panels['square'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] > 0.15] 
        far_panels['polygon'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] < -0.7 or tup[1][0] > 0.9]
        far_panels['trapezoid'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] < -0.15]

        # sample the panel positions
        occupied_idxs = []
        for i, name in enumerate(ONE_OBJ_EACH):
            try:
                qpos_slice = self.physics.named.data.qpos._convert_key(f"{name}_joint")
            except KeyError:
                print('Skipping object: ', name, ' because its _joint does not exist in the xml file')
                continue
            assert int(qpos_slice.stop - qpos_slice.start) == 7, "object qpos must be 7-dim"
            start = qpos_slice.start
            stop = qpos_slice.stop
            shape = name.split('_')[1]
            
            idx = self.random_state.choice(far_panels[shape]) 
            # remove this index from the list of available panels
            for shape, idxs in far_panels.items():
                if idx in idxs:
                    idxs.remove(idx)

            panel_name, panel_pos, panel_size = tosample_panels[idx]
            self.obj_to_panel[name] = panel_name
            # sample a random position within the panel
            # new_pos = self.random_state.uniform(
            #     low=panel_pos - panel_size / 2 + 0.001,
            #     high=panel_pos + panel_size / 2 - 0.001, 
            # )
            new_pos = panel_pos.copy()
 
            new_quat = Quaternion(
                axis=[0,0,1], 
                angle=self.random_state.uniform(low=0, high=2*np.pi)
                ) 
            new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
            old_pos = self.physics.named.data.qpos[start:stop]
            new_pos[2] = old_pos[2]
            self.physics.named.data.qpos[start:stop] = np.concatenate([new_pos, new_quat])
            
             
        self.physics.forward()
        self.physics.step(100)
    
    def get_obs(self):
        obs = super().get_obs()
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation"
        return obs
    
    def describe_robot_state(self, obs: EnvState, robot_name: str):
        agent_name = self.get_agent_name(robot_name)
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos

        dist_to_panels = [(name, np.linalg.norm(robot_state.ee_xpos[:2] - pos[:2])) for name, pos in self.panel_coords.items()]
        closest_panel = min(dist_to_panels, key=lambda x: x[1])[0]
        robot_desp = "" # f"{agent_name}'s gripper is closest to {closest_panel}, "

        if len(robot_state.contacts) == 0:
            obj = "empty"
        else:
            obj = "holding " + ",".join([c for c in robot_state.contacts])
        # robot_desp += f"{agent_name}'s gripper is at ({x:.2f} {y:.2f} {z:.2f}), holding {obj},"
        robot_desp += f"{agent_name}'s gripper is {obj},"
        
        reachables = []
        not_reachables = []
        for block_name in ONE_OBJ_EACH:
            block_state = obs.objects[block_name]
            top_site = block_state.sites[f'{block_name}_top']
            if self.check_reach_range(robot_name, top_site.xpos):
                reachables.append(block_name)
            else:
                not_reachables.append(block_name)

        if len(reachables) > 0:
            robot_desp += f" can reach cubes: "
            for obj in reachables:
                robot_desp += f"{obj}, "
        if len(not_reachables) > 0:
            robot_desp += f"can't reach cubes: "
            for obj in not_reachables:
                robot_desp += f"{obj}, "
        
        return robot_desp
    
    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        ret = []
        cube_ids = [self.physics.model.body(cube).id for cube in self.cube_names ]
       
        table_id = self.physics.model.body("table").id 
        bin_ids = [self.physics.model.body(bin_name).id for bin_name in SORTING_BIN_NAMES]
        
        for link_id in self.robots["Alice"].all_link_body_ids + self.robots["Bob"].all_link_body_ids + self.robots["Chad"].all_link_body_ids:
            for cube_id in cube_ids:
                ret.append((link_id, cube_id))
            for bin_id in bin_ids:
                ret.append((link_id, bin_id))
            ret.append((link_id, table_id))

        for cube_id in cube_ids:
            ret.append((cube_id, table_id))
            for cube_id2 in cube_ids:
                if cube_id != cube_id2:
                    ret.append((cube_id, cube_id2))
            for bin_id in bin_ids:
                ret.append((cube_id, bin_id)) 

        return ret
    
    def get_cube_panel(self, obs, cube_name: str):
        cube_state = obs.objects[cube_name]
        dist_to_panels = [(name, np.linalg.norm(cube_state.xpos[:2] - pos[:2])) for name, pos in self.panel_coords.items()]
        closest_panel = min(dist_to_panels, key=lambda x: x[1])[0] 
        for pname in ["panel2", "panel4", "panel6"]:
            if pname in obs.objects[cube_name].contacts:
                closest_panel = pname
                break
        return closest_panel

    def describe_cube_state(self, obs: EnvState, cube_name: str):
        cube_state = obs.objects[cube_name]
        top_site = cube_state.sites[f'{cube_name}_top']
        x, y, z = top_site.xpos
        cube_desp = ""
        for slot_name, pos in self.bin_slot_pos.items():
            if np.linalg.norm(pos[:2] - cube_state.xpos[:2]) < self.align_threshold:
                slot_name = "_".join(slot_name.split("_")[:-1])
                cube_desp = f"{cube_name} is in {slot_name}"
                break 
        if len(cube_desp) == 0:
            closest_panel = self.get_cube_panel(obs, cube_name)
            cube_desp = f"{cube_name} is on {closest_panel}"
        return cube_desp

    def describe_obs(self, obs: EnvState):
        """ For each cube, just describe whether it's on a bin, or between which two bins, no output numerical coordinates """
        object_desp = "[Scene description]\n"  
        for cube_name in ONE_OBJ_EACH:
            object_desp += self.describe_cube_state(obs, cube_name)+"\n" 
        
        robot_desp = ""
        for robot_name, agent_name in self.robot_name_map.items():
            robot_desp += self.describe_robot_state(obs, robot_name)+"\n"
        robot_desp = robot_desp[:-2]+".\n"
        full_desp = object_desp + robot_desp
        return full_desp  
    
    def get_reward_done(self, obs):
        reward = 1
        done = True
        for block_name in ONE_OBJ_EACH:
            block_state = obs.objects[block_name] 
            correct_bin = self.cube_to_bin[block_name]
            bin_pos = self.bin_slot_pos[f"{correct_bin}_middle"]
            if np.linalg.norm(bin_pos[:2] - block_state.xpos[:2]) > self.align_threshold and (correct_bin not in obs.objects[block_name].contacts):
                reward = 0
                done = False
                break
        return reward, done
 

    def describe_task_context(self):
        return SORT_TASK_CONTEXT 
    
    def get_grasp_site(self, obj_name: str = "pink_polygon") -> str:
        return f"{obj_name}_top"
    
    def get_target_pos(self, agent_name, target_name, target_type: str = 'site') -> Optional[np.ndarray]:
        """ useful for parsing place targets """
        ret = None 
        robot_name = self.robot_name_map_inv[agent_name]
        if 'panel' in target_name:
            try:
                ret = self.physics.data.geom(target_name).xpos.copy()
            except KeyError:
                return None  

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
        # print(f"Agent: {agent_name} target site for {target_name} is {ret}")
 
        return ret 

    def get_contact(self):
        contacts = super().get_contact()
        # temp fix! 
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c in self.cube_names]
        contacts["panda"] = [c for c in contacts["panda"] if c in self.cube_names]
        contacts["ur5e_suction"] = [c for c in contacts["ur5e_suction"] if c in self.cube_names]
        return contacts
    
    def get_task_feedback(self, llm_plan, pose_dict):
        feedback = ""
        for agent_name, action_str in llm_plan.action_strs.items():
            if ('PICK' in action_str and 'PLACE' not in action_str) or \
                ('PLACE' in action_str and 'PICK' not in action_str):
                    feedback += f"{agent_name}'s ACTION must contain both PICK and PLACE"
            if 'PICK' in action_str and 'PLACE' in action_str:
                obj = action_str.split('PICK')[1].split('PLACE')[0].strip()
                target = action_str.split('PLACE')[1].strip()
                if obj in self.cube_names and target in self.cube_to_bin.values():
                    correct_panel = self.cube_to_bin[obj]
                    if correct_panel not in target:
                        valid_panels = ", ".join(
                            [correct_panel, 'panel3', 'panel5']
                        )
                        feedback += f"{agent_name}'s ACTION is not valid, {obj} cube can only be placed on {valid_panels}, but not on {target}"
        if all(['WAIT' in action_str for action_str in llm_plan.action_strs.values()]):
            feedback += f"You can't all WAIT. The task is not complete, at least one robot should be acting."
        return feedback 
    
    def get_object_joint_name(self, obj_name):
        return f"{obj_name}_joint"

    # ---------------- RL extension hooks ----------------
    PANEL_NAMES = [f"panel{i}" for i in range(1, 8)]

    def get_action_vocab(self) -> Dict[str, List[str]]:
        return dict(
            agents=["Alice", "Bob", "Chad"],
            objects=["WAIT"] + list(ONE_OBJ_EACH),
            targets=list(self.PANEL_NAMES),
        )

    def _agent_reachable_panel_idxs(self, agent_name: str) -> List[int]:
        names = self.reachable_panels[agent_name]
        return [self.PANEL_NAMES.index(n) for n in names]

    def get_action_mask(self, obs: EnvState) -> Dict[str, Any]:
        vocab = self.get_action_vocab()
        n_obj = len(vocab["objects"])      # WAIT + 3 cubes
        n_tgt = len(vocab["targets"])      # 7 panels
        ret = {}
        for agent_name in vocab["agents"]:
            robot_name = self.robot_name_map_inv[agent_name]
            robot_state = getattr(obs, robot_name)
            # holding == has any cube name in contacts
            holding = any(c in self.cube_names for c in robot_state.contacts)

            obj_mask = np.zeros(n_obj, dtype=bool)
            target_mask = np.zeros((n_obj, n_tgt), dtype=bool)

            # WAIT is always legal; targets dim is irrelevant for WAIT row
            obj_mask[0] = True
            target_mask[0, :] = True

            # If hands are full, only WAIT is legal (a separate RELEASE/PLACE
            # is bundled inside PICK X PLACE Y in this benchmark).
            if holding:
                ret[agent_name] = dict(obj_mask=obj_mask, target_mask=target_mask)
                continue

            reach_panel_idxs = self._agent_reachable_panel_idxs(agent_name)
            for oi, oname in enumerate(vocab["objects"][1:], start=1):
                # cube must be in this agent's reach
                top_site = obs.objects[oname].sites[f"{oname}_top"]
                if not self.check_reach_range(robot_name, top_site.xpos):
                    continue
                obj_mask[oi] = True
                # only the panels this agent can reach are valid place targets
                for ti in reach_panel_idxs:
                    target_mask[oi, ti] = True

                # ---- task-specific cube/target compatibility ----
                # (mirrors get_task_feedback): each cube is only allowed on its
                # correct bin or the shared relay panels (panel3, panel5).
                correct_panel = self.cube_to_bin[oname]  # panel2/4/6
                allowed = {correct_panel, "panel3", "panel5"}
                for ti in range(n_tgt):
                    if vocab["targets"][ti] not in allowed:
                        target_mask[oi, ti] = False
            # if no PICK is feasible after masking, target row stays zero;
            # ensure at least WAIT remains.
            ret[agent_name] = dict(obj_mask=obj_mask, target_mask=target_mask)
        return ret

    def _cube_progress(self, obs: EnvState) -> Dict[str, Dict[str, float]]:
        """For each cube: distance to its correct bin and whether it's already there."""
        info = {}
        for cname in self.cube_names:
            target_bin = self.cube_to_bin[cname]
            bin_xy = self.bin_slot_pos[f"{target_bin}_middle"][:2]
            cube_xy = obs.objects[cname].xpos[:2]
            d = float(np.linalg.norm(bin_xy - cube_xy))
            done = (d < self.align_threshold) or (target_bin in obs.objects[cname].contacts)
            info[cname] = dict(dist=d, done=done, target=target_bin)
        return info

    def get_rl_reward(self, prev_obs: EnvState, obs: EnvState,
                      action_info: Dict) -> Tuple[float, Dict]:
        """
        Composite reward:
          + task: per-cube first-time placement (+10) and global success (+50)
          + shaping: potential-based on cube-to-target distance
          + handoff: helping move someone else's cube into reach (+2)
          - invalid action / collision penalties
          - per-step time cost (-0.05) and all-WAIT (-0.5)
        """
        prev_info = self._cube_progress(prev_obs)
        info = self._cube_progress(obs)

        r = 0.0
        breakdown = {}

        # 4.1 task reward (sub-goal first achievement)
        r_goal = 0.0
        for cname, cur in info.items():
            if cur["done"] and (not prev_info[cname]["done"]):
                r_goal += 10.0
        breakdown["r_goal"] = r_goal
        r += r_goal

        # 4.1 global success bonus
        all_done = all(v["done"] for v in info.values())
        r_done = 50.0 if (all_done and not all(v["done"] for v in prev_info.values())) else 0.0
        breakdown["r_done"] = r_done
        r += r_done

        # 4.2 potential-based shaping (negative distance sum)
        gamma = float(action_info.get("gamma", 0.99))
        phi_prev = -sum(v["dist"] for v in prev_info.values())
        phi_cur = -sum(v["dist"] for v in info.values())
        r_shape = gamma * phi_cur - phi_prev
        breakdown["r_shape"] = r_shape
        r += r_shape

        # 4.3 handoff: any agent moved a cube to extend reachability for the
        # cube's owner, while the cube is not yet at goal.
        r_handoff = 0.0
        for cname, cur in info.items():
            if cur["done"]:
                continue
            owner = next(a for a, (cube, _) in self.cube_targets.items() if cube == cname)
            owner_robot = self.robot_name_map_inv[owner]
            prev_reach = self.check_reach_range(owner_robot, prev_obs.objects[cname].xpos)
            cur_reach = self.check_reach_range(owner_robot, obs.objects[cname].xpos)
            if (not prev_reach) and cur_reach:
                r_handoff += 2.0
        breakdown["r_handoff"] = r_handoff
        r += r_handoff

        # 4.4 penalties from action_info
        n_invalid = int(action_info.get("n_invalid", 0))
        r_invalid = -2.0 * n_invalid
        breakdown["r_invalid"] = r_invalid
        r += r_invalid

        n_collision = int(action_info.get("n_collision", 0))
        r_collision = -1.0 * n_collision
        breakdown["r_collision"] = r_collision
        r += r_collision

        if bool(action_info.get("all_wait", False)):
            r += -0.5
            breakdown["r_all_wait"] = -0.5

        r_step = -0.05
        r += r_step
        breakdown["r_step"] = r_step

        breakdown["done"] = all_done
        return float(r), breakdown

    def chat_mode_prompt(self, chat_history: List[str] = []):
        return SORT_TASK_CHAT_PROMPT
        
    def central_plan_prompt(self):
        return SORT_TASK_PLAN_PROMPT
    
    def dialog_mode_prompt(self):
        return SORT_TASK_DIALOG_PROMPT
 
    def get_waypoint_feedback(
        self, 
        waypoint_paths: Dict[str, List],
        display = False,
        save_img = False,
        img_path = 'test.jpg',
        ):
        """
        Give feedback to the robots about the waypoints they are going to visit.
        """
        bad_waypoints = defaultdict(list)
        for robot_name, path in waypoint_paths.items(): 
            for waypoint in path:
                if not self.check_reach_range(robot_name, waypoint):
                    bad_waypoints[robot_name].append(waypoint)
        summ = ""
        for name, waypoints in bad_waypoints.items():
            summ += f"{name}: {waypoints} \n"
        if display:
            print(summ)
            self.render_point_cloud = True 
            obs = self.get_obs() 
            path_ls = list(waypoint_paths.values())
            visualize_voxel_scene(obs.scene, path_pts=path_ls, path_colors=[], save_img=save_img, img_path=img_path)
        if summ == "":
            summ = "Reachability feedback: sucess."
        else:
            summ = "Reachability feedback: failed. These steps are beyond the robot's reach: \n" + summ
        return summ 
            

        


if __name__ == "__main__":
    env = SortOneBlockTask(np_seed=0, render_point_cloud=0)
    obs = env.reset()
    # obs.scene.show()
    print(env.get_action_prompt())
    print(env.get_agent_prompt(obs, "Alice"))
    breakpoint()
    img=env.physics.render(camera_id="teaser", height=480, width=600)


