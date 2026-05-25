"""Bridge between the trained RL policy and the real MuJoCo task env.

The real env executes plans via:
    LLM response (string) -> LLMResponseParser -> List[LLMPathPlan]
                          -> PlannedPathPolicy -> SimAction -> env.step

The trained MAPPO policy outputs (obj_idx, target_idx) per agent. We don't
need an LLM at all -- we just synthesize a response string in the exact
format the parser expects, then reuse the existing RRT + execute chain.

Public surface:
    - obs_to_rl_features(env, obs)  -> dict of per-agent obs/mask
    - rl_action_to_response(env, codec, joint_action)  -> EXECUTE-format str
    - RealEnvActionAdapter            -> wraps the trained agent as a
                                         "policy" with a .response(obs) call
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np

from rocobench.envs import EnvState, MujocoSimEnv
from rocobench.rl.action_codec import ActionCodec
from rocobench.rl.sort_symbolic_env import (
    CUBES, CUBE_TO_BIN, CUBE_OWNER, RELAY_PANELS, _PANEL_X
)


def obs_to_rl_features(env: MujocoSimEnv, obs: EnvState,
                       sticky_progress: Optional[set] = None) -> Dict[str, Any]:
    """Convert a real EnvState into the same flat features used by the
    symbolic env's MAPPO policy.

    Layout (must match SortSymbolicEnv.get_obs):
        [cube_panel_onehot (3 x 7) | holding (3,) | progress (3,) | agent_id (3,)]

    Args:
        sticky_progress: optional set of cube names that have ever been at
            their goal during this episode. If a cube name is in this set,
            its progress flag is forced to 1 even if physics drift moved
            the cube slightly (matches the symbolic env's ever_done flag,
            so the policy stops trying to re-place finished cubes).
    """
    sticky_progress = sticky_progress if sticky_progress is not None else set()
    panels = [f"panel{i}" for i in range(1, 8)]
    agents = ["Alice", "Bob", "Chad"]

    cube_onehot = np.zeros((len(CUBES), len(panels)), dtype=np.float32)
    for i, cube in enumerate(CUBES):
        # find current panel: nearest by xy, with contact override
        cube_state = obs.objects[cube]
        chosen = None
        for pname in ["panel2", "panel4", "panel6"]:
            if pname in cube_state.contacts:
                chosen = pname
                break
        if chosen is None:
            best = None
            best_d = float("inf")
            for pname in panels:
                geom = env.physics.data.geom(pname).xpos
                d = float(np.linalg.norm(geom[:2] - cube_state.xpos[:2]))
                if d < best_d:
                    best_d, best = d, pname
            chosen = best
        cube_onehot[i, panels.index(chosen)] = 1.0

    holding = np.zeros(len(agents), dtype=np.float32)
    for ai, ag in enumerate(agents):
        robot_name = env.robot_name_map_inv[ag]
        rs = getattr(obs, robot_name)
        if any(c in CUBES for c in rs.contacts):
            holding[ai] = 1.0

    progress = np.zeros(len(CUBES), dtype=np.float32)
    for i, cube in enumerate(CUBES):
        target_bin = CUBE_TO_BIN[cube]
        bin_xy = env.bin_slot_pos[f"{target_bin}_middle"][:2]
        cur_done = ((np.linalg.norm(bin_xy - obs.objects[cube].xpos[:2])
                     < env.align_threshold)
                    or (target_bin in obs.objects[cube].contacts))
        if cur_done:
            sticky_progress.add(cube)
        if cube in sticky_progress:
            progress[i] = 1.0

    global_feat = np.concatenate([cube_onehot.reshape(-1), holding, progress])

    # Build per-agent features and masks
    masks = env.get_action_mask(obs)
    codec = ActionCodec(env.get_action_vocab())
    # Force-disable picking any cube already marked done. This protects
    # against physics drift causing a "done" cube to lose its contact flag
    # and become pickable again, which would let the policy waste steps
    # re-placing finished cubes.
    for ag, m in masks.items():
        for cube in sticky_progress:
            if cube in CUBES:
                oi = 1 + CUBES.index(cube)
                m["obj_mask"][oi] = False
                m["target_mask"][oi, :] = False
    per_agent = {}
    for ai, ag in enumerate(agents):
        ag_id = np.zeros(len(agents), dtype=np.float32)
        ag_id[ai] = 1.0
        per_agent[ag] = dict(
            obs=np.concatenate([global_feat, ag_id]),
            mask=codec.flat_mask(masks[ag]).astype(np.float32),
        )
    flat_masks = np.stack([per_agent[ag]["mask"] for ag in agents], axis=0)
    return dict(per_agent=per_agent, global_obs=global_feat, masks=flat_masks)


def rl_action_to_response(codec: ActionCodec,
                          joint_action: Dict[str, int]) -> str:
    """Format an EXECUTE response string the parser will accept."""
    lines = ["EXECUTE"]
    for ag, flat in joint_action.items():
        action_str = codec.to_str(int(flat))
        lines.append(f"NAME {ag} ACTION {action_str}")
    return "\n".join(lines) + "\n"
