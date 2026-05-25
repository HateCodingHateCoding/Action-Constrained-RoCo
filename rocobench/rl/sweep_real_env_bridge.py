"""Bridge for the real Sweep MuJoCo env <-> symbolic policy.

The trained MAPPO policy reads obs in the symbolic format
(SweepSymbolicEnv.get_obs). For real-env rollout we need to:
  1. Read the real EnvState to recover each cube's status (LOOSE / IN_DUSTPAN
     / IN_TRASH).
  2. Track each agent's "focus" cube ourselves -- there's no clean signal
     for that in the real env; we update it from the runner whenever an
     agent picks a MOVE action.
  3. Repackage as the symbolic obs the policy expects.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from rocobench.envs import MujocoSimEnv, EnvState
from rocobench.rl.action_codec import ActionCodec
from rocobench.rl.sweep_symbolic_env import (
    CUBES, VERBS, TARGETS, LOOSE, IN_DUSTPAN, IN_TRASH,
)


def cube_status(env: MujocoSimEnv, obs: EnvState, cube: str) -> int:
    """Infer LOOSE / IN_DUSTPAN / IN_TRASH from the real env."""
    bin_xy = env.physics.data.body("trash_bin_bottom").xpos
    cube_xy = obs.objects[cube].xpos if cube in obs.objects else \
              env.physics.data.body(cube).xpos
    if float(np.linalg.norm(bin_xy - cube_xy)) < 0.2:
        return IN_TRASH
    contact_dict = env.get_contact()
    if "dustpan" in contact_dict.get(cube, set()):
        return IN_DUSTPAN
    return LOOSE


def obs_to_sweep_features(env: MujocoSimEnv, obs: EnvState,
                          focus: Dict[str, Optional[str]],
                          sticky_swept: Optional[set] = None
                          ) -> Dict[str, Any]:
    """Match SweepSymbolicEnv.get_obs layout exactly.

    Args:
        sticky_swept: optional set of cube names that the runner has already
            executed a SWEEP for. These are forced to IN_DUSTPAN regardless
            of physics, since the real broom contact may not actually move
            the cube into the dustpan body (sim-to-real gap).
    """
    sticky_swept = sticky_swept or set()
    agents = ["Alice", "Bob"]
    status = np.zeros((len(CUBES), 3), dtype=np.float32)
    statuses: Dict[str, int] = {}
    for i, c in enumerate(CUBES):
        s = cube_status(env, obs, c)
        if s == LOOSE and c in sticky_swept:
            s = IN_DUSTPAN
        statuses[c] = s
        status[i, s] = 1.0
    focus_arr = np.zeros((len(agents), len(CUBES) + 1), dtype=np.float32)
    for i, ag in enumerate(agents):
        f = focus.get(ag)
        if f is None or f not in CUBES:
            focus_arr[i, len(CUBES)] = 1.0
        else:
            focus_arr[i, CUBES.index(f)] = 1.0
    global_feat = np.concatenate([status.reshape(-1), focus_arr.reshape(-1)])

    # Re-derive the legality mask straight from the real env (we already
    # wrote SweepTask.get_action_mask in task_sweep.py).
    masks_named = env.get_action_mask(obs)
    codec = ActionCodec(env.get_action_vocab())
    # If a cube is sticky-swept, prevent re-sweep / re-move to it.
    if sticky_swept:
        sweep_idx = codec.objects.index("SWEEP") if "SWEEP" in codec.objects else None
        move_idx = codec.objects.index("MOVE") if "MOVE" in codec.objects else None
        for ag, m in masks_named.items():
            for cube in sticky_swept:
                if cube not in codec.targets:
                    continue
                ti = codec.targets.index(cube)
                if sweep_idx is not None:
                    m["target_mask"][sweep_idx, ti] = False
                if move_idx is not None:
                    m["target_mask"][move_idx, ti] = False
    flat_masks = np.stack(
        [codec.flat_mask(masks_named[ag]) for ag in agents], axis=0
    )

    per_agent: Dict[str, Dict[str, np.ndarray]] = {}
    for ai, ag in enumerate(agents):
        ag_id = np.zeros(len(agents), dtype=np.float32)
        ag_id[ai] = 1.0
        local = np.concatenate([global_feat, ag_id])
        per_agent[ag] = dict(obs=local, mask=flat_masks[ai].astype(np.float32))

    return dict(per_agent=per_agent, global_obs=global_feat,
                masks=flat_masks, statuses=statuses)


def sweep_action_to_response(codec: ActionCodec,
                             joint_action: Dict[str, int]) -> str:
    """EXECUTE-string formatter consistent with the existing parser."""
    lines = ["EXECUTE"]
    for ag, flat in joint_action.items():
        action_str = codec.to_str(int(flat))
        lines.append(f"NAME {ag} ACTION {action_str}")
    return "\n".join(lines) + "\n"
