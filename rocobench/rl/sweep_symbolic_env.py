"""Symbolic version of SweepTask for fast MAPPO training.

The real SweepTask (rocobench/envs/task_sweep.py) is a 2-agent (Alice with
dustpan, Bob with broom) cleanup task. The symbolic abstraction:

    cube state ∈ {LOOSE, IN_DUSTPAN, IN_TRASH}

Action vocab matches `SweepTask.get_action_vocab` exactly so the trained
policy is portable. Actions:

    Alice: WAIT(self), MOVE(cube), DUMP(trash_bin)
    Bob:   WAIT(self), MOVE(cube), SWEEP(cube)

Dynamics (one step ≈ "one round of joint moves"):
    - MOVE just sets that agent's "currently focused cube".
    - SWEEP succeeds when:
        Bob's action is SWEEP X
        Alice has previously MOVEd to X (her focus == X)
        cube X is still LOOSE
      -> X transitions LOOSE -> IN_DUSTPAN.
    - DUMP succeeds when Alice's action is DUMP and the dustpan has ≥1 cube.
      -> all IN_DUSTPAN cubes become IN_TRASH.
    - Anything else is a no-op (with optional invalid penalty if action
      outside the legality mask).

This captures the high-level cooperation requirement (you must MOVE together
before SWEEP) without modeling physics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .action_codec import ActionCodec


CUBES = ["red_cube", "green_cube", "blue_cube"]
VERBS = ["WAIT", "MOVE", "SWEEP", "DUMP"]
TARGETS = list(CUBES) + ["trash_bin", "self"]
LOOSE, IN_DUSTPAN, IN_TRASH = 0, 1, 2


@dataclass
class SweepState:
    cube_status: Dict[str, int]                      # cube -> {0,1,2}
    focus: Dict[str, Optional[str]]                  # agent -> cube it has MOVEd to
    last_action: Dict[str, int] = field(default_factory=dict)
    t: int = 0
    ever_dumped: int = 0


class SweepSymbolicEnv:
    AGENTS = ["Alice", "Bob"]
    CUBES = CUBES

    def __init__(self, max_steps: int = 30, gamma: float = 0.99,
                 seed: int = 0, randomize_init: bool = True,
                 use_mask: bool = True):
        self.max_steps = max_steps
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.randomize_init = randomize_init
        self.use_mask = use_mask
        self.codec = ActionCodec(self.get_action_vocab())
        self.state: Optional[SweepState] = None

    def get_action_vocab(self) -> Dict[str, List[str]]:
        # Verb-as-object lets us reuse ActionCodec. Index 0 must be WAIT.
        return dict(agents=list(self.AGENTS),
                    objects=list(VERBS),
                    targets=list(TARGETS))

    # ----- mask -----
    def get_action_mask(self) -> Dict[str, Dict[str, np.ndarray]]:
        s = self.state
        n_obj = len(VERBS)
        n_tgt = len(TARGETS)
        ret: Dict[str, Dict[str, np.ndarray]] = {}
        any_in_dustpan = any(v == IN_DUSTPAN for v in s.cube_status.values())
        for ag in self.AGENTS:
            obj_mask = np.zeros(n_obj, dtype=bool)
            tgt_mask = np.zeros((n_obj, n_tgt), dtype=bool)
            # WAIT(self) always legal
            obj_mask[VERBS.index("WAIT")] = True
            tgt_mask[VERBS.index("WAIT"), TARGETS.index("self")] = True

            # MOVE <cube> legal as long as cube is still LOOSE
            move_idx = VERBS.index("MOVE")
            for cube in CUBES:
                if s.cube_status[cube] == LOOSE:
                    obj_mask[move_idx] = True
                    tgt_mask[move_idx, TARGETS.index(cube)] = True

            # SWEEP <cube>: only Bob, and only when Alice already focused on
            # the same cube (i.e. she's in position with the dustpan).
            if ag == "Bob":
                sweep_idx = VERBS.index("SWEEP")
                alice_focus = s.focus.get("Alice")
                for cube in CUBES:
                    if s.cube_status[cube] != LOOSE:
                        continue
                    if alice_focus != cube:
                        continue
                    obj_mask[sweep_idx] = True
                    tgt_mask[sweep_idx, TARGETS.index(cube)] = True

            # DUMP -> trash_bin: only Alice, and only when dustpan non-empty
            if ag == "Alice" and any_in_dustpan:
                dump_idx = VERBS.index("DUMP")
                obj_mask[dump_idx] = True
                tgt_mask[dump_idx, TARGETS.index("trash_bin")] = True

            ret[ag] = dict(obj_mask=obj_mask, target_mask=tgt_mask)
        return ret

    def flat_masks(self) -> np.ndarray:
        m = self.get_action_mask()
        flat = np.stack([self.codec.flat_mask(m[ag]) for ag in self.AGENTS], axis=0)
        if not self.use_mask:
            flat = np.ones_like(flat, dtype=bool)
        return flat

    # ----- reset / step -----
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.state = SweepState(
            cube_status={c: LOOSE for c in CUBES},
            focus={ag: None for ag in self.AGENTS},
            last_action={ag: 0 for ag in self.AGENTS},
            t=0,
            ever_dumped=0,
        )
        return self.get_obs()

    def _phi(self) -> float:
        # Higher = closer to goal: each cube gives 0/1/2 progress points.
        return float(sum(self.state.cube_status.values()))

    def step(self, joint_action: Dict[str, int]) -> Tuple[Dict, float, bool, Dict]:
        s = self.state
        prev_phi = self._phi()
        prev_dumped = sum(1 for v in s.cube_status.values() if v == IN_TRASH)
        prev_in_dustpan = sum(1 for v in s.cube_status.values() if v == IN_DUSTPAN)

        masks = self.flat_masks()
        n_invalid = 0
        # Snap illegal actions to WAIT(self) and accumulate penalty.
        for ai, ag in enumerate(self.AGENTS):
            flat = int(joint_action[ag])
            if not masks[ai, flat]:
                n_invalid += 1
                joint_action[ag] = self.codec.encode(VERBS.index("WAIT"),
                                                    TARGETS.index("self"))

        # Apply MOVEs first (update focus)
        for ag in self.AGENTS:
            oi, ti = self.codec.decode(int(joint_action[ag]))
            verb = VERBS[oi]
            if verb == "MOVE":
                s.focus[ag] = TARGETS[ti]

        # Then resolve SWEEP (depends on Alice's *current* focus)
        bob_oi, bob_ti = self.codec.decode(int(joint_action["Bob"]))
        if VERBS[bob_oi] == "SWEEP":
            target_cube = TARGETS[bob_ti]
            if (s.focus["Alice"] == target_cube and
                    s.cube_status.get(target_cube) == LOOSE):
                s.cube_status[target_cube] = IN_DUSTPAN

        # Then DUMP (Alice)
        alice_oi, alice_ti = self.codec.decode(int(joint_action["Alice"]))
        if VERBS[alice_oi] == "DUMP" and TARGETS[alice_ti] == "trash_bin":
            for c in CUBES:
                if s.cube_status[c] == IN_DUSTPAN:
                    s.cube_status[c] = IN_TRASH

        s.t += 1
        for ag in self.AGENTS:
            s.last_action[ag] = int(joint_action[ag])

        # Reward
        cur_phi = self._phi()
        cur_dumped = sum(1 for v in s.cube_status.values() if v == IN_TRASH)
        cur_in_dustpan = sum(1 for v in s.cube_status.values() if v == IN_DUSTPAN)

        r_sweep = 2.0 * max(0, cur_in_dustpan + cur_dumped
                            - prev_in_dustpan - prev_dumped)
        r_dump = 5.0 * (cur_dumped - prev_dumped)
        all_done = cur_dumped == len(CUBES)
        r_done = 30.0 if all_done and prev_dumped < len(CUBES) else 0.0
        r_shape = self.gamma * cur_phi - prev_phi
        r_invalid = -2.0 * n_invalid
        r_step = -0.05
        r = r_sweep + r_dump + r_done + r_shape + r_invalid + r_step

        done = all_done or s.t >= self.max_steps
        info = dict(success=all_done, n_invalid=n_invalid,
                    breakdown=dict(r_sweep=r_sweep, r_dump=r_dump,
                                   r_done=r_done, r_shape=r_shape,
                                   r_invalid=r_invalid, r_step=r_step))
        return self.get_obs(), float(r), done, info

    # ----- observation -----
    def get_obs(self) -> Dict[str, Any]:
        s = self.state
        # cube status one-hot (3 cubes × 3 states)
        status = np.zeros((len(CUBES), 3), dtype=np.float32)
        for i, c in enumerate(CUBES):
            status[i, s.cube_status[c]] = 1.0
        # focus one-hot per agent (n_agents × (cubes+1))
        focus = np.zeros((len(self.AGENTS), len(CUBES) + 1), dtype=np.float32)
        for i, ag in enumerate(self.AGENTS):
            f = s.focus[ag]
            if f is None:
                focus[i, len(CUBES)] = 1.0
            else:
                focus[i, CUBES.index(f)] = 1.0
        global_feat = np.concatenate([status.reshape(-1), focus.reshape(-1)])

        masks = self.flat_masks()
        per_agent = {}
        for ai, ag in enumerate(self.AGENTS):
            ag_id = np.zeros(len(self.AGENTS), dtype=np.float32)
            ag_id[ai] = 1.0
            local = np.concatenate([global_feat, ag_id])
            per_agent[ag] = dict(obs=local, mask=masks[ai].astype(np.float32))
        return dict(per_agent=per_agent, global_obs=global_feat, masks=masks)

    @property
    def obs_dim(self) -> int:
        return len(CUBES) * 3 + len(self.AGENTS) * (len(CUBES) + 1) + len(self.AGENTS)

    @property
    def state_dim(self) -> int:
        return len(CUBES) * 3 + len(self.AGENTS) * (len(CUBES) + 1)

    @property
    def n_actions(self) -> int:
        return self.codec.flat_dim

    @property
    def n_agents(self) -> int:
        return len(self.AGENTS)
