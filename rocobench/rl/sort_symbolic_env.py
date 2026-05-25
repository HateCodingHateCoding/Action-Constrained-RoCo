"""Lightweight symbolic version of SortOneBlockTask for RL training.

Why a symbolic env? The high-level RL policy operates on discrete actions
(WAIT / PICK X PLACE Y). The MuJoCo physics + RRT pipeline is slow and
not differentiable, so for *training* we use a symbolic simulator that
replicates only the parts of the world the high-level policy cares about:

    - 7 panels at fixed x positions
    - 3 cubes, each on some panel
    - 3 robots with fixed reach intervals over panel x positions
    - PICK X PLACE Y succeeds iff X is reachable, Y is reachable,
      hands are empty, and (X, Y) satisfies the cube-target rule.

After training, the same policy can be plugged into the real env via the
`ActionCodec` -- the action interface is identical.

Coordinates and reach ranges below are taken from
`task_sort.SortOneBlockTask.get_robot_reach_range` and the panel
definitions in `task_sort.xml`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .action_codec import ActionCodec


# Approximate panel x positions along the assembly line (panel1..panel7).
# Pulled to match the sorting layout used in the real task; only relative
# ordering matters for the symbolic dynamics.
_PANEL_X = {
    "panel1": -1.2,
    "panel2": -0.8,
    "panel3": -0.4,
    "panel4": 0.0,
    "panel5": 0.4,
    "panel6": 0.8,
    "panel7": 1.2,
}

# x-axis reach intervals per agent, derived from get_robot_reach_range.
_REACH_X = {
    "Alice": (-1.4, -0.1),  # panels 1..3
    "Bob":   (-0.7, 0.7),   # panels 3..5
    "Chad":  (0.2, 1.5),    # panels 5..7
}

CUBES = ["blue_square", "pink_polygon", "yellow_trapezoid"]
CUBE_TO_BIN = {"blue_square": "panel2", "pink_polygon": "panel4",
               "yellow_trapezoid": "panel6"}
CUBE_OWNER = {"blue_square": "Alice", "pink_polygon": "Bob",
              "yellow_trapezoid": "Chad"}
RELAY_PANELS = {"panel3", "panel5"}


@dataclass
class SortState:
    cube_panel: Dict[str, str]            # cube -> panel name
    holding: Dict[str, Optional[str]]     # agent -> cube being held (or None)
    last_action: Dict[str, int] = field(default_factory=dict)
    t: int = 0
    ever_done: set = field(default_factory=set)  # cubes that have reached goal at least once


class SortSymbolicEnv:
    """Multi-agent symbolic environment with the same vocab/mask as task_sort."""

    PANELS = [f"panel{i}" for i in range(1, 8)]
    AGENTS = ["Alice", "Bob", "Chad"]

    def __init__(self, max_steps: int = 30, gamma: float = 0.99,
                 seed: int = 0, randomize_init: bool = True,
                 use_mask: bool = True,
                 use_shaping: bool = True,
                 use_handoff: bool = True):
        self.max_steps = max_steps
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.randomize_init = randomize_init
        self.use_mask = use_mask
        self.use_shaping = use_shaping
        self.use_handoff = use_handoff
        self.codec = ActionCodec(self.get_action_vocab())
        self.state: Optional[SortState] = None

    # --------------------------------------------------------------
    # Vocab / mask -- identical structure to the real task hooks.
    # --------------------------------------------------------------
    def get_action_vocab(self) -> Dict[str, List[str]]:
        return dict(agents=list(self.AGENTS),
                    objects=["WAIT"] + list(CUBES),
                    targets=list(self.PANELS))

    def _agent_can_reach_panel(self, agent: str, panel: str) -> bool:
        lo, hi = _REACH_X[agent]
        x = _PANEL_X[panel]
        return lo <= x <= hi

    def get_action_mask(self) -> Dict[str, Dict[str, np.ndarray]]:
        s = self.state
        n_obj = 1 + len(CUBES)
        n_tgt = len(self.PANELS)
        ret: Dict[str, Dict[str, np.ndarray]] = {}
        for ag in self.AGENTS:
            obj_mask = np.zeros(n_obj, dtype=bool)
            tgt_mask = np.zeros((n_obj, n_tgt), dtype=bool)
            obj_mask[0] = True               # WAIT
            tgt_mask[0, 0] = True            # canonical WAIT slot
            if s.holding[ag] is None:
                for oi, cube in enumerate(CUBES, start=1):
                    panel = s.cube_panel[cube]
                    if not self._agent_can_reach_panel(ag, panel):
                        continue
                    # don't disturb cubes already at their goal
                    if self._cube_done(cube):
                        continue
                    # at least one place target reachable?
                    correct = CUBE_TO_BIN[cube]
                    allowed = {correct} | RELAY_PANELS
                    feasible = False
                    for ti, tgt in enumerate(self.PANELS):
                        if tgt in allowed and self._agent_can_reach_panel(ag, tgt):
                            tgt_mask[oi, ti] = True
                            feasible = True
                    if feasible:
                        obj_mask[oi] = True
            ret[ag] = dict(obj_mask=obj_mask, target_mask=tgt_mask)
        return ret

    def flat_masks(self) -> np.ndarray:
        m = self.get_action_mask()
        flat = np.stack([self.codec.flat_mask(m[ag]) for ag in self.AGENTS], axis=0)
        if not self.use_mask:
            # ablation: every action is "legal" -> the policy must learn the
            # constraints from rewards alone
            flat = np.ones_like(flat, dtype=bool)
        return flat

    # --------------------------------------------------------------
    # Reset / step.
    # --------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        if self.randomize_init:
            # Match the real task: blue_square far from Alice, polygon far
            # from Bob, trapezoid far from Chad. This makes handoff matter.
            far_for = {
                "blue_square":      ["panel4", "panel5", "panel6", "panel7"],
                "pink_polygon":     ["panel1", "panel2", "panel6", "panel7"],
                "yellow_trapezoid": ["panel1", "panel2", "panel3", "panel4"],
            }
            occupied = set()
            placement = {}
            for cube in CUBES:
                choices = [p for p in far_for[cube] if p not in occupied]
                pick = self.rng.choice(choices)
                placement[cube] = str(pick)
                occupied.add(pick)
        else:
            placement = {"blue_square": "panel6", "pink_polygon": "panel1",
                         "yellow_trapezoid": "panel2"}
        self.state = SortState(
            cube_panel=placement,
            holding={ag: None for ag in self.AGENTS},
            last_action={ag: 0 for ag in self.AGENTS},
            t=0,
            ever_done=set(),
        )
        return self.get_obs()

    def _cube_done(self, cube: str) -> bool:
        return self.state.cube_panel.get(cube) == CUBE_TO_BIN[cube]

    def _all_done(self) -> bool:
        return all(self._cube_done(c) for c in CUBES)

    def _phi(self) -> float:
        """Potential = -sum(distance from cube to its bin) on x-axis."""
        s = self.state
        dist = 0.0
        for cube, panel in s.cube_panel.items():
            target = CUBE_TO_BIN[cube]
            dist += abs(_PANEL_X[panel] - _PANEL_X[target])
        return -dist

    def step(self, joint_action: Dict[str, int]) -> Tuple[Dict, float, bool, Dict]:
        """joint_action: {agent_name: flat_action_id}"""
        s = self.state
        prev_cube_panel = dict(s.cube_panel)
        prev_phi = self._phi()
        prev_owner_reach = {
            cube: self._agent_can_reach_panel(CUBE_OWNER[cube], s.cube_panel[cube])
            for cube in CUBES
        }
        prev_done = {cube: self._cube_done(cube) for cube in CUBES}

        masks = self.flat_masks()
        n_invalid = 0

        # Resolve actions in fixed order. Conflicts (two agents picking the
        # same cube) are broken by agent index order; the loser becomes
        # invalid (penalty applies).
        chosen_picks: Dict[str, str] = {}  # cube -> agent who claimed it
        for ai, ag in enumerate(self.AGENTS):
            flat = int(joint_action[ag])
            if not masks[ai, flat]:
                n_invalid += 1
                joint_action[ag] = 0  # force WAIT
                flat = 0
            oi, ti = self.codec.decode(flat)
            if oi == 0:
                continue
            cube = CUBES[oi - 1]
            if cube in chosen_picks:
                n_invalid += 1
                joint_action[ag] = 0
                continue
            chosen_picks[cube] = ag

        # Apply PICK·PLACE: instantaneous teleport in the symbolic world.
        for ai, ag in enumerate(self.AGENTS):
            flat = int(joint_action[ag])
            oi, ti = self.codec.decode(flat)
            if oi == 0:
                continue
            cube = CUBES[oi - 1]
            target_panel = self.PANELS[ti]
            s.cube_panel[cube] = target_panel
            s.last_action[ag] = flat

        s.t += 1

        # Reward shaping (mirrors task_sort.get_rl_reward).
        r = 0.0
        breakdown = {}
        # Once a cube reaches its goal we record it; r_goal pays only once.
        r_goal = 0.0
        for cube in CUBES:
            if self._cube_done(cube) and (cube not in s.ever_done):
                r_goal += 10.0
                s.ever_done.add(cube)
        breakdown["r_goal"] = r_goal
        r += r_goal

        all_done_now = self._all_done()
        r_done = 50.0 if (all_done_now and not all(prev_done.values())) else 0.0
        breakdown["r_done"] = r_done
        r += r_done

        cur_phi = self._phi()
        r_shape = self.gamma * cur_phi - prev_phi
        if not self.use_shaping:
            r_shape = 0.0
        breakdown["r_shape"] = r_shape
        r += r_shape

        r_handoff = 0.0
        if self.use_handoff:
            for cube in CUBES:
                if self._cube_done(cube):
                    continue
                owner = CUBE_OWNER[cube]
                cur_reach = self._agent_can_reach_panel(owner, s.cube_panel[cube])
                if (not prev_owner_reach[cube]) and cur_reach:
                    r_handoff += 2.0
        breakdown["r_handoff"] = r_handoff
        r += r_handoff

        breakdown["r_invalid"] = -2.0 * n_invalid
        r += breakdown["r_invalid"]

        all_wait = all(int(joint_action[ag]) == 0 for ag in self.AGENTS)
        if all_wait:
            r += -0.5
            breakdown["r_all_wait"] = -0.5

        r += -0.05
        breakdown["r_step"] = -0.05

        done = all_done_now or s.t >= self.max_steps
        info = dict(breakdown=breakdown, success=all_done_now,
                    n_invalid=n_invalid)
        return self.get_obs(), float(r), done, info

    # --------------------------------------------------------------
    # Observations: per-agent local + shared global.
    # --------------------------------------------------------------
    def get_obs(self) -> Dict[str, Any]:
        s = self.state
        # Global feature vector: cube positions (one-hot panel) + holding flags.
        cube_onehot = np.zeros((len(CUBES), len(self.PANELS)), dtype=np.float32)
        for i, cube in enumerate(CUBES):
            cube_onehot[i, self.PANELS.index(s.cube_panel[cube])] = 1.0
        holding = np.array(
            [1.0 if s.holding[ag] is not None else 0.0 for ag in self.AGENTS],
            dtype=np.float32,
        )
        progress = np.array(
            [1.0 if self._cube_done(c) else 0.0 for c in CUBES], dtype=np.float32
        )
        global_feat = np.concatenate([cube_onehot.reshape(-1), holding, progress])

        masks = self.flat_masks()
        per_agent = {}
        for ai, ag in enumerate(self.AGENTS):
            ag_id = np.zeros(len(self.AGENTS), dtype=np.float32)
            ag_id[ai] = 1.0
            local = np.concatenate([global_feat, ag_id])
            per_agent[ag] = dict(obs=local, mask=masks[ai].astype(np.float32))
        return dict(per_agent=per_agent,
                    global_obs=np.concatenate([global_feat]),
                    masks=masks)

    @property
    def obs_dim(self) -> int:
        return len(CUBES) * len(self.PANELS) + len(self.AGENTS) + len(CUBES) + len(self.AGENTS)

    @property
    def state_dim(self) -> int:
        return len(CUBES) * len(self.PANELS) + len(self.AGENTS) + len(CUBES)

    @property
    def n_actions(self) -> int:
        return self.codec.flat_dim

    @property
    def n_agents(self) -> int:
        return len(self.AGENTS)
