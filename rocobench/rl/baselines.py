"""Baseline policies for the symbolic Sort env. Used in the comparison
table against MAPPO.

All policies expose the same interface as MAPPOAgent:
    act(obs_batch, mask_batch) -> (actions, log_probs)

`log_probs` is returned for compatibility but ignored by the eval harness.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from .action_codec import ActionCodec
from .sort_symbolic_env import (
    SortSymbolicEnv, CUBES, CUBE_TO_BIN, CUBE_OWNER, _PANEL_X, _REACH_X,
    RELAY_PANELS,
)


class RandomMaskedPolicy:
    """Uniformly sample from the legal action set."""

    def __init__(self, env: SortSymbolicEnv, seed: int = 0):
        self.env = env
        self.rng = np.random.RandomState(seed)

    def act(self, obs_batch, mask_batch) -> Tuple[np.ndarray, np.ndarray]:
        actions = np.zeros(mask_batch.shape[0], dtype=np.int64)
        for i, m in enumerate(mask_batch):
            legal = np.where(m > 0.5)[0]
            actions[i] = int(self.rng.choice(legal))
        return actions, np.zeros_like(actions, dtype=np.float32)

    def value(self, _state):  # for compatibility
        return 0.0


class RandomNoMaskPolicy:
    """Uniformly sample from the FULL action space, ignoring legality.

    Illegal actions get clipped back to WAIT inside the env (and trigger
    the invalid-action penalty). Used to demonstrate why the mask matters.
    """

    def __init__(self, env: SortSymbolicEnv, seed: int = 0):
        self.env = env
        self.rng = np.random.RandomState(seed)
        self.n_actions = env.n_actions

    def act(self, obs_batch, mask_batch) -> Tuple[np.ndarray, np.ndarray]:
        actions = self.rng.randint(0, self.n_actions, size=mask_batch.shape[0]).astype(np.int64)
        return actions, np.zeros_like(actions, dtype=np.float32)

    def value(self, _state):
        return 0.0


class ScriptedHeuristicPolicy:
    """A simple hand-crafted "good enough" baseline.

    Rule per agent:
      1. If my own cube is reachable, place it on its target panel.
      2. Else if any other cube is reachable AND moving it to a relay panel
         (panel3/panel5) brings it into the owner's reach, do that.
      3. Else WAIT.
    """

    def __init__(self, env: SortSymbolicEnv):
        self.env = env
        self.codec = env.codec
        self.agents = env.AGENTS

    def _reachable(self, agent: str, panel: str) -> bool:
        lo, hi = _REACH_X[agent]
        return lo <= _PANEL_X[panel] <= hi

    def _choose(self, agent: str) -> int:
        s = self.env.state
        # 1) own cube available
        own_cube = next(c for c, ow in CUBE_OWNER.items() if ow == agent)
        if (s.cube_panel[own_cube] != CUBE_TO_BIN[own_cube]
                and self._reachable(agent, s.cube_panel[own_cube])
                and self._reachable(agent, CUBE_TO_BIN[own_cube])):
            oi = 1 + CUBES.index(own_cube)
            ti = self.env.PANELS.index(CUBE_TO_BIN[own_cube])
            return self.codec.encode(oi, ti)

        # 2) help someone else: move their cube to a relay panel I can reach
        for cube in CUBES:
            owner = CUBE_OWNER[cube]
            if owner == agent:
                continue
            if s.cube_panel[cube] == CUBE_TO_BIN[cube]:
                continue
            if not self._reachable(agent, s.cube_panel[cube]):
                continue
            if self._reachable(owner, s.cube_panel[cube]):
                continue  # owner can already reach it; no help needed
            for relay in RELAY_PANELS:
                if not self._reachable(agent, relay):
                    continue
                if not self._reachable(owner, relay):
                    continue
                oi = 1 + CUBES.index(cube)
                ti = self.env.PANELS.index(relay)
                return self.codec.encode(oi, ti)
        return 0  # WAIT

    def act(self, obs_batch, mask_batch) -> Tuple[np.ndarray, np.ndarray]:
        actions = np.array([self._choose(ag) for ag in self.agents], dtype=np.int64)
        # respect masks; if heuristic picked an illegal action (e.g. someone
        # already grabbed it), fall back to WAIT
        for i, a in enumerate(actions):
            if not bool(mask_batch[i, a] > 0.5):
                actions[i] = 0
        return actions, np.zeros_like(actions, dtype=np.float32)

    def value(self, _state):
        return 0.0
