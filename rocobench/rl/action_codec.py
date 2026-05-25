"""Discrete factorized action codec used by the RL high-level policy.

Action layout per agent:
    a = (obj_idx, target_idx)
    obj_idx == 0  -> WAIT (target_idx ignored)
    obj_idx >= 1  -> PICK objects[obj_idx] PLACE targets[target_idx]

We expose:
- encode/decode helpers between (obj_idx, target_idx) and a single flat
  action id in [0, n_obj * n_tgt). Index 0 is reserved for WAIT.
- a `flat_mask(obs)` helper that turns the per-agent dict masks returned by
  task envs into a single (n_obj * n_tgt,) boolean vector, the form most
  training code wants.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np


class ActionCodec:
    def __init__(self, vocab: Dict[str, List[str]]):
        self.agents: List[str] = list(vocab["agents"])
        self.objects: List[str] = list(vocab["objects"])  # objects[0] == "WAIT"
        self.targets: List[str] = list(vocab["targets"])
        assert self.objects[0].upper() == "WAIT", \
            "objects[0] must be 'WAIT' sentinel"
        # Action format heuristic: if "objects" looks like verbs (WAIT, MOVE,
        # SWEEP, DUMP), emit "VERB target"; otherwise treat objects[1:] as
        # things to pick (sort task) and emit "PICK obj PLACE target".
        upper = {o.upper() for o in self.objects[1:]}
        self.verb_mode = bool(upper & {"MOVE", "SWEEP", "DUMP", "OPEN", "PUSH"})
        self.n_obj = len(self.objects)
        self.n_tgt = len(self.targets)
        self.flat_dim = self.n_obj * self.n_tgt

    # ---------- encode / decode ----------
    def encode(self, obj_idx: int, target_idx: int) -> int:
        return int(obj_idx) * self.n_tgt + int(target_idx)

    def decode(self, flat_id: int) -> Tuple[int, int]:
        flat_id = int(flat_id)
        return flat_id // self.n_tgt, flat_id % self.n_tgt

    def to_str(self, flat_id: int) -> str:
        oi, ti = self.decode(flat_id)
        if oi == 0:
            return "WAIT"
        if self.verb_mode:
            verb = self.objects[oi]
            tgt = self.targets[ti]
            if verb.upper() == "DUMP":
                # DUMP doesn't need an explicit target token in the parser,
                # but we keep it for traceability.
                return f"DUMP {tgt}"
            return f"{verb} {tgt}"
        return f"PICK {self.objects[oi]} PLACE {self.targets[ti]}"

    # ---------- mask conversion ----------
    def flat_mask(self, agent_mask: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert {obj_mask, target_mask} into a flat (n_obj*n_tgt,) bool mask.

        The WAIT row (obj_idx == 0) collapses to a single legal entry at
        flat id 0; the other (n_tgt - 1) WAIT cells are forced illegal so
        the policy can't waste probability mass on duplicates of WAIT.
        """
        obj_mask = np.asarray(agent_mask["obj_mask"], dtype=bool)
        tgt_mask = np.asarray(agent_mask["target_mask"], dtype=bool)
        assert obj_mask.shape == (self.n_obj,)
        assert tgt_mask.shape == (self.n_obj, self.n_tgt)

        flat = np.zeros((self.n_obj, self.n_tgt), dtype=bool)
        if obj_mask[0]:
            flat[0, 0] = True
        for oi in range(1, self.n_obj):
            if not obj_mask[oi]:
                continue
            flat[oi] = tgt_mask[oi]
        return flat.reshape(-1)
