"""LLM + RL hybrid policy.

The LLM proposes one or more candidate actions per agent. The RL policy
chooses among them, restricted to the candidates and the legal action
mask. This is the project's core "make the open question into multiple
choice" mechanism.

Workflow per step:
    1. Build a one-shot prompt asking the LLM for K=1..3 candidates per
       agent, in the same `EXECUTE\nNAME ... ACTION ...` format the
       parser already understands.
    2. Parse those candidates into per-agent action ids via ActionCodec.
    3. Build a candidate mask: ones at LLM-suggested positions intersected
       with the legality mask.
    4. If the candidate mask still has ≥1 legal entry per agent, use it;
       otherwise fall back to the legality mask alone.
    5. Sample from the masked logits as usual.

If the LLM API is unavailable the policy gracefully degrades to RL-only.
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .action_codec import ActionCodec
from .mappo import MAPPOAgent
from .sort_symbolic_env import CUBES, CUBE_TO_BIN


_TASK_HINT = """You are coordinating 3 robots sorting 3 cubes into target panels.

Reach (panels each robot can reach):
- Alice: panel1, panel2, panel3
- Bob:   panel3, panel4, panel5
- Chad:  panel5, panel6, panel7

Goals (each cube belongs to one robot):
- blue_square     -> panel2 (Alice's job)
- pink_polygon    -> panel4 (Bob's job)
- yellow_trapezoid-> panel6 (Chad's job)

Cubes can be relayed via panel3 / panel5 if a cube is out of its owner's reach.

Output exactly one EXECUTE block with one ACTION per robot. Each ACTION is
either "WAIT" or "PICK <cube> PLACE <panel>". Do NOT propose all WAIT unless
every cube is already at its goal panel.

Format example:
EXECUTE
NAME Alice ACTION PICK yellow_trapezoid PLACE panel3
NAME Bob ACTION PICK blue_square PLACE panel3
NAME Chad ACTION PICK pink_polygon PLACE panel5
"""


_RE_LINE = re.compile(
    r"NAME\s+(Alice|Bob|Chad)\s+ACTION\s+(WAIT|PICK\s+(\w+)\s+PLACE\s+(panel\d))",
    re.IGNORECASE,
)


def parse_llm_proposals(response: str, codec: ActionCodec) -> Dict[str, List[int]]:
    """Extract per-agent suggested flat action ids from an EXECUTE-format
    LLM response. Robust to extra commentary the LLM might include.
    """
    out: Dict[str, List[int]] = {ag: [] for ag in ["Alice", "Bob", "Chad"]}
    if not response:
        return out
    text = response.split("EXECUTE", 1)[-1] if "EXECUTE" in response else response
    for ag, full, cube, panel in _RE_LINE.findall(text):
        ag = ag.capitalize()
        full = full.upper()
        if full == "WAIT":
            out[ag].append(0)
            continue
        cube_l = cube.lower()
        panel_l = panel.lower()
        if cube_l not in codec.objects:
            continue
        if panel_l not in codec.targets:
            continue
        oi = codec.objects.index(cube_l)
        ti = codec.targets.index(panel_l)
        flat = codec.encode(oi, ti)
        out[ag].append(flat)
    return out


class LLMRLHybridPolicy:
    """Combines a trained MAPPO agent with on-the-fly LLM suggestions.

    Args:
        rl_agent: a trained MAPPOAgent.
        codec: matching ActionCodec.
        llm_call: callable(prompt:str)->str returning the LLM response,
            or None to disable LLM querying (degrades to RL-only).
        agents: ordered agent names.
        prefer_llm: if True, restrict RL to LLM candidates whenever the
            intersection with the legal mask is non-empty. If False, only
            use the LLM's candidates as a soft prior (logit boost).
    """
    def __init__(self, rl_agent: MAPPOAgent, codec: ActionCodec,
                 llm_call=None,
                 agents: Optional[List[str]] = None,
                 prefer_llm: bool = True,
                 show_mask_to_llm: bool = False):
        self.rl = rl_agent
        self.codec = codec
        self.llm_call = llm_call
        self.agents = agents or ["Alice", "Bob", "Chad"]
        self.prefer_llm = prefer_llm
        self.show_mask_to_llm = show_mask_to_llm
        self.last_llm_response: Optional[str] = None
        self.last_candidates: Dict[str, List[int]] = {}

    def _build_prompt(self, scene_desc: str,
                      mask_batch: Optional[np.ndarray] = None) -> str:
        prompt = f"{_TASK_HINT}\n\n[Scene]\n{scene_desc}\n"
        if self.show_mask_to_llm and mask_batch is not None:
            prompt += "\n[Legal actions for each robot this turn]\n"
            for i, ag in enumerate(self.agents):
                legal_ids = np.where(mask_batch[i] > 0.5)[0].tolist()
                legal_strs = [self.codec.to_str(int(a)) for a in legal_ids]
                prompt += f"  {ag}: {legal_strs}\n"
            prompt += ("\nPick exactly one ACTION per robot from the list above. "
                       "Do NOT propose any action outside that list.\n")
        prompt += "\nProvide your single best joint plan now."
        return prompt

    def act(self, obs_batch: np.ndarray, mask_batch: np.ndarray,
            scene_desc: str = "") -> Tuple[np.ndarray, np.ndarray]:
        # 1. Try the LLM
        candidates: Dict[str, List[int]] = {ag: [] for ag in self.agents}
        if self.llm_call is not None:
            try:
                prompt = self._build_prompt(scene_desc, mask_batch=mask_batch)
                response = self.llm_call(prompt)
                self.last_llm_response = response
                candidates = parse_llm_proposals(response, self.codec)
            except Exception as e:
                print(f"  [hybrid] LLM call failed: {e}; falling back to RL-only")
                candidates = {ag: [] for ag in self.agents}
        self.last_candidates = candidates

        # Track how many LLM-proposed actions are illegal (for ablation reporting).
        n_illegal = 0
        for i, ag in enumerate(self.agents):
            for c in candidates.get(ag, []):
                if mask_batch[i, c] <= 0.5:
                    n_illegal += 1
        self.last_n_illegal = n_illegal

        # 2. Combine with legality mask.
        new_mask = mask_batch.copy()
        # Sanity check: if the LLM proposed all-WAIT, that's a hallucination
        # signal -- ignore it. Real progress requires at least one robot to act.
        all_wait_llm = (self.prefer_llm and candidates and
                        all(set(cs) == {0} for cs in candidates.values() if cs))
        if all_wait_llm:
            print("  [hybrid] LLM proposed all-WAIT; ignoring as likely hallucination")
            candidates = {ag: [] for ag in self.agents}
            self.last_candidates = candidates

        if self.prefer_llm:
            for i, ag in enumerate(self.agents):
                cand = candidates.get(ag, [])
                if not cand:
                    continue
                cand_mask = np.zeros_like(mask_batch[i], dtype=bool)
                for flat in cand:
                    cand_mask[flat] = True
                # Intersect with legal mask; if intersection is non-empty,
                # narrow the search; otherwise leave as legal-only.
                inter = cand_mask & (mask_batch[i] > 0.5)
                if inter.any():
                    new_mask[i] = inter.astype(np.float32)
        # 3. Sample from the (possibly narrowed) mask via the RL agent.
        actions, logp = self.rl.act(obs_batch, new_mask)
        return actions, logp

    def value(self, state):
        return self.rl.value(state)
