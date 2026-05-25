"""Mask-aware vs vanilla LLM ablation.

Compares two variants of the LLM+RL hybrid policy:
  - vanilla:    LLM gets a generic task description, often proposes illegal actions
  - mask-aware: LLM is shown the legal action set explicitly per turn

Metrics:
  - success rate (RL still cleans up either way; both should be high)
  - LLM illegal-action rate (how often does the LLM hallucinate?)
  - candidate-vs-legal intersection ratio

The headline number is "illegal-action rate" -- mask-aware should be
much lower, demonstrating that the mask provides usable structure even
to a frozen LLM.
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from rocobench.rl import (
    ActionCodec, MAPPOAgent, SortSymbolicEnv, LLMRLHybridPolicy,
)
from rocobench.rl.mappo import MAPPOConfig
from prompting.llm_api import chat_completion


def evaluate(env, policy, n_episodes: int) -> dict:
    successes, returns, lens, invalids = [], [], [], []
    n_llm_illegal_total, n_steps_total = 0, 0
    for ep in range(n_episodes):
        obs = env.reset()
        ep_ret, ep_len, ep_inv = 0.0, 0, 0
        done = False
        info = {}
        while not done:
            obs_arr = np.stack([obs["per_agent"][ag]["obs"] for ag in env.AGENTS], axis=0)
            mask_arr = np.stack([obs["per_agent"][ag]["mask"] for ag in env.AGENTS], axis=0)
            scene = "; ".join(
                f"{c} on {env.state.cube_panel[c]}" for c in env.state.cube_panel
            )
            actions, _ = policy.act(obs_arr, mask_arr, scene_desc=scene)
            joint = {ag: int(actions[i]) for i, ag in enumerate(env.AGENTS)}
            obs, r, done, info = env.step(joint)
            ep_ret += r
            ep_len += 1
            ep_inv += int(info.get("n_invalid", 0))
            n_llm_illegal_total += getattr(policy, "last_n_illegal", 0)
            n_steps_total += 1
            if ep_len >= env.max_steps:
                break
        successes.append(1.0 if info.get("success") else 0.0)
        returns.append(ep_ret)
        lens.append(ep_len)
        invalids.append(ep_inv)
    return dict(success=float(np.mean(successes)),
                steps=float(np.mean(lens)),
                ret=float(np.mean(returns)),
                llm_illegal_per_step=(n_llm_illegal_total / max(1, n_steps_total)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/sort_mappo_smoke.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--llm-model", default="glm-4-flash")
    args = p.parse_args()

    env = SortSymbolicEnv(seed=2024)
    codec = ActionCodec(env.get_action_vocab())
    cfg = MAPPOConfig(device="cpu")
    rl = MAPPOAgent(env.obs_dim, env.state_dim, env.n_actions, env.n_agents, cfg=cfg)
    rl.load(args.ckpt)
    print(f"Loaded MAPPO from {args.ckpt}")

    def _llm(prompt: str) -> str:
        content, _ = chat_completion(
            model=args.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.0, max_retries=3,
        )
        return content or ""

    rows = []
    for label, show_mask in [("vanilla", False), ("mask-aware", True)]:
        print(f"\n=== {label} (show_mask_to_llm={show_mask}) ===")
        agent = LLMRLHybridPolicy(rl, codec, llm_call=_llm,
                                  show_mask_to_llm=show_mask)
        t0 = time.time()
        m = evaluate(env, agent, args.episodes)
        m["time"] = time.time() - t0
        print(f"  succ={m['success']:.2%}  steps={m['steps']:.2f}  "
              f"ret={m['ret']:.2f}  illegal/step={m['llm_illegal_per_step']:.3f}  "
              f"time={m['time']:.1f}s")
        rows.append((label, m))

    print("\n=== Mask-aware ablation summary ===")
    print(f"{'variant':14s} | {'success':>8s} | {'steps':>6s} | {'illegal/step':>12s}")
    print("-" * 50)
    for label, m in rows:
        print(f"{label:14s} | {m['success']:>8.2%} | {m['steps']:>6.2f} | "
              f"{m['llm_illegal_per_step']:>12.3f}")


if __name__ == "__main__":
    main()
