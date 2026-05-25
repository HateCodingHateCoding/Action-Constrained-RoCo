"""One-shot benchmark for the project's comparison + ablation experiments.

Methods evaluated (all on the symbolic Sort env):

  - random_no_mask     | random policy, no action mask, no constraints
  - random_masked      | random over the legal action set
  - scripted           | hand-crafted heuristic (close to optimal)
  - mappo_full         | our method: mask + shaping + handoff
  - mappo_no_mask      | ablation: drop the action mask
  - mappo_no_shaping   | ablation: drop potential-based shaping
  - mappo_no_handoff   | ablation: drop the handoff bonus

For each method we report:
    success_rate, avg_steps, avg_invalids, avg_return

Usage:
    python benchmark_rl.py
    python benchmark_rl.py --eval-episodes 200 --train-steps 30000
"""
from __future__ import annotations
import argparse
import time
from typing import Callable, Dict
import numpy as np

from rocobench.rl import (
    SortSymbolicEnv, MAPPOAgent, train_mappo,
    RandomMaskedPolicy, RandomNoMaskPolicy, ScriptedHeuristicPolicy,
)
from rocobench.rl.mappo import MAPPOConfig


def evaluate(env: SortSymbolicEnv, policy, n_episodes: int) -> Dict[str, float]:
    """policy must expose .act(obs_arr, mask_arr) -> (actions, _)"""
    successes, returns, lens, invalids = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        ep_ret, ep_len, ep_inv = 0.0, 0, 0
        done = False
        info = {}
        while not done:
            obs_arr = np.stack([obs["per_agent"][ag]["obs"]
                                for ag in env.AGENTS], axis=0)
            mask_arr = np.stack([obs["per_agent"][ag]["mask"]
                                 for ag in env.AGENTS], axis=0)
            actions, _ = policy.act(obs_arr, mask_arr)
            joint = {ag: int(actions[i]) for i, ag in enumerate(env.AGENTS)}
            obs, r, done, info = env.step(joint)
            ep_ret += r
            ep_len += 1
            ep_inv += int(info.get("n_invalid", 0))
            if ep_len >= env.max_steps:
                break
        successes.append(1.0 if info.get("success") else 0.0)
        returns.append(ep_ret)
        lens.append(ep_len)
        invalids.append(ep_inv)
    return dict(success=float(np.mean(successes)),
                steps=float(np.mean(lens)),
                invalid=float(np.mean(invalids)),
                ret=float(np.mean(returns)))


def make_env(use_mask=True, use_shaping=True, use_handoff=True,
             seed=42) -> SortSymbolicEnv:
    return SortSymbolicEnv(seed=seed, use_mask=use_mask,
                           use_shaping=use_shaping, use_handoff=use_handoff)


def train_one(label: str, train_steps: int, **env_kwargs) -> MAPPOAgent:
    print(f"\n--- training [{label}] ({train_steps} steps) ---")
    env = make_env(**env_kwargs)
    cfg = MAPPOConfig(rollout_len=256)
    agent = train_mappo(env, total_steps=train_steps, log_interval=20, cfg=cfg)
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=30_000)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    eval_env = make_env(seed=args.seed)
    np.random.seed(args.seed)

    rows = []  # (label, metrics)

    # ---- baselines ----
    print("\n=== Baselines ===")
    for label, factory in [
        ("random_no_mask", lambda: RandomNoMaskPolicy(eval_env, seed=args.seed)),
        ("random_masked",  lambda: RandomMaskedPolicy(eval_env, seed=args.seed)),
        ("scripted",       lambda: ScriptedHeuristicPolicy(eval_env)),
    ]:
        m = evaluate(eval_env, factory(), args.eval_episodes)
        print(f"  {label:18s}  succ={m['success']:.2%}  steps={m['steps']:.2f}  "
              f"inv={m['invalid']:.2f}  ret={m['ret']:.2f}")
        rows.append((label, m))

    # ---- mappo full ----
    full = train_one("mappo_full", args.train_steps,
                     use_mask=True, use_shaping=True, use_handoff=True)
    m = evaluate(eval_env, full, args.eval_episodes)
    rows.append(("mappo_full", m))
    print(f"  mappo_full         succ={m['success']:.2%}  steps={m['steps']:.2f}  "
          f"inv={m['invalid']:.2f}  ret={m['ret']:.2f}")

    # ---- ablations ----
    ablations = [
        ("mappo_no_mask",    dict(use_mask=False, use_shaping=True,  use_handoff=True)),
        ("mappo_no_shaping", dict(use_mask=True,  use_shaping=False, use_handoff=True)),
        ("mappo_no_handoff", dict(use_mask=True,  use_shaping=True,  use_handoff=False)),
    ]
    for label, kwargs in ablations:
        agent = train_one(label, args.train_steps, **kwargs)
        # always evaluate against the FULL env (with mask + full reward)
        m = evaluate(eval_env, agent, args.eval_episodes)
        rows.append((label, m))
        print(f"  {label:18s}  succ={m['success']:.2%}  steps={m['steps']:.2f}  "
              f"inv={m['invalid']:.2f}  ret={m['ret']:.2f}")

    # ---- summary table ----
    print("\n=================== Final benchmark ===================")
    print(f"{'method':20s} | {'success':>8s} | {'steps':>6s} | "
          f"{'invalids':>8s} | {'return':>8s}")
    print("-" * 60)
    for label, m in rows:
        print(f"{label:20s} | {m['success']:>8.2%} | {m['steps']:>6.2f} | "
              f"{m['invalid']:>8.2f} | {m['ret']:>8.2f}")


if __name__ == "__main__":
    main()
