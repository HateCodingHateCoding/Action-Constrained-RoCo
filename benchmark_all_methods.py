"""End-to-end benchmark across all 4 high-level policies on the symbolic env.
Useful as a fast sanity check before paying the cost of real-MuJoCo runs.

Usage:
    python benchmark_all_methods.py --eval-episodes 100 --llm-model glm-4-flash
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from rocobench.rl import (
    ActionCodec, MAPPOAgent, SortSymbolicEnv,
    RandomMaskedPolicy, ScriptedHeuristicPolicy, LLMRLHybridPolicy,
)
from rocobench.rl.mappo import MAPPOConfig


def evaluate(env, policy, n_episodes: int, with_scene: bool = False) -> dict:
    successes, returns, lens, invalids, llm_used = [], [], [], [], 0
    for _ in range(n_episodes):
        obs = env.reset()
        ep_ret, ep_len, ep_inv = 0.0, 0, 0
        done = False
        info = {}
        while not done:
            obs_arr = np.stack([obs["per_agent"][ag]["obs"] for ag in env.AGENTS], axis=0)
            mask_arr = np.stack([obs["per_agent"][ag]["mask"] for ag in env.AGENTS], axis=0)
            if with_scene:
                # Symbolic env doesn't have describe_obs; build a tiny scene
                scene = "; ".join(
                    f"{c} on {env.state.cube_panel[c]}" for c in env.state.cube_panel
                )
                actions, _ = policy.act(obs_arr, mask_arr, scene_desc=scene)
            else:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/sort_mappo_smoke.pt")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--llm-model", default="glm-4-flash")
    p.add_argument("--llm-episodes", type=int, default=20,
                   help="hybrid eval may be slow; use fewer episodes")
    args = p.parse_args()

    env = SortSymbolicEnv(seed=999)
    codec = ActionCodec(env.get_action_vocab())
    cfg = MAPPOConfig(device="cpu")
    rl = MAPPOAgent(env.obs_dim, env.state_dim, env.n_actions, env.n_agents, cfg=cfg)
    rl.load(args.ckpt)
    print(f"Loaded MAPPO from {args.ckpt}")

    rows = []
    print("\n=== Quick baselines ===")
    for label, factory, eps, with_scene in [
        ("random_masked", lambda: RandomMaskedPolicy(env, seed=0), args.eval_episodes, False),
        ("scripted", lambda: ScriptedHeuristicPolicy(env), args.eval_episodes, False),
        ("rl_only", lambda: rl, args.eval_episodes, False),
    ]:
        t0 = time.time()
        m = evaluate(env, factory(), eps)
        m["time"] = time.time() - t0
        print(f"  {label:14s}  succ={m['success']:.2%}  steps={m['steps']:.2f}  "
              f"inv={m['invalid']:.2f}  time={m['time']:.1f}s")
        rows.append((label, m))

    # Hybrid: needs the LLM. May be slow due to API latency.
    print(f"\n=== Hybrid (LLM={args.llm_model}, {args.llm_episodes} eps) ===")
    from prompting.llm_api import chat_completion

    def _llm(prompt: str) -> str:
        content, _ = chat_completion(
            model=args.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.0, max_retries=3,
        )
        return content or ""

    hybrid = LLMRLHybridPolicy(rl, codec, llm_call=_llm)
    t0 = time.time()
    m = evaluate(env, hybrid, args.llm_episodes, with_scene=True)
    m["time"] = time.time() - t0
    print(f"  hybrid          succ={m['success']:.2%}  steps={m['steps']:.2f}  "
          f"inv={m['invalid']:.2f}  time={m['time']:.1f}s")
    rows.append(("hybrid", m))

    print("\n=== Final ===")
    print(f"{'method':14s} | {'success':>8s} | {'steps':>6s} | {'inv':>5s} | {'time':>7s}")
    print("-" * 56)
    for label, m in rows:
        print(f"{label:14s} | {m['success']:>8.2%} | {m['steps']:>6.2f} | "
              f"{m['invalid']:>5.2f} | {m.get('time', 0):>6.1f}s")


if __name__ == "__main__":
    main()
