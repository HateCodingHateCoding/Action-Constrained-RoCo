"""Train MAPPO on the symbolic Sweep task.

Demonstrates that the same training code (mappo.py) handles a totally
different task — different agents (2 vs 3), different verbs, different
goal — without modification. Only the env (vocab/mask/dynamics/reward)
changes.

Usage:
    python train_rl_sweep.py --steps 30000
"""
from __future__ import annotations
import argparse
import numpy as np

from rocobench.rl import SweepSymbolicEnv, MAPPOAgent, train_mappo
from rocobench.rl.mappo import MAPPOConfig


def evaluate(env: SweepSymbolicEnv, agent: MAPPOAgent, n_episodes: int = 50,
             render: bool = False) -> dict:
    successes, returns, lens, invalids = [], [], [], []
    for ep in range(n_episodes):
        obs = env.reset()
        ep_ret, ep_len, ep_inv = 0.0, 0, 0
        traj = []
        done = False
        while not done:
            obs_arr = np.stack([obs["per_agent"][ag]["obs"] for ag in env.AGENTS], axis=0)
            mask_arr = np.stack([obs["per_agent"][ag]["mask"] for ag in env.AGENTS], axis=0)
            actions, _ = agent.act(obs_arr, mask_arr)
            joint = {ag: int(actions[i]) for i, ag in enumerate(env.AGENTS)}
            if render:
                from rocobench.rl.sweep_symbolic_env import VERBS, TARGETS
                rendered = []
                for i, ag in enumerate(env.AGENTS):
                    oi, ti = env.codec.decode(joint[ag])
                    rendered.append(f"{ag}={VERBS[oi]} {TARGETS[ti]}")
                traj.append(f"  t={ep_len}: " + " | ".join(rendered)
                            + f"  status={env.state.cube_status}")
            obs, r, done, info = env.step(joint)
            ep_ret += r
            ep_len += 1
            ep_inv += int(info.get("n_invalid", 0))
        successes.append(1.0 if info.get("success") else 0.0)
        returns.append(ep_ret)
        lens.append(ep_len)
        invalids.append(ep_inv)
        if render and ep < 3:
            print(f"-- Episode {ep} success={successes[-1]:.0f} "
                  f"return={ep_ret:.2f} len={ep_len} --")
            for line in traj:
                print(line)
    return dict(success_rate=float(np.mean(successes)),
                return_mean=float(np.mean(returns)),
                len_mean=float(np.mean(lens)),
                invalid_mean=float(np.mean(invalids)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="checkpoints/sweep_mappo.pt")
    p.add_argument("--load", type=str, default="")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    env = SweepSymbolicEnv(seed=args.seed)
    cfg = MAPPOConfig(rollout_len=256)

    if args.eval_only:
        agent = MAPPOAgent(env.obs_dim, env.state_dim, env.n_actions,
                           env.n_agents, cfg=cfg)
        if not args.load:
            raise SystemExit("--eval-only requires --load <ckpt>")
        agent.load(args.load)
    else:
        agent = train_mappo(env, total_steps=args.steps, save_path=args.save,
                            cfg=cfg)

    print("\n=== Evaluation ===")
    metrics = evaluate(env, agent, n_episodes=args.eval_episodes,
                       render=args.render)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
