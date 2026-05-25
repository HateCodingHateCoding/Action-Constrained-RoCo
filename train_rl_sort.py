"""Train MAPPO on the symbolic Sort task.

Examples:
    python train_rl_sort.py
    python train_rl_sort.py --steps 200000 --save checkpoints/sort_mappo.pt
    python train_rl_sort.py --eval-only --load checkpoints/sort_mappo.pt
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from rocobench.rl import SortSymbolicEnv, MAPPOAgent, train_mappo
from rocobench.rl.mappo import MAPPOConfig


def evaluate(env: SortSymbolicEnv, agent: MAPPOAgent, n_episodes: int = 50,
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
                rendered = [env.codec.to_str(joint[ag]) for ag in env.AGENTS]
                traj.append(f"  t={ep_len}: " +
                            " | ".join(f"{ag}={s}" for ag, s in zip(env.AGENTS, rendered)))
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
    return dict(
        success_rate=float(np.mean(successes)),
        return_mean=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        len_mean=float(np.mean(lens)),
        invalid_mean=float(np.mean(invalids)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000,
                   help="total environment steps for training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save", type=str, default="checkpoints/sort_mappo.pt")
    p.add_argument("--load", type=str, default="")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--render", action="store_true",
                   help="print action trajectories during eval")
    args = p.parse_args()

    env = SortSymbolicEnv(seed=args.seed)
    cfg = MAPPOConfig(lr=args.lr, rollout_len=args.rollout, device=args.device)

    if args.eval_only:
        agent = MAPPOAgent(env.obs_dim, env.state_dim, env.n_actions,
                           env.n_agents, cfg=cfg)
        if not args.load:
            raise SystemExit("--eval-only requires --load <ckpt>")
        agent.load(args.load)
        print(f"Loaded {args.load}")
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
