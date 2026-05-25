"""Train MAPPO under several ablation configs, save the learning curves to
JSON, and produce a single learning-curve plot.

Outputs:
    figures/training_curves.json  - raw per-iteration metrics
    figures/fig_curves.png        - plot of success rate vs env steps

Usage:
    python make_training_curves.py --steps 25000
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rocobench.rl import SortSymbolicEnv, train_mappo
from rocobench.rl.mappo import MAPPOConfig


CONFIGS = [
    ("full",         dict(use_mask=True,  use_shaping=True,  use_handoff=True),
     "#3b8bba", "-"),
    ("no_mask",      dict(use_mask=False, use_shaping=True,  use_handoff=True),
     "#d65f5f", "-"),
    ("no_shaping",   dict(use_mask=True,  use_shaping=False, use_handoff=True),
     "#a0a0a0", "--"),
    ("no_handoff",   dict(use_mask=True,  use_shaping=True,  use_handoff=False),
     "#888888", ":"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="figures")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = MAPPOConfig(rollout_len=256)

    all_curves = {}
    for label, env_kwargs, _color, _ls in CONFIGS:
        print(f"\n=== training [{label}] ({args.steps} steps) ===")
        env = SortSymbolicEnv(seed=args.seed, **env_kwargs)
        t0 = time.time()
        _, curve = train_mappo(env, total_steps=args.steps,
                               log_interval=20, cfg=cfg, return_log=True)
        curve["elapsed_sec"] = time.time() - t0
        all_curves[label] = curve

    out_json = os.path.join(args.out_dir, "training_curves.json")
    with open(out_json, "w") as f:
        json.dump(all_curves, f, indent=2)
    print(f"\nSaved raw curves to {out_json}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, _kw, color, ls in CONFIGS:
        c = all_curves[label]
        x = np.array(c["env_steps"])
        y = np.array(c["success"], dtype=np.float64)
        ax1.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1.7)
        yr = np.array(c["return_avg"], dtype=np.float64)
        ax2.plot(x, yr, label=label, color=color, linestyle=ls, linewidth=1.7)

    ax1.set_xlabel("Environment steps")
    ax1.set_ylabel("Success rate (rolling avg)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("MAPPO learning curves on Sort task")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.set_xlabel("Environment steps")
    ax2.set_ylabel("Avg episode return")
    ax2.set_title("Episode return (rolling avg)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower right", fontsize=9)

    out_png = os.path.join(args.out_dir, "fig_curves.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning-curve plot to {out_png}")


if __name__ == "__main__":
    main()
