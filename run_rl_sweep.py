"""Run a trained MAPPO Sweep policy on the real MuJoCo SweepTask.

Same structure as run_rl_sort.py: skip the LLM, format actions into the
EXECUTE response string the existing LLMResponseParser understands, then
run RRT + execute via PlannedPathPolicy.

Usage:
    python run_rl_sweep.py --load checkpoints/sweep_mappo.pt --num-runs 3 --max-steps 12
"""
from __future__ import annotations
import argparse
import logging
import os
import numpy as np

from rocobench.envs import SweepTask
from rocobench import PlannedPathPolicy
from rocobench.rl import (
    ActionCodec, MAPPOAgent, SweepSymbolicEnv,
    obs_to_sweep_features, sweep_action_to_response,
)
from rocobench.rl.mappo import MAPPOConfig
from rocobench.rl.sweep_symbolic_env import VERBS, TARGETS, CUBES
from prompting.parser import LLMResponseParser

logging.basicConfig(level=logging.INFO)


def _try_execute(env, plans, robots, env_obs):
    obs = env_obs
    for plan in plans:
        try:
            policy = PlannedPathPolicy(
                physics=env.physics,
                robots=robots,
                path_plan=plan,
                graspable_object_names=env.get_graspable_objects(),
                allowed_collision_pairs=env.get_allowed_collision_pairs(),
                control_freq=10,
            )
            plan_ok, why = policy.plan(env)
        except Exception as e:
            print(f"    policy/IK exception: {e}")
            return False, obs
        print(f"    RRT plan_ok={plan_ok} reason={why}")
        if not plan_ok:
            return False, obs
        try:
            while not policy.plan_exhausted:
                sim_action = policy.act(obs, env.physics)
                obs, _r, _done, _info = env.step(sim_action, verbose=False)
        except Exception as e:
            print(f"    execution exception: {e}")
            return False, obs
        obs = env.get_obs()
    return True, obs


def run_one_episode(env: SweepTask, agent: MAPPOAgent, codec: ActionCodec,
                    parser: LLMResponseParser, max_steps: int = 12,
                    render_dir: str = None, video_format: str = "mp4",
                    sticky_dustpan: bool = True) -> dict:
    obs = env.reset(reload=True)
    success = False
    step = 0
    history = []
    robots = env.get_sim_robots()
    focus = {ag: None for ag in ["Alice", "Bob"]}
    # Symbolic env teleports SWEEP -> IN_DUSTPAN; in real physics the cube
    # may never make it to the dustpan within one MOVE+SWEEP cycle. We
    # remember which cubes have already been swept *at the policy level*
    # so the policy can move on to the next cube / DUMP. When we then
    # query the real env for done, the cube must actually be in the bin
    # (snap or DUMP physics handles the final transfer).
    swept_cubes: set = set()

    for step in range(max_steps):
        feats = obs_to_sweep_features(env, obs, focus,
                                      sticky_swept=swept_cubes if sticky_dustpan else None)
        agents = ["Alice", "Bob"]
        obs_arr = np.stack([feats["per_agent"][ag]["obs"] for ag in agents], axis=0)
        mask_arr = np.stack([feats["per_agent"][ag]["mask"] for ag in agents], axis=0)
        actions, _ = agent.act(obs_arr, mask_arr)
        joint = {ag: int(actions[i]) for i, ag in enumerate(agents)}

        # Update focus for any MOVE action.
        for ag in agents:
            oi, ti = codec.decode(joint[ag])
            if VERBS[oi] == "MOVE":
                focus[ag] = TARGETS[ti]

        if all(j == 0 for j in joint.values()):
            print(f"  step {step}: policy chose all-WAIT, stopping.")
            break

        response = sweep_action_to_response(codec, joint)
        print(f"  step {step}:\n{response}")
        history.append(response)

        ok, reason, plans = parser.parse(obs, response)
        if not ok:
            print(f"  parser rejected: {reason}")
            break

        sim_data = env.save_intermediate_state()
        exec_ok, obs = _try_execute(env, plans, robots, obs)
        if not exec_ok:
            print("  execution failed, rewinding step.")
            env.load_saved_state(sim_data)
            obs = env.get_obs()
        else:
            # Mark cubes as sticky-swept once a SWEEP action ran successfully.
            if sticky_dustpan:
                bob_oi, bob_ti = codec.decode(joint["Bob"])
                if VERBS[bob_oi] == "SWEEP":
                    target_cube = TARGETS[bob_ti]
                    if target_cube in CUBES:
                        swept_cubes.add(target_cube)
                        print(f"    sticky: marked {target_cube} as IN_DUSTPAN")

        _r, done = env.get_reward_done(obs)
        if done:
            success = True
            print(f"  task solved at step {step}")
            break

    if render_dir is not None:
        os.makedirs(render_dir, exist_ok=True)
        vid_name = os.path.join(render_dir, f"rollout.{video_format}")
        try:
            env.export_render_to_video(vid_name, out_type=video_format, fps=20)
            print(f"  video saved to {vid_name}")
        except Exception as e:
            print(f"  video export skipped: {e}")
    return dict(success=success, steps=step + 1, history=history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True, help="MAPPO checkpoint path")
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-dir", type=str, default="data/rl_real_sweep")
    p.add_argument("--video-format", type=str, default="mp4")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    sym_env = SweepSymbolicEnv(seed=0)
    cfg = MAPPOConfig(device=args.device)
    agent = MAPPOAgent(sym_env.obs_dim, sym_env.state_dim,
                       sym_env.n_actions, sym_env.n_agents, cfg=cfg)
    agent.load(args.load)
    print(f"Loaded sweep MAPPO from {args.load}")

    env = SweepTask(np_seed=args.seed, render_point_cloud=False)
    codec = ActionCodec(env.get_action_vocab())
    parser = LLMResponseParser(
        env=env, llm_output_mode="action_only",
        robot_agent_names=env.robot_name_map,
        response_keywords=["NAME", "ACTION"],
        direct_waypoints=0,
        use_prepick=env.use_prepick,
        use_preplace=env.use_preplace,
    )

    successes, lens = [], []
    for run in range(args.num_runs):
        env.seed(np_seed=args.seed + run)
        print(f"\n===== Run {run} (seed={args.seed + run}) =====")
        out = run_one_episode(env, agent, codec, parser,
                              max_steps=args.max_steps,
                              render_dir=os.path.join(args.render_dir, f"run_{run}"),
                              video_format=args.video_format)
        successes.append(1.0 if out["success"] else 0.0)
        lens.append(out["steps"])
        print(f"Run {run}: success={out['success']} steps={out['steps']}")

    print("\n=== Summary ===")
    print(f"  success_rate: {np.mean(successes):.2%}  ({int(sum(successes))}/{len(successes)})")
    print(f"  avg steps:    {np.mean(lens):.2f}")


if __name__ == "__main__":
    main()
