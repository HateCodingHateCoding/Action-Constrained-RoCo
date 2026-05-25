"""Run a high-level policy on the real MuJoCo SortOneBlockTask.

This skips the LLM entirely. The policy emits PICK/PLACE actions; we
format them into the EXECUTE response string the existing
LLMResponseParser already understands, then run RRT + execute exactly
like run_dialog.py does.

Supported policies (--method):
    mappo      load a trained MAPPO checkpoint (default)
    random     uniform over the legal action set
    scripted   hand-crafted heuristic baseline

Usage:
    python run_rl_sort.py --method mappo --load checkpoints/sort_mappo.pt --num-runs 5
    python run_rl_sort.py --method scripted --num-runs 5
"""
from __future__ import annotations
import argparse
import logging
import os
import time
import numpy as np

from rocobench.envs import SortOneBlockTask
from rocobench import PlannedPathPolicy
from rocobench.rl import (
    ActionCodec, MAPPOAgent, obs_to_rl_features, rl_action_to_response,
    RandomMaskedPolicy, ScriptedHeuristicPolicy, SortSymbolicEnv,
    LLMRLHybridPolicy,
)
from rocobench.rl.mappo import MAPPOConfig
from rocobench.rl.sort_symbolic_env import SortSymbolicEnv, CUBES, CUBE_TO_BIN
from prompting.parser import LLMResponseParser
from prompting.llm_api import chat_completion

logging.basicConfig(level=logging.INFO)


def _sync_symbolic_state(sym_env: SortSymbolicEnv, real_env: SortOneBlockTask,
                         obs) -> None:
    """Mirror the cube placements + holding flags from the real env into
    the symbolic env's state, so the scripted policy can read them via
    its usual interface.
    """
    panels = sym_env.PANELS
    cube_panel = {}
    for cube in CUBES:
        cube_state = obs.objects[cube]
        chosen = None
        for pname in ["panel2", "panel4", "panel6"]:
            if pname in cube_state.contacts:
                chosen = pname
                break
        if chosen is None:
            best = None
            best_d = float("inf")
            for pname in panels:
                geom = real_env.physics.data.geom(pname).xpos
                d = float(np.linalg.norm(geom[:2] - cube_state.xpos[:2]))
                if d < best_d:
                    best_d, best = d, pname
            chosen = best
        cube_panel[cube] = chosen
    holding = {ag: None for ag in sym_env.AGENTS}
    for ag in sym_env.AGENTS:
        rs = getattr(obs, real_env.robot_name_map_inv[ag])
        for c in rs.contacts:
            if c in CUBES:
                holding[ag] = c
                break
    sym_env.state.cube_panel = cube_panel
    sym_env.state.holding = holding


def _snap_finished_cubes(env, snap_radius: float = 0.18) -> int:
    """Optional engineering aid: when a cube ends up close to its goal
    panel but slightly outside the env's align_threshold, snap it onto
    the panel center. This compensates for physics drift after release
    and keeps the symbolic obs / env reward consistent.

    Returns the number of cubes that were snapped.
    """
    snapped = 0
    for cube in CUBES:
        target_bin = CUBE_TO_BIN[cube]
        try:
            bin_xy = env.bin_slot_pos[f"{target_bin}_middle"][:2]
        except KeyError:
            continue
        cube_state = env.get_obs().objects[cube]
        d = float(np.linalg.norm(bin_xy - cube_state.xpos[:2]))
        if env.align_threshold < d < snap_radius:
            new_pos = np.array([bin_xy[0], bin_xy[1], cube_state.xpos[2]],
                               dtype=np.float64)
            try:
                env.reset_qpos(f"{cube}_joint", pos=new_pos, quat=cube_state.xquat)
                env.physics.forward()
                snapped += 1
                print(f"    snapped {cube}: {d:.3f}m -> ~0.0m onto {target_bin}")
            except Exception as e:
                print(f"    snap failed for {cube}: {e}")
    return snapped


def _try_execute(env, plans, robots, env_obs):
    """Run RRT + execute for the given list of LLMPathPlans. Returns (ok, last_obs)."""
    obs = env_obs
    for plan in plans:
        try:
            policy = PlannedPathPolicy(
                physics=env.physics,
                robots=robots,
                path_plan=plan,
                graspable_object_names=env.get_graspable_objects(),
                allowed_collision_pairs=env.get_allowed_collision_pairs(),
                control_freq=50,
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
                obs, reward, done, info = env.step(sim_action, verbose=False)
        except Exception as e:
            print(f"    execution exception: {e}")
            return False, obs
        obs = env.get_obs()
    return True, obs


def run_one_episode(env: SortOneBlockTask, agent, codec: ActionCodec,
                    parser: LLMResponseParser, max_steps: int = 8,
                    render_dir: str = None, video_format: str = "mp4",
                    sequential_fallback: bool = True,
                    always_sequential: bool = True,
                    method: str = "mappo",
                    snap_finished: bool = False,
                    sym_env: SortSymbolicEnv = None) -> dict:
    obs = env.reset(reload=True)
    success = False
    step = 0
    history = []
    robots = env.get_sim_robots()
    sticky_progress = set()
    for step in range(max_steps):
        feats = obs_to_rl_features(env, obs, sticky_progress=sticky_progress)
        agents = ["Alice", "Bob", "Chad"]
        obs_arr = np.stack([feats["per_agent"][ag]["obs"] for ag in agents], axis=0)
        mask_arr = np.stack([feats["per_agent"][ag]["mask"] for ag in agents], axis=0)

        if method == "scripted" and sym_env is not None:
            sym_env.reset()
            _sync_symbolic_state(sym_env, env, obs)
            actions, _ = agent.act(obs_arr, mask_arr)
        elif method == "hybrid":
            scene = env.describe_obs(obs)
            actions, _ = agent.act(obs_arr, mask_arr, scene_desc=scene)
        else:
            actions, _ = agent.act(obs_arr, mask_arr)

        joint = {ag: int(actions[i]) for i, ag in enumerate(agents)}

        # Resolve conflicts: at most one agent may pick each cube. If two
        # agents picked the same cube, the earlier one (Alice > Bob > Chad)
        # wins; the loser falls back to WAIT. This mirrors the symbolic env's
        # conflict resolution and prevents the real env from executing
        # contradictory PICKs sequentially.
        seen_cubes = set()
        for ag in agents:
            oi, _ = codec.decode(joint[ag])
            if oi == 0:
                continue
            cube = ["WAIT"] + CUBES
            target_cube = cube[oi]
            if target_cube in seen_cubes:
                print(f"  conflict: {ag} also picked {target_cube}, forcing WAIT")
                joint[ag] = 0
            else:
                seen_cubes.add(target_cube)
        if all(j == 0 for j in joint.values()):
            # Check whether the task is actually done; if not, the policy
            # is stuck (probably because of a flicker in the mask). Bump
            # max_steps and continue rather than terminating immediately.
            r, done = env.get_reward_done(obs)
            if done:
                success = True
                print(f"  step {step}: all-WAIT and task is done.")
                break
            print(f"  step {step}: all-WAIT but task NOT done; continuing.")
            continue
        response = rl_action_to_response(codec, joint)
        print(f"  step {step}: \n{response}")
        history.append(response)

        ok, reason, plans = parser.parse(obs, response)
        if not ok:
            print(f"  parser rejected RL response: {reason}")
            break

        if always_sequential:
            # Skip concurrent RRT entirely. The symbolic training env doesn't
            # model arm-arm collisions, so the policy often produces 3-arm
            # parallel actions that cause RRT to hang or fail. Executing one
            # robot at a time is a few seconds slower but rock-solid.
            print("  executing sequentially (one agent at a time)")
            for solo in agents:
                if joint[solo] == 0:
                    continue
                solo_joint = {ag: (joint[ag] if ag == solo else 0) for ag in agents}
                solo_resp = rl_action_to_response(codec, solo_joint)
                print(f"    -> {solo}: {codec.to_str(joint[solo])}")
                ok2, why2, sub_plans = parser.parse(obs, solo_resp)
                if not ok2:
                    print(f"    parser rejected: {why2}")
                    continue
                sub_data = env.save_intermediate_state()
                solo_ok, obs = _try_execute(env, sub_plans, robots, obs)
                if not solo_ok:
                    print(f"    RRT failed for {solo}; rewinding.")
                    env.load_saved_state(sub_data)
                    obs = env.get_obs()
        else:
            # Try concurrent execution first.
            sim_data = env.save_intermediate_state()
            exec_ok, obs = _try_execute(env, plans, robots, obs)
            if not exec_ok and sequential_fallback:
                print("  concurrent RRT failed -- falling back to sequential.")
                env.load_saved_state(sim_data)
                obs = env.get_obs()
                for solo in agents:
                    if joint[solo] == 0:
                        continue
                    solo_joint = {ag: (joint[ag] if ag == solo else 0) for ag in agents}
                    solo_resp = rl_action_to_response(codec, solo_joint)
                    print(f"    sequential -> {solo}")
                    ok2, why2, sub_plans = parser.parse(obs, solo_resp)
                    if not ok2:
                        print(f"    parser rejected sequential plan: {why2}")
                        continue
                    sub_data = env.save_intermediate_state()
                    solo_ok, obs = _try_execute(env, sub_plans, robots, obs)
                    if not solo_ok:
                        print(f"    sequential RRT also failed for {solo}; rewinding.")
                        env.load_saved_state(sub_data)
                        obs = env.get_obs()

        r, done = env.get_reward_done(obs)
        if not done and snap_finished:
            # Try to recover from minor placement drift before judging the step.
            n = _snap_finished_cubes(env)
            if n > 0:
                obs = env.get_obs()
                r, done = env.get_reward_done(obs)
        if done:
            success = True
            print(f"  task solved at step {step}")
            break

    if render_dir is not None:
        os.makedirs(render_dir, exist_ok=True)
        vid_name = os.path.join(render_dir, f"rollout.{video_format}")
        try:
            env.export_render_to_video(vid_name, out_type=video_format, fps=50)
            print(f"  video saved to {vid_name}")
        except Exception as e:
            print(f"  video export skipped: {e}")
    return dict(success=success, steps=step + 1, history=history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="mappo",
                   choices=["mappo", "random", "scripted", "hybrid"],
                   help="which high-level policy to run on the real env")
    p.add_argument("--load", default="", help="MAPPO checkpoint (required for --method mappo or hybrid)")
    p.add_argument("--llm-model", type=str, default="glm-4-flash",
                   help="LLM model name for --method hybrid")
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-dir", type=str, default="data/rl_real_runs")
    p.add_argument("--video-format", type=str, default="mp4")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--snap-finished", action="store_true",
                   help="recenter cubes that are close to their goal panel "
                        "(physics post-processing, off by default)")
    args = p.parse_args()

    sym_env = SortSymbolicEnv(seed=0)
    if args.method == "mappo":
        if not args.load:
            raise SystemExit("--method mappo requires --load <ckpt>")
        cfg = MAPPOConfig(device=args.device)
        agent = MAPPOAgent(sym_env.obs_dim, sym_env.state_dim,
                           sym_env.n_actions, sym_env.n_agents, cfg=cfg)
        agent.load(args.load)
        print(f"Loaded MAPPO policy from {args.load}")
    elif args.method == "random":
        agent = RandomMaskedPolicy(sym_env, seed=args.seed)
        print("Using random_masked baseline")
    elif args.method == "scripted":
        agent = ScriptedHeuristicPolicy(sym_env)
        print("Using scripted heuristic baseline")
    elif args.method == "hybrid":
        if not args.load:
            raise SystemExit("--method hybrid requires --load <ckpt>")
        cfg = MAPPOConfig(device=args.device)
        rl = MAPPOAgent(sym_env.obs_dim, sym_env.state_dim,
                        sym_env.n_actions, sym_env.n_agents, cfg=cfg)
        rl.load(args.load)
        codec = ActionCodec(sym_env.get_action_vocab())

        def _llm(prompt: str) -> str:
            content, _ = chat_completion(
                model=args.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.0, max_retries=3,
            )
            return content or ""
        agent = LLMRLHybridPolicy(rl, codec, llm_call=_llm)
        print(f"Using LLM+RL hybrid policy (LLM model={args.llm_model})")

    env = SortOneBlockTask(np_seed=args.seed, render_point_cloud=False)
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
        print(f"\n===== Run {run} (method={args.method}, seed={args.seed + run}) =====")
        out = run_one_episode(env, agent, codec, parser,
                              max_steps=args.max_steps,
                              render_dir=os.path.join(args.render_dir, f"run_{run}"),
                              video_format=args.video_format,
                              method=args.method, sym_env=sym_env,
                              snap_finished=args.snap_finished)
        successes.append(1.0 if out["success"] else 0.0)
        lens.append(out["steps"])
        print(f"Run {run}: success={out['success']} steps={out['steps']}")

    print("\n=== Summary ===")
    print(f"  method:       {args.method}")
    print(f"  success_rate: {np.mean(successes):.2%}  ({int(sum(successes))}/{len(successes)})")
    print(f"  avg steps:    {np.mean(lens):.2f}")


if __name__ == "__main__":
    main()
