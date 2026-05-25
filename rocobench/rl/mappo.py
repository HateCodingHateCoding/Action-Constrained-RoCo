"""Minimal MAPPO with masked categorical actor and centralized critic.

Designed for the symbolic sort env (small action/state dims). Pure PyTorch,
no external deps beyond torch and numpy.

Usage:
    from rocobench.rl import SortSymbolicEnv, train_mappo
    env = SortSymbolicEnv(seed=0)
    train_mappo(env, total_steps=200_000)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError as e:
    raise ImportError(
        "MAPPO trainer requires PyTorch. Install via `pip install torch`."
    ) from e


def _mlp(sizes: List[int], act=nn.Tanh) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class MaskedCategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden, n_actions])

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        # mask is 1.0 for legal, 0.0 for illegal
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(mask > 0.5, logits, torch.full_like(logits, neg_inf))
        return Categorical(logits=logits)


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, hidden=(128, 128)):
        super().__init__()
        self.net = _mlp([state_dim, *hidden, 1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class MAPPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    minibatch: int = 256
    rollout_len: int = 256
    max_grad_norm: float = 0.5
    device: str = "cpu"


class MAPPOAgent:
    def __init__(self, obs_dim: int, state_dim: int, n_actions: int,
                 n_agents: int, cfg: Optional[MAPPOConfig] = None):
        self.cfg = cfg or MAPPOConfig()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.device = torch.device(self.cfg.device)
        # Parameter sharing across agents (small problem, low variance).
        self.actor = MaskedCategoricalActor(obs_dim, n_actions).to(self.device)
        self.critic = CentralCritic(state_dim).to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

    @torch.no_grad()
    def act(self, obs_batch: np.ndarray, mask_batch: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
        """obs_batch: (n_agents, obs_dim); mask_batch: (n_agents, n_actions)."""
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.float32, device=self.device)
        dist = self.actor(obs_t, mask_t)
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy()

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(s).item())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        obs = batch["obs"]
        masks = batch["masks"]
        actions = batch["actions"]
        old_logp = batch["log_probs"]
        adv = batch["advantages"]
        ret = batch["returns"]
        states = batch["states"]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = obs.shape[0]
        idxs = np.arange(N)
        stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0)
        for _ in range(cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, cfg.minibatch):
                mb = idxs[start:start + cfg.minibatch]
                mb = torch.as_tensor(mb, dtype=torch.long)

                dist = self.actor(obs[mb], masks[mb])
                logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()
                ratio = (logp - old_logp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[mb]
                pi_loss = -torch.min(s1, s2).mean()

                v = self.critic(states[mb])
                v_loss = F.mse_loss(v, ret[mb])

                loss = pi_loss + cfg.vf_coef * v_loss - cfg.entropy_coef * entropy
                self.opt_a.zero_grad(set_to_none=True)
                self.opt_c.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt_a.step()
                self.opt_c.step()

                stats["pi_loss"] += float(pi_loss.item())
                stats["v_loss"] += float(v_loss.item())
                stats["entropy"] += float(entropy.item())
        return stats

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


def _collect_rollout(env, agent: MAPPOAgent, rollout_len: int):
    obs_buf, mask_buf, act_buf, logp_buf = [], [], [], []
    state_buf, rew_buf, done_buf, val_buf = [], [], [], []
    ep_returns, ep_lens, successes = [], [], []
    cur_obs = env.get_obs()
    ep_ret, ep_len = 0.0, 0
    for _ in range(rollout_len):
        per_agent = cur_obs["per_agent"]
        obs_arr = np.stack([per_agent[ag]["obs"] for ag in env.AGENTS], axis=0)
        mask_arr = np.stack([per_agent[ag]["mask"] for ag in env.AGENTS], axis=0)
        actions, logps = agent.act(obs_arr, mask_arr)
        v = agent.value(cur_obs["global_obs"])

        joint = {ag: int(actions[i]) for i, ag in enumerate(env.AGENTS)}
        nxt, r, done, info = env.step(joint)
        for i in range(env.n_agents):
            obs_buf.append(obs_arr[i])
            mask_buf.append(mask_arr[i])
            act_buf.append(int(actions[i]))
            logp_buf.append(float(logps[i]))
            state_buf.append(cur_obs["global_obs"])
            rew_buf.append(float(r))      # shared team reward
            done_buf.append(bool(done))
            val_buf.append(v)

        ep_ret += r
        ep_len += 1
        if done:
            ep_returns.append(ep_ret)
            ep_lens.append(ep_len)
            successes.append(1.0 if info.get("success") else 0.0)
            cur_obs = env.reset()
            ep_ret, ep_len = 0.0, 0
        else:
            cur_obs = nxt
    last_v = agent.value(cur_obs["global_obs"])
    return dict(
        obs=np.array(obs_buf, dtype=np.float32),
        masks=np.array(mask_buf, dtype=np.float32),
        actions=np.array(act_buf, dtype=np.int64),
        log_probs=np.array(logp_buf, dtype=np.float32),
        states=np.array(state_buf, dtype=np.float32),
        rewards=np.array(rew_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=np.bool_),
        values=np.array(val_buf, dtype=np.float32),
        last_value=last_v,
        ep_returns=ep_returns,
        ep_lens=ep_lens,
        successes=successes,
        cur_obs=cur_obs,
    )


def _compute_gae(rewards, dones, values, last_value, gamma, lam, n_agents):
    """Per-agent GAE: rewards/dones/values are flattened along (T, n_agents).
    We treat each agent's slice independently with the shared team reward.
    """
    T = len(rewards) // n_agents
    adv = np.zeros_like(rewards)
    # Reshape into (T, n_agents)
    r = rewards.reshape(T, n_agents)
    d = dones.reshape(T, n_agents).astype(np.float32)
    v = values.reshape(T, n_agents)
    next_adv = np.zeros(n_agents, dtype=np.float32)
    next_v = np.full(n_agents, last_value, dtype=np.float32)
    out = np.zeros_like(v)
    for t in reversed(range(T)):
        nonterm = 1.0 - d[t]
        delta = r[t] + gamma * next_v * nonterm - v[t]
        next_adv = delta + gamma * lam * nonterm * next_adv
        out[t] = next_adv
        next_v = v[t]
    adv_flat = out.reshape(-1)
    ret_flat = adv_flat + values
    return adv_flat, ret_flat


def train_mappo(env, total_steps: int = 100_000, log_interval: int = 5,
                save_path: Optional[str] = None,
                cfg: Optional[MAPPOConfig] = None,
                return_log: bool = False):
    cfg = cfg or MAPPOConfig()
    agent = MAPPOAgent(
        obs_dim=env.obs_dim, state_dim=env.state_dim,
        n_actions=env.n_actions, n_agents=env.n_agents, cfg=cfg,
    )
    env.reset()
    iters = max(1, total_steps // cfg.rollout_len)
    # Per-iteration record: env_steps reached, smoothed return, smoothed succ
    curve = dict(env_steps=[], return_avg=[], success=[])
    log = dict(steps=0, ep_return_avg=[], success_rate=[])
    t0 = time.time()
    for it in range(iters):
        roll = _collect_rollout(env, agent, cfg.rollout_len)
        adv, ret = _compute_gae(roll["rewards"], roll["dones"], roll["values"],
                                roll["last_value"], cfg.gamma, cfg.gae_lambda,
                                env.n_agents)
        batch = dict(
            obs=torch.as_tensor(roll["obs"], dtype=torch.float32, device=agent.device),
            masks=torch.as_tensor(roll["masks"], dtype=torch.float32, device=agent.device),
            actions=torch.as_tensor(roll["actions"], dtype=torch.long, device=agent.device),
            log_probs=torch.as_tensor(roll["log_probs"], dtype=torch.float32, device=agent.device),
            advantages=torch.as_tensor(adv, dtype=torch.float32, device=agent.device),
            returns=torch.as_tensor(ret, dtype=torch.float32, device=agent.device),
            states=torch.as_tensor(roll["states"], dtype=torch.float32, device=agent.device),
        )
        stats = agent.update(batch)
        log["steps"] += cfg.rollout_len
        if roll["ep_returns"]:
            log["ep_return_avg"].append(float(np.mean(roll["ep_returns"])))
            log["success_rate"].append(float(np.mean(roll["successes"])))
        # Always record a curve point even if no episode finished this rollout.
        curve["env_steps"].append(log["steps"])
        curve["return_avg"].append(
            float(np.mean(log["ep_return_avg"][-10:])) if log["ep_return_avg"] else float("nan")
        )
        curve["success"].append(
            float(np.mean(log["success_rate"][-10:])) if log["success_rate"] else float("nan")
        )
        if (it + 1) % log_interval == 0 or it == iters - 1:
            print(f"[iter {it+1}/{iters}] env_steps={log['steps']} "
                  f"ep_return={curve['return_avg'][-1]:.2f} "
                  f"success={curve['success'][-1]:.2%} "
                  f"pi_loss={stats['pi_loss']:.3f} v_loss={stats['v_loss']:.3f} "
                  f"entropy={stats['entropy']:.3f} "
                  f"elapsed={time.time()-t0:.1f}s")
    if save_path:
        agent.save(save_path)
        print(f"Saved checkpoint to {save_path}")
    if return_log:
        return agent, curve
    return agent
