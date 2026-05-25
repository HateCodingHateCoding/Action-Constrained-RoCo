# Multi-Arm Collaboration with RL-Guided Action Selection

> Extension of [RoCo: Dialectic Multi-Robot Collaboration with Large Language Models](https://project-roco.github.io/)
> Final Project — Intelligent Systems Course

---

## Abstract

RoCo uses LLM-based dialog to coordinate multiple robotic arms. However,
LLMs hallucinate — proposing actions that violate physical constraints
(e.g., reaching for objects outside an arm's workspace). We introduce a
reinforcement learning (RL) layer that constrains the LLM's open-ended
output to a **verified legal action set**, turning the problem from
"open-ended question answering" into "multiple choice."

Key results on the Sort task (3 arms, 7 panels, 3 cubes):
- LLM-only (GLM-4-flash): **0% success** (0/3 seeds)
- RL + action mask (ours): **100% success**, 3.6 steps avg
- Hybrid (LLM candidates + RL selection): **90–100%** depending on prompt
- Removing the mask during training: success collapses from 100% → 1.5%

---

## 1. Motivation

The RoCo framework lets N robotic arms collaborate via multi-round LLM
dialog. Each round, the LLM proposes a joint action plan (e.g., "Alice:
PICK blue_square PLACE panel3"). The plan is then executed via IK + RRT.

**Problem**: The LLM frequently proposes physically impossible actions:
- Picking objects outside an arm's reach
- Placing cubes on panels that violate task rules
- Proposing all-WAIT when progress is needed

Our measured hallucination rate: **1.67 illegal actions per decision step**
(GLM-4-flash on the Sort task without constraints).

---

## 2. Approach: Factorized Discrete Actions + Legality Mask

### 2.1 Action Space

Each agent i selects a high-level action per step:

```
a_i ∈ {WAIT} ∪ {(verb, target) | verb ∈ V_task, target ∈ T_task}
```

Encoded as two softmax heads (verb/object × target), with a **boolean
legality mask** that sets logits of illegal combinations to −∞.

### 2.2 Legality Mask

Per-agent, per-step mask derived from:
- Arm reach ranges (which panels/objects are physically accessible)
- Task rules (which cube-panel pairings are allowed)
- Current state (can't pick if already holding; can't re-pick a finished cube)

The mask is the core contribution — it converts the LLM's open-ended
action space into a constrained "multiple choice" problem.

### 2.3 State Space

Flat vector per agent:
- Object positions (one-hot panel encoding)
- Holding flags
- Task progress (which sub-goals are achieved)
- Agent identity one-hot

Global state (for centralized critic): concatenation of all agents' features.

### 2.4 Reward Function

| Component | Value | Purpose |
|-----------|-------|---------|
| Sub-goal achieved | +10 | First time a cube reaches its target |
| All goals done | +50 | Episode success bonus |
| Potential shaping | γΦ(s') − Φ(s) | Distance-based progress signal |
| Handoff bonus | +2 | Moving a cube into another agent's reach |
| Invalid action | −2 | Backup penalty (mask should prevent these) |
| Per-step cost | −0.05 | Encourage efficiency |

### 2.5 Training Algorithm

**MAPPO** (Multi-Agent PPO) with:
- Parameter-shared actor across agents
- Centralized critic (sees global state)
- MaskedCategorical distribution for action sampling
- 25k environment steps to convergence (~30 seconds on CPU)

---

## 3. LLM + RL Hybrid Policy

Architecture:
1. LLM receives scene description → proposes one EXECUTE plan
2. Parser extracts per-agent candidate action IDs
3. Candidates ∩ legality mask → narrowed mask
4. RL policy samples from narrowed mask

Anti-hallucination guard: if LLM proposes all-WAIT, ignore candidates
entirely (fall back to full legality mask).

**Mask-aware variant**: Include the legal action set in the LLM prompt.
Result: hallucination rate drops 4.7× (1.67 → 0.35 illegal/step).

---

## 4. Experimental Results

### 4.1 Main Comparison (Sort task, symbolic env, 200 episodes)

| Method | Success | Avg Steps |
|--------|--------:|----------:|
| LLM-only (GLM-4-flash) | 0% | timeout |
| Random + mask | 65.5% | 22.1 |
| Scripted heuristic | 22.0% | 24.3 |
| **MAPPO + mask (ours)** | **100%** | **3.6** |
| Hybrid (vanilla) | 62.5% | 14.5 |
| Hybrid (mask-aware) | **100%** | 4.25 |

### 4.2 Ablation Study

| Config | Success | Notes |
|--------|--------:|-------|
| Full (mask + shaping + handoff) | 100% | Baseline |
| No mask (training only) | **1.5%** | Policy learns to "cheat" |
| No shaping | 100% | Slower convergence only |
| No handoff | 100% | Slower convergence only |

### 4.3 Cross-Task Generalization

Same MAPPO code, different task (Sweep: 2 agents, MOVE/SWEEP/DUMP):
- **100% success, 7.24 steps avg** (30k steps, 21s training)
- Zero modifications to training code — only a new symbolic env class

### 4.4 Real MuJoCo Evaluation (Sort task, 5 seeds)

| Config | Success | Avg Steps |
|--------|--------:|----------:|
| Default | 40% (2/5) | 6.8 |
| With snap-finished | **60% (3/5)** | 5.6 |

Remaining failures: IK/RRT solver limitations in multi-arm configurations.

### 4.5 Cross-Task LLM-only Comparison

| Task | LLM-only (GLM-4-flash) | RL+mask (ours) |
|------|------------------------:|---------------:|
| Sort (3 arms) | 0% (0/3) | 100% |
| Sweep (2 arms) | 0% (0/1+) | 100% |

Both tasks show the same pattern: the LLM proposes actions that violate
task constraints, gets rejected by environment feedback, but fails to
correct itself within the step budget. The RL policy, constrained by the
mask, never proposes an illegal action and converges to optimal cooperation
in minutes of training.

---

## 5. Key Findings

1. **The mask is the method, not an optimization.** Removing it during
   training causes the policy to learn exploits that collapse at evaluation.

2. **Constraints are shareable auxiliary information.** The same legality
   mask helps both RL (during training) and LLM (during inference). Showing
   the mask to the LLM cuts hallucinations 4.7×.

3. **Scripted rules can't handle multi-hop cooperation.** The Sort task
   requires 3-step relay chains that no hand-written heuristic captures.
   Learned policies discover these automatically.

4. **The framework generalizes across tasks.** Changing the task requires
   only filling in a new vocab table and mask function — no changes to the
   actor, critic, or training loop.

---

## 6. Repository Structure

```
rocobench/rl/
├── action_codec.py          # Factorized action encoding
├── sort_symbolic_env.py     # Fast symbolic Sort env for training
├── sweep_symbolic_env.py    # Fast symbolic Sweep env for training
├── mappo.py                 # MAPPO with masked categorical
├── baselines.py             # Random / scripted baselines
├── llm_hybrid.py            # LLM + RL hybrid policy
├── real_env_bridge.py       # EnvState ↔ RL features (Sort)
└── sweep_real_env_bridge.py # EnvState ↔ RL features (Sweep)

train_rl_sort.py             # Train on Sort
train_rl_sweep.py            # Train on Sweep
run_rl_sort.py               # Real-env runner (Sort)
run_rl_sweep.py              # Real-env runner (Sweep)
benchmark_rl.py              # Full comparison table
benchmark_mask_aware.py      # Mask-aware LLM ablation
make_plots.py                # Generate presentation figures
make_training_curves.py      # Training curve comparison
```

---

## 7. Reproduction

```bash
# Train (CPU, ~30s each)
python train_rl_sort.py --steps 30000
python train_rl_sweep.py --steps 30000

# Evaluate
python benchmark_rl.py --train-steps 25000 --eval-episodes 200
python benchmark_mask_aware.py --episodes 8

# Real MuJoCo
python run_rl_sort.py --method mappo --load checkpoints/sort_mappo.pt \
    --num-runs 5 --snap-finished

# Generate figures
python make_plots.py
python make_training_curves.py
```
