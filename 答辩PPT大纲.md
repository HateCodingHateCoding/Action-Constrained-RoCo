# 答辩 PPT 大纲

> 项目：智能系统期末项目 — RoCo 复现 + RL 扩展
> 大纲面向 ~12 页 PPT，每页配讲话点 (talking points)。
> 所有图都在 [figures/](figures/) 里，可以直接拖进 PPT。

---

## 第 1 页 · 标题页

**多机械臂 LLM 协作 + 强化学习扩展**
"把 LLM 的开放问答变成在合法集合里做选择题"

讲话点：
- 一句话开场：让 LLM 协调机器人时它会幻觉，我们用 RL 屏蔽幻觉。
- 团队分工 + 我们这一组负责的 RL 扩展。

---

## 第 2 页 · 问题背景

**RoCo baseline 的痛点**

讲话点：
- LLM 让多个机械臂用对话协作（panel 排序、扫地、抓物 …）。
- 痛点：LLM 会"幻觉"——例如让 Alice 抓她够不到的方块。
- **我们实测**：用 GLM-4-flash 跑 RoCo 原版 sort 任务，3 个 seed 全失败。
  典型失败：Alice 反复尝试把 pink_polygon 放到 panel2（应该是 panel4）。

**配图**：[figures/fig_comparison.png](figures/fig_comparison.png) 右半（步数）
（也可以贴一段失败的对话日志做截图）

---

## 第 3 页 · 我们的扩展思路

```
+---------+           +-------+           +-----------+
| 场景观测 +---------> | 高层  | -选动作-> | IK + RRT  |
+---------+           | 策略  |           | 底层执行  |
                      +-------+           +-----------+
                          ^
       +------------------+
       | LLM 候选 + 合法性 mask + 协作奖励
       +------------------+
```

讲话点：
- 思路：在 LLM 之上加 RL 高层策略，**只让 RL 在合法动作集合里挑**。
- 这是把"开放问答"变成"选择题"——LLM 给候选，RL 选最稳的那一个。
- 底层 IK + RRT 完全不动，RL 模块完全独立、可插拔。

---

## 第 4 页 · 动作空间设计

**因子化离散动作 + 合法性 mask**

```
a_i = (verb, target)
verb_idx ∈ {WAIT, ...}, target_idx ∈ {panels / cubes / ...}
两个 head 分别 softmax，再用 mask 把非法位 logit 设为 -∞
```

讲话点：
- 动作空间不是连续关节角，而是**高层离散**：拣什么、放哪里。
- 用因子化两个 head 避免动作空间爆炸。
- mask 自动把"够不到的物体""不该放的目标"等屏蔽掉——这就是"选择题"的具象。
- 这个模板对所有 RoCo 任务通用，每个任务只填三张表（动词、物体、目标）。

---

## 第 5 页 · 状态 + 奖励

讲话点：
- 状态：每个 agent 的位置、朝向、抓取 + 物体位置 + 任务进度 + 合法 mask。
- 奖励三层：
  - **任务**：达到子目标 +10、全部完成 +50；
  - **塑形**：potential-based，物体离目标越近越奖励；
  - **协作**：把别人够不到的物体挪到他可达 +2；
  - **约束**：非法动作 −2（mask 已屏蔽，惩罚是双保险）。
- 算法：MAPPO（多智能体 PPO，CTDE 中心化训练 / 分布式执行）。

---

## 第 6 页 · 主结果（对比实验）

**所有方法在 sort 任务上的对比**

| 方法 | 成功率 | 平均步数 |
| --- | ---: | ---: |
| LLM-only (GLM-4-flash) | 0% (0/3) | timeout |
| 随机 + mask | 65.5% | 22.1 |
| Scripted 启发式 | 22% | 24.3 |
| **RL+mask（我们的）** | **100%** | **3.6** |
| Hybrid (LLM+RL, vanilla) | 62.5% | 14.5 |
| Hybrid (LLM+RL, mask-aware) | **100%** | 4.25 |

**配图**：[figures/fig_comparison.png](figures/fig_comparison.png)

讲话点：
- 横向对比 5 种方法，**我们的方法 RL+mask 最强：100% 成功，3.6 步内完成**。
- LLM-only 直接挂——动机有数据。
- 即使最简单的随机策略，**只要有 mask** 也能拿 65%——侧面说明约束有多重要。
- 加 mask-aware 后混合策略也能到 100%——后面会展开。

---

## 第 7 页 · RL 消融实验：mask 是核心

**配图**：[figures/fig_ablation.png](figures/fig_ablation.png)

讲话点：
- 拆掉 shaping 或 handoff 奖励：成功率 100% 不变（只让收敛慢一点）。
- **拆掉 mask 训练：成功率 100% → 1.5%**——崩了。
- 解释：训练时无约束 → RL 学到"作弊"策略 → 上线被合法性约束屏蔽 → 直接挂。
- **结论：mask 不是优化项，是核心方法本身**。这条数据是"做选择题"假设的实证。

---

## 第 8 页 · 训练曲线（收敛速度）

**配图**：[figures/fig_curves.png](figures/fig_curves.png)

讲话点：
- 同 25k 步预算，mask 让收敛快 ~4 倍（1.3k → 5.0k 步达 80% 成功率）。
- no_mask 的训练曲线"看起来"也到了 100%，因为它在自己的"无约束训练环境"里
  就是高分；但**评估环境是合法的**，所以上线时它会崩。
- shaping/handoff 让收敛更稳定，但不是决定胜负的。

---

## 第 9 页 · LLM+RL 混合策略

**Mask-aware 的提示工程**

```
[vanilla]          告诉 LLM 任务规则 → LLM 经常幻觉
[mask-aware]       同时把"这一轮各机器人可选的合法动作集"喂给 LLM
```

8 局对比：

| 提示策略 | 成功率 | LLM 非法动作/步 |
| --- | ---: | ---: |
| vanilla | 62.5% | **1.67** |
| mask-aware | **100%** | **0.35**（4.7×↓） |

**配图**：[figures/fig_mask_aware.png](figures/fig_mask_aware.png)

讲话点：
- 同样的 mask，**既让 RL 训得好，也能让 LLM 少幻觉**。
- 这是个更深的论点：**约束是可共享的辅助信息**——不是只服务于 RL。
- 也是工程教训：盲信 LLM 输出（如 vanilla 模式）不可取，反幻觉守卫不可省。

---

## 第 10 页 · 跨任务通用性 — sweep

**配图**：[figures/fig_cross_task.png](figures/fig_cross_task.png)

```
sort  : 3 agents, 7 panels, PICK+PLACE
sweep : 2 agents, 3 cubes, MOVE+SWEEP+DUMP
```

讲话点：
- 换任务时**训练代码（mappo.py）一行没改**，只填了 SweepSymbolicEnv。
- sweep 任务：30k 步训完（21 秒）→ 100% 成功，平均 7.24 步。
- 学到的策略还**自发发现流水线**：t=2 Alice 在 dump 时 Bob 已经在去下个方块。
- 这是"框架统一、词表分任务、mask 按任务写规则"的实证。

---

## 第 11 页 · 真实 MuJoCo 端到端

**Sort 任务 5 seed 真实物理仿真**

| 配置 | 成功率 | 平均步数 |
| --- | ---: | ---: |
| 默认 | 40% (2/5) | 6.8 |
| **加 `--snap-finished`** | **60% (3/5)** | 5.6 |

讲话点：
- 真实 MuJoCo 上不会 100% — 有 30%~40% 的物理 / RRT 失败缺口。
- 失败原因诊断清楚：放置漂移、多臂 RRT 求解失败。
- 我们加了 `--snap-finished` 工程修复（不动 env 语义，只补救物理）：从 40% → 60%。
- 剩余 30% 缺口是"复现+迁移"组的 IK 调优范围，不是 RL 问题。
- demo 视频：[data/rl_real_runs_v4/run_1/rollout.mp4](data/rl_real_runs_v4/run_1/rollout.mp4)
  里面能清楚看到三只手臂跨臂中转的协作动作。

---

## 第 12 页 · 总结 + 未来工作

讲话点：

**三个核心论点 + 实证**：
1. LLM-only 在 sort 任务上 0% — 项目动机；
2. 训练时去 mask 让成功率从 100% → 1.5% — 核心方法；
3. 把 mask 喂给 LLM 让幻觉率从 1.67/步 → 0.35/步 — 约束可共享。

**已交付**：
- 一套通用的"vocab + mask + 奖励"模板，已在 sort、sweep 上验证；
- MAPPO 实现 + LLM+RL 混合策略 + 4 种基线；
- 全套对比 / 消融实验 + 答辩图；
- 真实 MuJoCo 端到端能跑（含 demo 视频）。

**未来可做**：
- 真实环境 IK / RRT 调优把成功率推到 80%+；
- 接 GLM-4.6 等更强模型重做 hybrid 实验；
- 用 mask-aware 提示工程做更细粒度的"LLM 学约束"研究。

---

## 附录 · 一键复现命令

```powershell
# 训练
python train_rl_sort.py --steps 30000 --save checkpoints/sort_mappo.pt
python train_rl_sweep.py --steps 30000 --save checkpoints/sweep_mappo.pt

# 评估
python benchmark_rl.py --train-steps 25000 --eval-episodes 200       # 主对比表
python benchmark_mask_aware.py --episodes 8                          # mask-aware 消融
python make_training_curves.py --steps 25000                         # 收敛曲线
python make_plots.py                                                 # 4 张答辩图

# 真实 MuJoCo
python run_rl_sort.py  --method mappo --load checkpoints/sort_mappo.pt --num-runs 5 --snap-finished
python run_rl_sweep.py --load checkpoints/sweep_mappo.pt --num-runs 3
```
