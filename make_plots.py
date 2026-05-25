"""Generate answer-friendly comparison plots for the project slides.

Produces three PNGs in figures/:
  fig_comparison.png      - bar chart of success rate across all methods
  fig_ablation.png        - mask/shaping/handoff ablation
  fig_mask_aware.png      - mask-aware vs vanilla LLM hybrid

Numbers are pulled from the benchmark scripts' actual outputs (see
benchmark_rl.py, benchmark_all_methods.py, benchmark_mask_aware.py).

Usage:
    python make_plots.py
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)


def _bar_with_labels(ax, labels, values, fmt="{:.0%}", color=None,
                     ylabel="", ymax=None):
    bars = ax.bar(labels, values, color=color)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.02 * (ymax or max(values))),
                fmt.format(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_comparison():
    methods = ["LLM-only", "Random\n+mask", "Scripted",
               "RL+mask\n(ours)", "Hybrid\n(LLM+RL)"]
    success = [0.00, 0.655, 0.220, 1.000, 0.900]
    steps =   [8.00, 22.10, 23.30, 3.60,  7.10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#d65f5f", "#a0a0a0", "#a0a0a0", "#3b8bba", "#5fa55f"]
    _bar_with_labels(ax1, methods, success, fmt="{:.0%}",
                     color=colors, ylabel="Success rate", ymax=1.15)
    ax1.set_title("Sort task success rate")
    _bar_with_labels(ax2, methods, steps, fmt="{:.1f}",
                     color=colors, ylabel="Avg steps to solve",
                     ymax=max(steps) * 1.18)
    ax2.set_title("Sort task average episode length\n(lower is better)")
    fig.suptitle("Method comparison on Sort task", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_ablation():
    """Highlight: removing the mask during training collapses the policy.
    Shaping/handoff are smaller effects -> shows the mask is the load-bearer.
    """
    cfgs = ["full", "no\nmask", "no\nshaping", "no\nhandoff"]
    success = [1.000, 0.015, 1.000, 1.000]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    colors = ["#3b8bba", "#d65f5f", "#a0a0a0", "#a0a0a0"]
    _bar_with_labels(ax, cfgs, success, fmt="{:.1%}",
                     color=colors, ylabel="Success rate", ymax=1.15)
    ax.set_title("MAPPO ablation on Sort task\n(removing the mask during training collapses the policy)")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_mask_aware():
    """Mask-aware LLM vs vanilla LLM: dramatic drop in hallucination rate."""
    variants = ["LLM only sees\ntask hint\n(vanilla)",
                "LLM also sees\nlegal action set\n(mask-aware)"]
    illegal = [1.672, 0.353]      # illegal LLM actions per step
    success = [0.625, 1.000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#d65f5f", "#3b8bba"]
    _bar_with_labels(ax1, variants, illegal, fmt="{:.2f}",
                     color=colors, ylabel="Illegal LLM actions / step",
                     ymax=max(illegal) * 1.2)
    ax1.set_title("LLM hallucination rate")
    _bar_with_labels(ax2, variants, success, fmt="{:.0%}",
                     color=colors, ylabel="Success rate", ymax=1.15)
    ax2.set_title("Hybrid policy success rate")
    fig.suptitle("Showing the legality mask to the LLM cuts hallucinations 4.7×",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig_mask_aware.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_cross_task():
    """Same RL framework, different task: 100% on sweep too."""
    tasks = ["Sort\n(3 agents,\n7 panels)", "Sweep\n(2 agents,\n3 cubes)"]
    success = [1.0, 1.0]
    steps = [3.6, 7.24]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    _bar_with_labels(ax1, tasks, success, fmt="{:.0%}",
                     color=["#3b8bba", "#5fa55f"],
                     ylabel="Success rate", ymax=1.15)
    ax1.set_title("MAPPO success rate")
    _bar_with_labels(ax2, tasks, steps, fmt="{:.1f}",
                     color=["#3b8bba", "#5fa55f"],
                     ylabel="Avg steps", ymax=max(steps) * 1.2)
    ax2.set_title("Avg steps to solve")
    fig.suptitle(
        "Cross-task generalization: same training code, no modification",
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig_cross_task.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_comparison()
    fig_ablation()
    fig_mask_aware()
    fig_cross_task()
    print(f"\nDone. Figures saved to {os.path.abspath(OUTDIR)}")
