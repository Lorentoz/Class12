"""Compare reflex and greedy agents over many episodes and visualize results.

Usage (from project root):
    python src/compare_agents.py --episodes 50 --seed 0 --save out/compare.png --randomize

Produces a figure with:
 - Bar chart of success rates
 - Box plots of episode lengths (successful episodes)
 - Histograms of final battery levels

Returns statistics dictionary for both agents.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from warehouse_env import WarehouseEnv
from warehouse_agent_greedy import GreedyManhattanAgent
from warehouse_agent_reflex import ReflexAgent

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


def run_n_episodes(agent_cls, num_episodes: int = 50, seed: int | None = None, randomize: bool = False):
    results = {
        "success": [],
        "length": [],
        "battery": [],
        "reward": [],
    }
    for i in range(num_episodes):
        # Seed random for reproducible environment randomization
        s = None if seed is None else seed + i
        if s is not None:
            random.seed(s)
        env = WarehouseEnv()
        agent = agent_cls(seed=s) if agent_cls is ReflexAgent or agent_cls is GreedyManhattanAgent else agent_cls()
        agent.reset()
        obs = env.reset(randomize=randomize)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        while steps < env.max_steps:
            act = agent.select_action(obs, env)
            obs, r, terminated, truncated, info = env.step(act)
            total_reward += r
            steps += 1
            if terminated or truncated:
                break
        results["success"].append(bool(terminated))
        results["length"].append(steps)
        results["battery"].append(env.state.battery)
        results["reward"].append(total_reward)
    return results


def analyze_and_plot(resA: Dict[str, List], resB: Dict[str, List], labels=("Reflex", "Greedy"), out_png: str | None = None):
    # Compute success rates
    import statistics

    stats = {}
    for name, r in zip(labels, (resA, resB)):
        successes = r["success"]
        lengths = r["length"]
        battery = r["battery"]
        rewards = r["reward"]
        succ_rate = 100.0 * sum(1 for s in successes if s) / len(successes)
        # lengths of successful episodes
        succ_lengths = [l for l, s in zip(lengths, successes) if s]
        stats[name] = {
            "success_rate": succ_rate,
            "mean_length_success": statistics.mean(succ_lengths) if succ_lengths else None,
            "median_length_success": statistics.median(succ_lengths) if succ_lengths else None,
            "lengths_all": lengths,
            "battery_all": battery,
            "rewards_all": rewards,
            "successes": successes,
        }

    if plt is None:
        print("matplotlib not available; skipping plots")
        return stats

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))

    # Subplot 1: success rates
    axs[0].bar(labels, [stats[labels[0]]["success_rate"], stats[labels[1]]["success_rate"]], color=["#4c72b0", "#55a868"])
    axs[0].set_ylabel("Success rate (%)")
    axs[0].set_title("Success Rate")

    # Subplot 2: boxplot of episode lengths (successful episodes)
    lengths_ref = [l for l, s in zip(resA["length"], resA["success"]) if s]
    lengths_gre = [l for l, s in zip(resB["length"], resB["success"]) if s]
    axs[1].boxplot([lengths_ref or [0], lengths_gre or [0]], labels=labels)
    axs[1].set_title("Episode lengths (successful episodes)")
    axs[1].set_ylabel("Steps")

    # Subplot 3: histograms of final battery
    axs[2].hist(resA["battery"], alpha=0.6, label=labels[0])
    axs[2].hist(resB["battery"], alpha=0.6, label=labels[1])
    axs[2].set_title("Final battery distributions")
    axs[2].set_xlabel("Battery")
    axs[2].legend()

    plt.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png)
        print(f"Saved comparison plot to {out_png}")
    else:
        plt.show()
    return stats


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--save", type=str, default="")
    args = p.parse_args(argv)

    print(f"Running {args.episodes} episodes per agent (randomize={args.randomize})")
    res_ref = run_n_episodes(ReflexAgent, num_episodes=args.episodes, seed=args.seed, randomize=args.randomize)
    res_gre = run_n_episodes(GreedyManhattanAgent, num_episodes=args.episodes, seed=args.seed, randomize=args.randomize)

    stats = analyze_and_plot(res_ref, res_gre, labels=("Reflex", "Greedy"), out_png=(args.save or None))

    # Print summary
    for name in ("Reflex", "Greedy"):
        s = stats[name]
        print(f"{name}: success_rate={s['success_rate']:.1f}%, mean_length_success={s['mean_length_success']}, median_length_success={s['median_length_success']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
