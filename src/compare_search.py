"""Compare UCS and A* search algorithms on warehouse pathfinding tasks.

Runs both algorithms on 10 randomized warehouse configurations, computing
paths from start → pickup and pickup → dropoff. Generates comparison plots
and summary statistics.

Usage:
    python src/compare_search.py --trials 10 --seed 0 --save out/search_compare.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from warehouse_env import WarehouseEnv
from ucs_pathfinder import uniformcost_search
from astar_pathfinder import astar_search

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for compatibility
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    plt = None  # type: ignore
    np = None  # type: ignore
    print(f"Warning: Could not import matplotlib/numpy: {e}")


def run_comparison_trial(env: WarehouseEnv, trial_id: int) -> Dict:
    """Run one trial: UCS and A* on start→pickup and pickup→dropoff."""
    obs = env.reset(randomize=True)
    start = obs["robot_pos"]
    pickup = obs["pickup_pos"]
    dropoff = obs["dropoff_pos"]

    results = {
        "trial_id": trial_id,
        "start": start,
        "pickup": pickup,
        "dropoff": dropoff,
        "ucs": {},
        "astar": {},
    }

    # Phase 1: start → pickup
    ucs_path_1, ucs_stats_1 = uniformcost_search(start, pickup, env)
    astar_path_1, astar_stats_1 = astar_search(start, pickup, env)

    results["ucs"]["phase1"] = {
        "path_length": len(ucs_path_1) if ucs_path_1 else None,
        "nodes_expanded": ucs_stats_1["nodes_expanded"],
        "frontier_peak": ucs_stats_1["frontier_peak"],
        "time_ms": ucs_stats_1["time_ms"],
    }
    results["astar"]["phase1"] = {
        "path_length": len(astar_path_1) if astar_path_1 else None,
        "nodes_expanded": astar_stats_1["nodes_expanded"],
        "frontier_peak": astar_stats_1["frontier_peak"],
        "time_ms": astar_stats_1["time_ms"],
    }

    # Phase 2: pickup → dropoff
    ucs_path_2, ucs_stats_2 = uniformcost_search(pickup, dropoff, env)
    astar_path_2, astar_stats_2 = astar_search(pickup, dropoff, env)

    results["ucs"]["phase2"] = {
        "path_length": len(ucs_path_2) if ucs_path_2 else None,
        "nodes_expanded": ucs_stats_2["nodes_expanded"],
        "frontier_peak": ucs_stats_2["frontier_peak"],
        "time_ms": ucs_stats_2["time_ms"],
    }
    results["astar"]["phase2"] = {
        "path_length": len(astar_path_2) if astar_path_2 else None,
        "nodes_expanded": astar_stats_2["nodes_expanded"],
        "frontier_peak": astar_stats_2["frontier_peak"],
        "time_ms": astar_stats_2["time_ms"],
    }

    return results


def aggregate_results(trials: List[Dict]) -> Dict:
    """Aggregate results across trials into summary statistics."""
    ucs_nodes = []
    astar_nodes = []
    ucs_times = []
    astar_times = []
    ucs_path_lens = []
    astar_path_lens = []

    for trial in trials:
        for phase in ["phase1", "phase2"]:
            ucs_nodes.append(trial["ucs"][phase]["nodes_expanded"])
            astar_nodes.append(trial["astar"][phase]["nodes_expanded"])
            ucs_times.append(trial["ucs"][phase]["time_ms"])
            astar_times.append(trial["astar"][phase]["time_ms"])
            if trial["ucs"][phase]["path_length"] is not None:
                ucs_path_lens.append(trial["ucs"][phase]["path_length"])
            if trial["astar"][phase]["path_length"] is not None:
                astar_path_lens.append(trial["astar"][phase]["path_length"])

    import statistics

    return {
        "ucs": {
            "mean_nodes_expanded": statistics.mean(ucs_nodes),
            "mean_time_ms": statistics.mean(ucs_times),
            "mean_path_length": statistics.mean(ucs_path_lens) if ucs_path_lens else None,
        },
        "astar": {
            "mean_nodes_expanded": statistics.mean(astar_nodes),
            "mean_time_ms": statistics.mean(astar_times),
            "mean_path_length": statistics.mean(astar_path_lens) if astar_path_lens else None,
        },
        "num_trials": len(trials),
        "total_phases": len(ucs_nodes),
    }


def plot_comparison(trials: List[Dict], summary: Dict, out_png: str | None = None):
    """Create comparison plots: nodes expanded, time, path lengths."""
    if plt is None or np is None:
        print("matplotlib/numpy not available; skipping plots")
        return

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Subplot 1: Mean nodes expanded
    methods = ["UCS", "A*"]
    nodes = [summary["ucs"]["mean_nodes_expanded"], summary["astar"]["mean_nodes_expanded"]]
    axs[0].bar(methods, nodes, color=["#4c72b0", "#55a868"])
    axs[0].set_ylabel("Mean nodes expanded")
    axs[0].set_title("Search Efficiency: Nodes Expanded")
    axs[0].grid(axis="y", alpha=0.3)

    # Subplot 2: Mean time
    times = [summary["ucs"]["mean_time_ms"], summary["astar"]["mean_time_ms"]]
    axs[1].bar(methods, times, color=["#4c72b0", "#55a868"])
    axs[1].set_ylabel("Time (ms)")
    axs[1].set_title("Computation Time")
    axs[1].grid(axis="y", alpha=0.3)

    # Subplot 3: Mean path length (should be equal if both optimal)
    paths = [summary["ucs"]["mean_path_length"], summary["astar"]["mean_path_length"]]
    if all(p is not None for p in paths):
        axs[2].bar(methods, paths, color=["#4c72b0", "#55a868"])
        axs[2].set_ylabel("Mean steps")
        axs[2].set_title("Solution Quality: Mean Path Length")
        axs[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png)
        print(f"Saved comparison plot to {out_png}")
    else:
        plt.show()


def print_summary_table(trials: List[Dict], summary: Dict):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SEARCH ALGORITHM COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Total trials: {summary['num_trials']}")
    print(f"Total search tasks (phases): {summary['total_phases']}")
    print("-" * 80)
    print(f"{'Metric':<30} {'UCS':<20} {'A*':<20}")
    print("-" * 80)
    print(
        f"{'Mean nodes expanded':<30} {summary['ucs']['mean_nodes_expanded']:<20.1f} {summary['astar']['mean_nodes_expanded']:<20.1f}"
    )
    print(
        f"{'Mean computation time (ms)':<30} {summary['ucs']['mean_time_ms']:<20.4f} {summary['astar']['mean_time_ms']:<20.4f}"
    )
    if summary["ucs"]["mean_path_length"] and summary["astar"]["mean_path_length"]:
        print(
            f"{'Mean path length (steps)':<30} {summary['ucs']['mean_path_length']:<20.1f} {summary['astar']['mean_path_length']:<20.1f}"
        )
    print("-" * 80)
    improvement = (
        (summary["ucs"]["mean_nodes_expanded"] - summary["astar"]["mean_nodes_expanded"])
        / summary["ucs"]["mean_nodes_expanded"]
        * 100
    )
    print(f"A* efficiency gain: {improvement:.1f}% fewer nodes expanded than UCS")
    print("=" * 80 + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="")
    args = p.parse_args(argv)

    print(f"Running {args.trials} comparison trials...")
    trials = []
    for i in range(args.trials):
        env = WarehouseEnv()
        trial = run_comparison_trial(env, trial_id=i)
        trials.append(trial)
        print(
            f"  Trial {i}: UCS nodes={trial['ucs']['phase1']['nodes_expanded'] + trial['ucs']['phase2']['nodes_expanded']}, "
            f"A* nodes={trial['astar']['phase1']['nodes_expanded'] + trial['astar']['phase2']['nodes_expanded']}"
        )

    summary = aggregate_results(trials)
    print_summary_table(trials, summary)

    plot_comparison(trials, summary, out_png=(args.save or None))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
