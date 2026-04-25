"""Compare local-search algorithms (hill-climbing, simulated annealing, genetic).

Runs multiple randomized starts per algorithm, plots convergence curves, and
saves best-final layout visualizations.
"""
from __future__ import annotations

import os
import random
import statistics
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rack_layout import RackLayout
from hill_climbing import steepest_ascent
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT_DIR, exist_ok=True)


def pad_history(hist: List[float], length: int) -> List[float]:
    if len(hist) >= length:
        return hist[:length]
    return hist + [hist[-1]] * (length - len(hist))


def render_layout(layout: RackLayout, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [p[0] for p in layout.positions]
    ys = [p[1] for p in layout.positions]
    ax.scatter(xs, ys, c="tab:blue", label="racks", s=40)
    ax.scatter([layout.depot[0]], [layout.depot[1]], c="tab:red", marker="X", s=80, label="depot")
    ax.set_xlim(-0.5, layout.grid_size - 0.5)
    ax.set_ylim(-0.5, layout.grid_size - 0.5)
    ax.set_title(f"Objective: {layout.objective():.2f}")
    ax.set_aspect('equal')
    ax.legend(loc="upper right")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)


def run_comparison(num_starts: int = 20, seed: int = 0):
    random.seed(seed)
    hc_histories = []
    sa_histories = []
    ga_histories = []
    hc_bests = []
    sa_bests = []
    ga_bests = []

    for i in range(num_starts):
        init = RackLayout()
        # Hill-climbing
        hc_best, hc_hist = steepest_ascent(init, max_iters=1000)
        hc_histories.append(hc_hist)
        hc_bests.append(hc_best)

        # Simulated annealing
        sa_best, sa_hist = simulated_annealing(init, T0=1.0, alpha=0.995, max_iters=2000)
        sa_histories.append(sa_hist)
        sa_bests.append(sa_best)

        # Genetic algorithm (population-randomized)
        ga_best, ga_hist = genetic_algorithm(pop_size=30, generations=200)
        ga_histories.append(ga_hist)
        ga_bests.append(ga_best)

    # Determine max length for padding
    maxlen = max(max(len(h) for h in hc_histories), max(len(h) for h in sa_histories), max(len(h) for h in ga_histories))

    # Pad and compute means
    hc_padded = [pad_history(h, maxlen) for h in hc_histories]
    sa_padded = [pad_history(h, maxlen) for h in sa_histories]
    ga_padded = [pad_history(h, maxlen) for h in ga_histories]

    hc_mean = [statistics.mean(col) for col in zip(*hc_padded)]
    sa_mean = [statistics.mean(col) for col in zip(*sa_padded)]
    ga_mean = [statistics.mean(col) for col in zip(*ga_padded)]

    iters = list(range(len(hc_mean)))

    plt.figure(figsize=(8, 5))
    plt.plot(iters, hc_mean, label="Hill-climbing")
    plt.plot(iters, sa_mean, label="Simulated Annealing")
    plt.plot(iters, ga_mean, label="Genetic Algorithm")
    plt.xlabel("Iteration")
    plt.ylabel("Objective (lower is better)")
    plt.legend()
    plt.title(f"Local Search Convergence (n_starts={num_starts})")
    plt.tight_layout()
    conv_path = os.path.join(OUT_DIR, "local_search_convergence.png")
    plt.savefig(conv_path)
    plt.close()

    # Save best layouts per algorithm (best across runs)
    best_hc = min(hc_bests, key=lambda s: s.objective())
    best_sa = min(sa_bests, key=lambda s: s.objective())
    best_ga = min(ga_bests, key=lambda s: s.objective())

    render_layout(best_hc, os.path.join(OUT_DIR, "best_hill_climbing.png"))
    render_layout(best_sa, os.path.join(OUT_DIR, "best_simulated_annealing.png"))
    render_layout(best_ga, os.path.join(OUT_DIR, "best_genetic_algorithm.png"))

    print(f"Saved convergence plot to {conv_path}")
    print(f"Saved best layouts to {OUT_DIR}")


if __name__ == "__main__":
    run_comparison(num_starts=20, seed=0)
