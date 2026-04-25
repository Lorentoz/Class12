"""Simulated annealing for RackLayout optimization."""
from __future__ import annotations

import math
import random
from typing import Tuple, List

from rack_layout import RackLayout


def simulated_annealing(
    initial: RackLayout,
    T0: float = 1.0,
    alpha: float = 0.995,
    max_iters: int = 5000,
):
    """Perform simulated annealing; returns (best_layout, history_obj).

    Cooling schedule: T <- alpha * T
    Acceptance: Metropolis criterion
    """
    current = initial.copy()
    best = current.copy()
    T = T0
    history = [current.objective()]
    for it in range(max_iters):
        # Propose random neighbor (single-rack random move)
        candidate = current.mutate()
        delta = candidate.objective() - current.objective()
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            current = candidate
            if current.objective() < best.objective():
                best = current.copy()
        history.append(best.objective())
        T *= alpha
        if T < 1e-6:
            break
    return best, history


if __name__ == "__main__":
    from rack_layout import RackLayout
    init = RackLayout()
    best, hist = simulated_annealing(init, T0=1.0, alpha=0.995, max_iters=2000)
    print("Initial obj:", init.objective())
    print("Best obj:", best.objective())
