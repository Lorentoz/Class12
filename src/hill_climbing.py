"""Steepest-ascent hill-climbing for RackLayout optimization."""
from __future__ import annotations

import copy
import math
from typing import Tuple, List

from rack_layout import RackLayout


def steepest_ascent(initial: RackLayout, max_iters: int = 1000) -> Tuple[RackLayout, List[float]]:
    """Perform steepest-ascent hill-climbing (minimization of objective).

    Returns final layout and list of objective values per iteration.
    """
    current = initial.copy()
    history = [current.objective()]
    for it in range(max_iters):
        neighs = current.neighbors()
        if not neighs:
            break
        # Evaluate neighbors and pick the one with lowest objective
        best = min(neighs, key=lambda s: s.objective())
        if best.objective() < current.objective():
            current = best
            history.append(current.objective())
        else:
            break
    return current, history


if __name__ == "__main__":
    from rack_layout import RackLayout
    init = RackLayout()
    final, hist = steepest_ascent(init, max_iters=200)
    print("Initial obj:", init.objective())
    print("Final obj:", final.objective())
