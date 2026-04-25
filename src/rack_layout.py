"""State representation and objective function for rack placement.

State: list of 20 unique (x,y) integer positions on a 20x20 grid.
Depot fixed at (10, 10) by default.

Objective: f(s) = (1/20) * sum_i d(depot, rack_i) + lambda * congestion_penalty
where congestion_penalty = count of racks with Manhattan distance to depot < 5
and lambda = 2.0
Lower is better.
"""
from __future__ import annotations

import random
from typing import List, Tuple

Position = Tuple[int, int]


def default_depot() -> Position:
    return (10, 10)


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class RackLayout:
    def __init__(self, positions: List[Position] | None = None, grid_size: int = 20, depot: Position | None = None):
        self.grid_size = grid_size
        self.depot = depot or default_depot()
        if positions is None:
            self.positions = self.random_positions()
        else:
            self.positions = positions

    def random_positions(self) -> List[Position]:
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        all_cells.remove(self.depot) if self.depot in all_cells else None
        chosen = random.sample(all_cells, 20)
        return chosen

    def objective(self, lam: float = 2.0) -> float:
        # average distance to depot
        distances = [manhattan(self.depot, p) for p in self.positions]
        avg = sum(distances) / len(distances)
        # congestion penalty: count racks within distance < 5
        congestion = sum(1 for d in distances if d < 5)
        return (avg) + lam * congestion / 1.0

    def copy(self) -> "RackLayout":
        return RackLayout(positions=list(self.positions), grid_size=self.grid_size, depot=self.depot)

    def neighbors(self) -> List["RackLayout"]:
        """Generate neighbors by moving one rack by +-1 in x or y, maintaining uniqueness and bounds."""
        neighs: List[RackLayout] = []
        for i, (x, y) in enumerate(self.positions):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                if (nx, ny) in self.positions:
                    continue
                newpos = list(self.positions)
                newpos[i] = (nx, ny)
                neighs.append(RackLayout(positions=newpos, grid_size=self.grid_size, depot=self.depot))
        return neighs

    def mutate(self) -> "RackLayout":
        # random move of one rack to a random empty cell
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        occupied = set(self.positions)
        choices = [c for c in all_cells if c not in occupied and c != self.depot]
        if not choices:
            return self.copy()
        i = random.randrange(len(self.positions))
        newpos = list(self.positions)
        newpos[i] = random.choice(choices)
        return RackLayout(positions=newpos, grid_size=self.grid_size, depot=self.depot)


def pretty_print(layout: RackLayout) -> None:
    grid = [["." for _ in range(layout.grid_size)] for _ in range(layout.grid_size)]
    dr, dc = layout.depot
    grid[dr][dc] = "X"
    for x, y in layout.positions:
        grid[x][y] = "R"
    for row in grid:
        print("".join(row))
