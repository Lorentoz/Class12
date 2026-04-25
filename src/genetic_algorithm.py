"""Genetic algorithm for RackLayout optimization."""
from __future__ import annotations

import random
from typing import List, Tuple

from rack_layout import RackLayout


def crossover(parent1: RackLayout, parent2: RackLayout) -> RackLayout:
    # Order-based crossover: take first half from p1, fill remaining from p2 preserving order
    n = len(parent1.positions)
    cut = n // 2
    p1_prefix = parent1.positions[:cut]
    p2_rest = [p for p in parent2.positions if p not in p1_prefix]
    child_positions = p1_prefix + p2_rest
    # Ensure uniqueness and correct length
    if len(child_positions) != n:
        # fallback: random fill
        child_positions = parent1.random_positions()
    return RackLayout(positions=child_positions, grid_size=parent1.grid_size, depot=parent1.depot)


def tournament_selection(pop: List[RackLayout], k: int = 3) -> RackLayout:
    return min(random.sample(pop, k), key=lambda s: s.objective())


def genetic_algorithm(
    pop_size: int = 30,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
) -> Tuple[RackLayout, List[float]]:
    # Initialize population
    pop = [RackLayout() for _ in range(pop_size)]
    history = [min(p.objective() for p in pop)]
    for gen in range(generations):
        new_pop: List[RackLayout] = []
        while len(new_pop) < pop_size:
            if random.random() < crossover_rate:
                p1 = tournament_selection(pop)
                p2 = tournament_selection(pop)
                child = crossover(p1, p2)
            else:
                child = tournament_selection(pop).copy()
            if random.random() < mutation_rate:
                child = child.mutate()
            new_pop.append(child)
        pop = new_pop
        best = min(pop, key=lambda s: s.objective())
        history.append(best.objective())
    best = min(pop, key=lambda s: s.objective())
    return best, history


if __name__ == "__main__":
    best, hist = genetic_algorithm(pop_size=30, generations=200)
    print("Best obj:", best.objective())
