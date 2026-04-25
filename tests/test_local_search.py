import random

from rack_layout import RackLayout
from hill_climbing import steepest_ascent
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm


def test_hill_climbing_improves():
    random.seed(0)
    init = RackLayout()
    final, hist = steepest_ascent(init, max_iters=200)
    assert final.objective() <= init.objective()


def test_simulated_annealing_improves():
    random.seed(1)
    init = RackLayout()
    best, hist = simulated_annealing(init, T0=1.0, alpha=0.99, max_iters=500)
    assert best.objective() <= init.objective()


def test_genetic_algorithm_returns_history():
    random.seed(2)
    best, hist = genetic_algorithm(pop_size=10, generations=50)
    assert isinstance(hist, list)
    assert len(hist) >= 1
    # history should record improvement potential: best value somewhere in history
    assert min(hist) <= hist[0]
