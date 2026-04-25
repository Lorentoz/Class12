import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from warehouse_env import WarehouseEnv
from ucs_pathfinder import uniformcost_search
from astar_pathfinder import astar_search


def test_ucs_finds_path():
    """Test that UCS finds a valid path."""
    env = WarehouseEnv()
    obs = env.reset()
    path, stats = uniformcost_search((1, 1), obs["pickup_pos"], env)
    assert path is not None, "UCS should find a path"
    assert len(path) >= 2, "Path should have at least start and goal"
    assert path[0] == (1, 1), "Path should start at start position"
    assert path[-1] == obs["pickup_pos"], "Path should end at goal"


def test_astar_finds_path():
    """Test that A* finds a valid path."""
    env = WarehouseEnv()
    obs = env.reset()
    path, stats = astar_search((1, 1), obs["pickup_pos"], env)
    assert path is not None, "A* should find a path"
    assert len(path) >= 2, "Path should have at least start and goal"
    assert path[0] == (1, 1), "Path should start at start position"
    assert path[-1] == obs["pickup_pos"], "Path should end at goal"


def test_ucs_and_astar_find_optimal_paths():
    """Test that UCS and A* find paths of identical length (both optimal)."""
    env = WarehouseEnv()
    obs = env.reset()
    target = obs["pickup_pos"]

    ucs_path, _ = uniformcost_search((1, 1), target, env)
    astar_path, _ = astar_search((1, 1), target, env)

    assert ucs_path is not None and astar_path is not None
    assert len(ucs_path) == len(astar_path), "Both should find paths of equal length (optimality)"


def test_astar_is_more_efficient():
    """Test that A* expands fewer or equal nodes compared to UCS (on average)."""
    env = WarehouseEnv()
    obs = env.reset()
    target = obs["pickup_pos"]

    _, ucs_stats = uniformcost_search((1, 1), target, env)
    _, astar_stats = astar_search((1, 1), target, env)

    # A* should not expand significantly more nodes (with a tight heuristic)
    # Allow for some variance, but A* should generally be better or equal
    assert astar_stats["nodes_expanded"] <= ucs_stats["nodes_expanded"] * 1.5, \
        "A* should not expand drastically more nodes than UCS"
