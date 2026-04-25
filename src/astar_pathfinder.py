"""A* pathfinder with Manhattan distance heuristic for warehouse navigation.

A* expands nodes in order of f(n) = g(n) + h(n), where:
- g(n): cost from start
- h(n): heuristic estimate (Manhattan distance to goal)

Returns: path (list of positions), statistics (nodes_expanded, path_length, time_ms)
"""
from __future__ import annotations

import heapq
import time
from typing import Callable, Dict, List, Optional, Tuple

Position = Tuple[int, int]


class Node:
    """Search node with path cost and heuristic tracking."""

    def __init__(
        self,
        position: Position,
        parent: Optional[Node] = None,
        action: Optional[str] = None,
        g: float = 0.0,
        h: float = 0.0,
    ):
        self.position = position
        self.parent = parent
        self.action = action
        self.g = g  # Cost from start
        self.h = h  # Heuristic estimate to goal
        self.f = g + h  # Total estimate

    def __lt__(self, other):
        """For priority queue: lower f has higher priority."""
        return self.f < other.f

    def path(self) -> List[Position]:
        """Reconstruct path from start to this node."""
        if self.parent is None:
            return [self.position]
        return self.parent.path() + [self.position]


def manhattan_distance(pos: Position, goal: Position) -> float:
    """Manhattan distance heuristic: |x_pos - x_goal| + |y_pos - y_goal|"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar_search(
    start: Position,
    goal: Position,
    env,
    heuristic: Callable[[Position, Position], float] = manhattan_distance,
    max_expansions: int = 10000,
) -> Tuple[Optional[List[Position]], Dict]:
    """
    A* Search for pathfinding with heuristic.

    Args:
        start: Starting position (r, c)
        goal: Goal position (r, c)
        env: WarehouseEnv instance (for grid checks)
        heuristic: Function h(n) estimating distance to goal (default: Manhattan)
        max_expansions: Safety limit to prevent infinite loops

    Returns:
        (path, stats) where path is a list of positions or None if not found,
        stats includes: nodes_expanded, path_length, time_ms, frontier_peak
    """
    start_time = time.time()
    h_start = heuristic(start, goal)
    start_node = Node(start, g=0, h=h_start)

    frontier = []
    heapq.heappush(frontier, (start_node.f, id(start_node), start_node))

    explored = set()
    frontier_peak = 1
    nodes_expanded = 0

    while frontier:
        f, _, node = heapq.heappop(frontier)

        if node.position == goal:
            elapsed = time.time() - start_time
            return node.path(), {
                "nodes_expanded": nodes_expanded,
                "path_length": len(node.path()),
                "time_ms": elapsed * 1000,
                "frontier_peak": frontier_peak,
            }

        if node.position in explored:
            continue

        explored.add(node.position)
        nodes_expanded += 1

        if nodes_expanded > max_expansions:
            break

        # Expand neighbors (cardinal moves)
        r, c = node.position
        for act in ["N", "E", "S", "W"]:
            dr, dc = env.MOVE_DELTAS[act]
            nr, nc = r + dr, c + dc
            if env._is_wall(nr, nc):
                continue
            child_pos = (nr, nc)
            if child_pos in explored:
                continue

            child_g = node.g + 1  # Each move costs 1
            child_h = heuristic(child_pos, goal)
            child_node = Node(child_pos, parent=node, action=act, g=child_g, h=child_h)
            heapq.heappush(frontier, (child_node.f, id(child_node), child_node))

        frontier_peak = max(frontier_peak, len(frontier))

    elapsed = time.time() - start_time
    return None, {
        "nodes_expanded": nodes_expanded,
        "path_length": None,
        "time_ms": elapsed * 1000,
        "frontier_peak": frontier_peak,
    }


if __name__ == "__main__":
    from warehouse_env import WarehouseEnv

    env = WarehouseEnv()
    obs = env.reset()
    print("Testing A* pathfinder")
    print(f"Start: {(1, 1)}, Pickup: {obs['pickup_pos']}")
    path, stats = astar_search((1, 1), obs["pickup_pos"], env)
    if path:
        print(f"Path found: {len(path)} steps")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
        print(f"  Time: {stats['time_ms']:.2f} ms")
    else:
        print("No path found")
