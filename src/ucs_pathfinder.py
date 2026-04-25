"""Uniform Cost Search (UCS) pathfinder for warehouse navigation.

For a given start and goal position, UCS finds the optimal (shortest) path
by expanding nodes in order of increasing path cost g(n).

Returns: path (list of positions), statistics (nodes_expanded, frontier_size, time_ms)
"""
from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

Position = Tuple[int, int]


class Node:
    """Search node with path cost tracking."""

    def __init__(
        self,
        position: Position,
        parent: Optional[Node] = None,
        action: Optional[str] = None,
        g: float = 0.0,
    ):
        self.position = position
        self.parent = parent
        self.action = action
        self.g = g  # Cost from start

    def __lt__(self, other):
        """For priority queue: lower cost has higher priority."""
        return self.g < other.g

    def path(self) -> List[Position]:
        """Reconstruct path from start to this node."""
        if self.parent is None:
            return [self.position]
        return self.parent.path() + [self.position]


def uniformcost_search(
    start: Position, goal: Position, env, max_expansions: int = 10000
) -> Tuple[Optional[List[Position]], Dict]:
    """
    Uniform Cost Search for pathfinding.

    Args:
        start: Starting position (r, c)
        goal: Goal position (r, c)
        env: WarehouseEnv instance (for grid checks)
        max_expansions: Safety limit to prevent infinite loops

    Returns:
        (path, stats) where path is a list of positions or None if not found,
        stats includes: nodes_expanded, path_length, time_ms, frontier_peak
    """
    start_time = time.time()
    start_node = Node(start, g=0)

    frontier = []
    heapq.heappush(frontier, (0, id(start_node), start_node))  # (cost, unique_id, node)

    explored: Dict[Position, float] = {}  # Maps position -> lowest cost found
    frontier_peak = 1
    nodes_expanded = 0

    while frontier:
        cost, _, node = heapq.heappop(frontier)

        if node.position == goal:
            elapsed = time.time() - start_time
            return node.path(), {
                "nodes_expanded": nodes_expanded,
                "path_length": len(node.path()),
                "time_ms": elapsed * 1000,
                "frontier_peak": frontier_peak,
            }

        if node.position in explored and explored[node.position] <= node.g:
            continue  # Skip if cheaper path was found

        explored[node.position] = node.g
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
            child_g = node.g + 1  # Each move costs 1

            # Only add if not explored or found a better path
            if child_pos not in explored or explored[child_pos] > child_g:
                child_node = Node(child_pos, parent=node, action=act, g=child_g)
                heapq.heappush(frontier, (child_g, id(child_node), child_node))

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
    print("Testing UCS pathfinder")
    print(f"Start: {(1, 1)}, Pickup: {obs['pickup_pos']}")
    path, stats = uniformcost_search((1, 1), obs["pickup_pos"], env)
    if path:
        print(f"Path found: {len(path)} steps")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
        print(f"  Time: {stats['time_ms']:.2f} ms")
    else:
        print("No path found")
