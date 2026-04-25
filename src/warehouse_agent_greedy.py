"""Greedy Manhattan agent for the Warehouse environment.

Behavior:
- Compute Manhattan distance to current goal: pickup if not carrying, dropoff if carrying.
- If on the pickup/dropoff tile, take `PICK`/`DROP` respectively.
- Prefer movement actions (N/E/S/W) that reduce Manhattan distance.
- If no move reduces the distance (stuck), choose a random valid move.
- Track last N positions (default 10); if current position was visited recently,
  trigger an escape behavior that performs a few (default 3) random moves.

The agent is implemented as a simple class with a `select_action(obs, env)` method
so it can be used with the project's `WarehouseEnv`.
"""
from collections import deque
import random
from typing import Deque, Tuple, Optional

Action = str


class GreedyManhattanAgent:
    """Greedy Manhattan agent with loop detection and escape moves."""

    def __init__(self, loop_history_size: int = 10, escape_steps: int = 3, seed: Optional[int] = None):
        self.loop_history_size = loop_history_size
        self.recent: Deque[Tuple[int, int]] = deque(maxlen=loop_history_size)
        self.escape_steps = escape_steps
        self.escape_counter = 0
        self.rng = random.Random(seed)

    def reset(self) -> None:
        """Reset internal loop detector and escape counter."""
        self.recent.clear()
        self.escape_counter = 0

    def select_action(self, obs: dict, env) -> Action:
        """Select an action given the observation (and environment for grid info).

        obs is expected to be the dictionary returned by `WarehouseEnv._observe()`.
        env is the `WarehouseEnv` instance (used for MOVE_DELTAS and wall checks).
        """
        pos = tuple(obs["robot_pos"])
        has_item = bool(obs["has_item"])  # True when carrying
        pickup = obs.get("pickup_pos")
        dropoff = obs.get("dropoff_pos")

        # If on a pickup/dropoff tile, perform the pickup/drop action.
        if not has_item and pickup == pos:
            return "PICK"
        if has_item and dropoff == pos:
            return "DROP"

        # If on a dropoff tile but not carrying anything, move toward pickup.
        if (not has_item) and (dropoff == pos):
            # Try to choose a move that reduces distance to the pickup tile.
            if pickup is not None:
                curr_dist_pickup = abs(pos[0] - pickup[0]) + abs(pos[1] - pickup[1])
                best = []
                best_d = curr_dist_pickup
                for act in ["N", "E", "S", "W"]:
                    dr, dc = env.MOVE_DELTAS[act]
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if env._is_wall(nr, nc):
                        continue
                    d = abs(nr - pickup[0]) + abs(nc - pickup[1])
                    if d < best_d:
                        best_d = d
                        best = [act]
                    elif d == best_d and best:
                        best.append(act)
                if best:
                    return self.rng.choice(best)
            # Fallback: random valid move
            return self._random_valid_move(pos, env)

        # If we are currently escaping, continue doing goal-directed moves when possible.
        if self.escape_counter > 0:
            self.escape_counter -= 1
            target = pickup if not has_item else dropoff
            if target is not None:
                curr = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                best = []
                best_d = curr
                for act in ["N", "E", "S", "W"]:
                    dr, dc = env.MOVE_DELTAS[act]
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if env._is_wall(nr, nc):
                        continue
                    d = abs(nr - target[0]) + abs(nc - target[1])
                    if d < best_d:
                        best_d = d
                        best = [act]
                    elif d == best_d and best:
                        best.append(act)
                if best:
                    return self.rng.choice(best)
            return self._random_valid_move(pos, env)

        # Loop detection: if we've been here recently, trigger escape behavior.
        if pos in self.recent:
            self.escape_counter = self.escape_steps
            # Prefer escaping toward the current goal (pickup if not carrying).
            goal_escape = pickup if not has_item else dropoff
            if goal_escape is not None:
                curr = abs(pos[0] - goal_escape[0]) + abs(pos[1] - goal_escape[1])
                best = []
                best_d = curr
                for act in ["N", "E", "S", "W"]:
                    dr, dc = env.MOVE_DELTAS[act]
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if env._is_wall(nr, nc):
                        continue
                    d = abs(nr - goal_escape[0]) + abs(nc - goal_escape[1])
                    if d < best_d:
                        best_d = d
                        best = [act]
                    elif d == best_d and best:
                        best.append(act)
                if best:
                    return self.rng.choice(best)
            # Fallback: a random valid move
            return self._random_valid_move(pos, env)

        # Record current position in history.
        self.recent.append(pos)

        # Determine goal position (pickup when not carrying, dropoff when carrying).
        goal = dropoff if has_item else pickup
        if goal is None:
            # No defined goal, just take a random valid move.
            return self._random_valid_move(pos, env)

        # Compute current Manhattan distance.
        curr_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Evaluate all cardinal moves and select those that reduce distance.
        best_actions = []
        best_dist = curr_dist
        for act in ["N", "E", "S", "W"]:
            dr, dc = env.MOVE_DELTAS[act]
            nr, nc = pos[0] + dr, pos[1] + dc
            # Skip invalid moves (walls / out-of-bounds).
            if env._is_wall(nr, nc):
                continue
            d = abs(nr - goal[0]) + abs(nc - goal[1])
            if d < best_dist:
                best_dist = d
                best_actions = [act]
            elif d == best_dist and best_actions:
                # Only add ties if we already have found a reducing move; this
                # keeps behavior greedy while allowing random tie-breaks.
                best_actions.append(act)

        if best_actions:
            # Choose randomly among equally-good moves for variety.
            return self.rng.choice(best_actions)

        # No move reduces the distance (stuck): fallback to a random valid move.
        return self._random_valid_move(pos, env)

    def _random_valid_move(self, pos: Tuple[int, int], env) -> Action:
        """Return a random valid cardinal move (or WAIT if nowhere to go)."""
        valid = []
        for act in ["N", "E", "S", "W"]:
            dr, dc = env.MOVE_DELTAS[act]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not env._is_wall(nr, nc):
                valid.append(act)
        if not valid:
            return "WAIT"
        return self.rng.choice(valid)


# Convenience helper to run an episode with the greedy agent (useful for manual testing)
if __name__ == "__main__":
    from warehouse_env import WarehouseEnv

    env = WarehouseEnv()
    agent = GreedyManhattanAgent(seed=0)
    obs = env.reset()
    frames = [env.render_grid()]
    rewards = []
    dists = []
    for _ in range(200):
        act = agent.select_action(obs, env)
        obs, r, done, trunc, info = env.step(act)
        frames.append(env.render_grid())
        rewards.append(r)
        # distance to pickup/dropoff depending on carrying
        goal = obs["dropoff_pos"] if obs["has_item"] else obs["pickup_pos"]
        if goal is not None:
            dists.append(abs(obs["robot_pos"][0] - goal[0]) + abs(obs["robot_pos"][1] - goal[1]))
        if done or trunc:
            break
    print(env.render_with_legend())
    print("Total reward:", sum(rewards))
