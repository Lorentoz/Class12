"""Simple reflex agent for the Warehouse environment.

Rules (condition-action):
- If at pickup and not carrying: PICK
- If at dropoff and carrying: DROP
- Else, move toward pickup when not carrying or toward dropoff when carrying.
  - If vertical distance >= horizontal distance: move N (if target is north) or S (if south).
  - Otherwise: move E if target east else W.
- If the chosen move is invalid (wall), choose a random valid move or WAIT.

This agent is stateless (no history); it only uses the current percept (observation).
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

Action = str


class ReflexAgent:
    """Stateless reflex agent using simple condition-action rules."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        """No internal state to reset for reflex agent."""
        return None

    def select_action(self, obs: dict, env) -> Action:
        pos: Tuple[int, int] = tuple(obs["robot_pos"])
        has_item: bool = bool(obs["has_item"])
        pickup = obs.get("pickup_pos")
        dropoff = obs.get("dropoff_pos")

        # If on pickup/dropoff do the appropriate action
        if (not has_item) and (pickup == pos):
            return "PICK"
        if has_item and (dropoff == pos):
            return "DROP"

        # Choose target based on carrying state
        target = dropoff if has_item else pickup
        if target is None:
            return self._random_valid_move(pos, env)

        tr, tc = target
        r, c = pos
        dr = tr - r
        dc = tc - c

        # Prefer vertical moves when vertical distance is >= horizontal distance
        preferred = None
        if abs(dr) >= abs(dc):
            preferred = "N" if dr < 0 else "S"
        else:
            preferred = "E" if dc > 0 else "W"

        # If preferred move is valid, take it; otherwise fall back
        if preferred and self._is_valid_move(preferred, pos, env):
            return preferred

        # If preferred invalid, try the other cardinal moves that reduce distance
        candidates = []
        for act in ["N", "E", "S", "W"]:
            if not self._is_valid_move(act, pos, env):
                continue
            dr2, dc2 = env.MOVE_DELTAS[act]
            nr, nc = r + dr2, c + dc2
            d_after = abs(nr - tr) + abs(nc - tc)
            d_before = abs(r - tr) + abs(c - tc)
            if d_after < d_before:
                candidates.append(act)

        if candidates:
            return self.rng.choice(candidates)

        # Nothing reduces distance or no valid reducing move: random valid move
        return self._random_valid_move(pos, env)

    def _is_valid_move(self, act: str, pos: Tuple[int, int], env) -> bool:
        dr, dc = env.MOVE_DELTAS[act]
        nr, nc = pos[0] + dr, pos[1] + dc
        return not env._is_wall(nr, nc)

    def _random_valid_move(self, pos: Tuple[int, int], env) -> Action:
        valid = []
        for act in ["N", "E", "S", "W"]:
            if self._is_valid_move(act, pos, env):
                valid.append(act)
        if not valid:
            return "WAIT"
        return self.rng.choice(valid)


if __name__ == "__main__":
    # Quick smoke test
    from warehouse_env import WarehouseEnv
    env = WarehouseEnv()
    agent = ReflexAgent(seed=0)
    obs = env.reset()
    print("pickup", obs["pickup_pos"], "dropoff", obs["dropoff_pos"], "start", obs["robot_pos"])
    print("action", agent.select_action(obs, env))
