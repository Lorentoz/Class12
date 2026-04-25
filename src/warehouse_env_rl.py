import numpy as np


class WarehouseEnv:
    """4x4 warehouse grid MDP for reinforcement learning.

    The agent has NO access to the transition model T or reward function R.
    It interacts solely through reset() and step(action).

    Grid layout:
        States: (x, y) with x in {0..3}, y in {0..3}
        Goal:   (3, 3)  → reward +1, terminal
        Hazard: (3, 2)  → reward -1, terminal
        All other states → reward -0.04 (living penalty)

    Actions: 0=North, 1=South, 2=West, 3=East
    Transition: 80% intended direction, 10% each perpendicular direction.
    """

    GRID_W = 4
    GRID_H = 4
    GOAL   = (3, 3)
    HAZARD = (3, 2)
    TERMINALS = {(3, 3), (3, 2)}
    # dx, dy for N, S, W, E
    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    ACTION_NAMES = ["N", "S", "W", "E"]
    N_ACTIONS = 4
    LIVING_REWARD = -0.04
    START = (0, 0)

    def __init__(self):
        self.state = self.START

    def reset(self):
        """Reset environment to start state."""
        self.state = self.START
        return self.state

    def get_all_states(self):
        """Return all non-terminal states."""
        return [
            (x, y)
            for x in range(self.GRID_W)
            for y in range(self.GRID_H)
            if (x, y) not in self.TERMINALS
        ]

    def step(self, action):
        """Execute action; return (next_state, reward, done).

        Reward convention: r = R(s'), i.e. the reward for entering next_state.
        If already in a terminal state, return immediately with reward 0.
        """
        if self.state in self.TERMINALS:
            return self.state, 0.0, True

        # Stochastic transition: 80/10/10
        intended = self.MOVES[action]
        if action in (0, 1):          # N or S → perpendiculars are W and E
            perps = [self.MOVES[2], self.MOVES[3]]
        else:                          # W or E → perpendiculars are N and S
            perps = [self.MOVES[0], self.MOVES[1]]

        r = np.random.random()
        if r < 0.8:
            move = intended
        elif r < 0.9:
            move = perps[0]
        else:
            move = perps[1]

        nx = max(0, min(self.GRID_W - 1, self.state[0] + move[0]))
        ny = max(0, min(self.GRID_H - 1, self.state[1] + move[1]))
        next_state = (nx, ny)

        if next_state == self.GOAL:
            reward, done = 1.0, True
        elif next_state == self.HAZARD:
            reward, done = -1.0, True
        else:
            reward, done = self.LIVING_REWARD, False

        self.state = next_state
        return next_state, reward, done
