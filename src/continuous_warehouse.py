import numpy as np


class ContinuousWarehouse:
    """Continuous-state warehouse with center hazard.

    State: (x, y, theta, v)
      - (x, y): position on the floor, [0, 4]^2
      - theta: heading angle, [0, 2*pi]
      - v: velocity, [0, 1]

    Actions: 0=North, 1=South, 2=East, 3=West
    Transitions: step_size movement in action direction + Gaussian noise.
    Goal:   circle of radius 0.5 around (3.5, 3.5)  → reward +1, terminal
    Hazard: circle of radius 0.5 around (2.0, 2.0)  → reward -1, terminal
    Otherwise: living penalty -0.04
    """

    ACTION_DELTAS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=float)  # N, S, E, W
    N_ACTIONS = 4
    ACTION_NAMES = ['North', 'South', 'East', 'West']

    def __init__(self, goal_center=(3.5, 3.5), hazard_center=(2.0, 2.0),
                 goal_radius=0.5, hazard_radius=0.5,
                 step_size=0.5, noise_std=0.1):
        self.goal = np.array(goal_center)
        self.hazard = np.array(hazard_center)
        self.goal_radius = goal_radius
        self.hazard_radius = hazard_radius
        self.step_size = step_size
        self.noise_std = noise_std
        self.state = None
        self.reset()

    def reset(self):
        """Reset to start state (0, 0, 0, 0) and return state."""
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        return self.state.copy()

    def step(self, action):
        """Take action; return (next_state, reward, done).

        Position is updated by step_size in the action direction plus
        isotropic Gaussian noise. Heading theta updates to point in the
        action direction; velocity v reflects the actual movement speed.
        """
        x, y, theta, v = self.state
        delta = self.ACTION_DELTAS[action]

        # Stochastic position update
        noise = np.random.normal(0.0, self.noise_std, 2)
        new_x = float(np.clip(x + self.step_size * delta[0] + noise[0], 0.0, 4.0))
        new_y = float(np.clip(y + self.step_size * delta[1] + noise[1], 0.0, 4.0))

        # Heading: direction of chosen action
        new_theta = float(np.arctan2(delta[1], delta[0]) % (2 * np.pi))

        # Velocity: normalised magnitude of actual movement
        movement = np.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
        max_move = self.step_size + 3 * self.noise_std
        new_v = float(np.clip(movement / max_move, 0.0, 1.0))

        self.state = np.array([new_x, new_y, new_theta, new_v])
        pos = np.array([new_x, new_y])

        if np.linalg.norm(pos - self.goal) <= self.goal_radius:
            return self.state.copy(), 1.0, True
        elif np.linalg.norm(pos - self.hazard) <= self.hazard_radius:
            return self.state.copy(), -1.0, True
        else:
            return self.state.copy(), -0.04, False
