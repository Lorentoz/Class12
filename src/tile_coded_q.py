"""tile_coded_q.py – Linear Q-learning with tile coding over (x, y).

8 offset tilings of the (x, y) position space, each with a 4×4 grid of
tiles.  Each tiling needs one extra tile per dimension to handle boundary
effects from the offset, giving ext = tiles_per_dim + 1 = 5 tiles per
dimension → 25 tiles per tiling.

Feature vector: binary with n_tilings × ext² = 8 × 25 = 200 entries.
Weight matrix W: (4 actions, 200 features) → 800 total parameters.

Semi-gradient TD update (off-policy Q-learning target):
    W[a] ← W[a] + α · (target − W[a]·φ) · φ
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuous_warehouse import ContinuousWarehouse


# ── Tile-Coded Agent ─────────────────────────────────────────────────────────
class TileCodedAgent:
    """Linear Q-function with tile coding over (x, y)."""

    def __init__(self, n_tilings=8, tiles_per_dim=4, n_actions=4,
                 alpha=0.05, gamma=0.99):
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.n_actions = n_actions
        # Scale alpha by n_tilings so effective step size is alpha
        self.alpha = alpha / n_tilings
        self.gamma = gamma

        # Extra tile per dim for symmetric boundary handling
        self.ext = tiles_per_dim + 1
        self.tiles_per_tiling = self.ext ** 2
        self.n_features = n_tilings * self.tiles_per_tiling

        # Weight matrix: one row per action
        self.W = np.zeros((n_actions, self.n_features))

        # Tile width and per-tiling offsets
        self.tile_width = 4.0 / tiles_per_dim
        self.offsets = np.array([
            [k * self.tile_width / n_tilings,
             k * self.tile_width / n_tilings]
            for k in range(n_tilings)
        ])

    def get_features(self, state):
        """Return binary feature vector from tile coding of (x, y)."""
        phi = np.zeros(self.n_features)
        x, y = state[0], state[1]
        for k in range(self.n_tilings):
            sx = x + self.offsets[k, 0]
            sy = y + self.offsets[k, 1]
            bx = int(sx / self.tile_width)
            by = int(sy / self.tile_width)
            bx = max(0, min(bx, self.ext - 1))
            by = max(0, min(by, self.ext - 1))
            idx = k * self.tiles_per_tiling + bx * self.ext + by
            phi[idx] = 1.0
        return phi

    def q_values(self, state):
        """Compute Q(state, ·) for all actions."""
        phi = self.get_features(state)
        return self.W @ phi

    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values(state)))

    def update(self, state, action, reward, next_state, done):
        """Semi-gradient Q-learning update."""
        phi = self.get_features(state)
        q_sa = self.W[action] @ phi
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_values(next_state))
        self.W[action] += self.alpha * (target - q_sa) * phi


# ── Training ─────────────────────────────────────────────────────────────────
def train_tile_coded(n_episodes=2000, epsilon_start=1.0, epsilon_decay=0.995,
                     epsilon_min=0.01, max_steps=200, seed=42):
    """Train tile-coded Q-learning agent; return (agent, rewards list)."""
    np.random.seed(seed)
    env = ContinuousWarehouse()
    agent = TileCodedAgent()

    epsilon = epsilon_start
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

    return agent, rewards


def rolling_avg(data, window=100):
    return np.convolve(data, np.ones(window) / window, mode='valid')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Training tile-coded Q-learning (2000 episodes)…")
    agent, rewards = train_tile_coded(n_episodes=2000)

    print(f"Tile-coded parameters: {agent.W.size}")
    print(f"  {agent.n_tilings} tilings × {agent.tiles_per_tiling} tiles/tiling"
          f" × {agent.n_actions} actions")
    print(f"Final avg reward (last 200 eps): {np.mean(rewards[-200:]):.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rolling_avg(rewards), color='#ff7f0e', alpha=0.9,
            label='Tile-Coded Q-Learning')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (rolling avg, w=100)')
    ax.set_title('Tile-Coded Linear Q-Learning')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tile_coded_learning_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: tile_coded_learning_curve.png")
