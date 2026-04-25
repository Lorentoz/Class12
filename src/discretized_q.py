"""discretized_q.py – Tabular Q-learning with grid discretization.

The (x, y) position is bucketed into a 10×10 grid (ignoring theta and v).
This gives 100 discrete states and a Q-table of 100×4 = 400 parameters.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuous_warehouse import ContinuousWarehouse

# ── Discretization ──────────────────────────────────────────────────────────
N_BINS = 10
BINS = np.linspace(0.0, 4.0, N_BINS + 1)


def discretize(x, y):
    """Map (x, y) → integer state index in [0, N_BINS²)."""
    bx = int(np.clip(np.digitize(x, BINS) - 1, 0, N_BINS - 1))
    by = int(np.clip(np.digitize(y, BINS) - 1, 0, N_BINS - 1))
    return bx * N_BINS + by


# ── Training ─────────────────────────────────────────────────────────────────
def train_discretized(n_episodes=2000, alpha=0.1, gamma=0.99,
                      epsilon_start=1.0, epsilon_decay=0.995,
                      epsilon_min=0.01, max_steps=200, seed=42):
    """Train tabular Q-learning on the discretized state space.

    Returns the Q-table and per-episode total rewards.
    """
    np.random.seed(seed)
    env = ContinuousWarehouse()

    n_states = N_BINS * N_BINS
    Q = np.zeros((n_states, env.N_ACTIONS))

    epsilon = epsilon_start
    rewards = []

    for _ in range(n_episodes):
        raw = env.reset()
        s = discretize(raw[0], raw[1])
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(env.N_ACTIONS)
            else:
                action = int(np.argmax(Q[s]))

            next_raw, reward, done = env.step(action)
            ns = discretize(next_raw[0], next_raw[1])
            total_reward += reward

            # Q-learning update
            target = reward if done else reward + gamma * np.max(Q[ns])
            Q[s, action] += alpha * (target - Q[s, action])

            s = ns
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

    return Q, rewards


def rolling_avg(data, window=100):
    return np.convolve(data, np.ones(window) / window, mode='valid')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Training discretized Q-learning (2000 episodes)…")
    Q, rewards = train_discretized(n_episodes=2000)

    # Parameter count
    n_params = N_BINS * N_BINS * 4
    print(f"Q-table parameters: {n_params}  ({N_BINS}×{N_BINS} states × 4 actions)")
    print(f"Final avg reward (last 200 eps): {np.mean(rewards[-200:]):.3f}")

    # Learning curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rolling_avg(rewards), color='#1f77b4', alpha=0.9,
            label='Discretized Q-Learning')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (rolling avg, w=100)')
    ax.set_title('Discretized Q-Learning – 10×10 Grid')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('discretized_learning_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: discretized_learning_curve.png")
