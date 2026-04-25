"""dqn_agent.py – Deep Q-Network with experience replay and target network.

Architecture:  state (4) → 64 → 64 → 4 actions  (ReLU activations)
Optimizer:     Adam, lr = 1e-3
Replay buffer: capacity 10 000, batch size 64
Target network updated every 100 steps.
State features normalised to [0, 1] before entering the network:
    x, y  ÷ 4   |  theta ÷ 2π  |  v already in [0, 1]

Also includes:
  - Comparison of all three agents (discretized, tile-coded, DQN)
  - Ablation study: full DQN vs. no-replay vs. no-target-network
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from continuous_warehouse import ContinuousWarehouse
from discretized_q import train_discretized
from tile_coded_q import train_tile_coded, TileCodedAgent


# ── Normalisation ─────────────────────────────────────────────────────────────
def normalize(state):
    """Map state (x, y, theta, v) to [0, 1]^4."""
    return np.array([
        state[0] / 4.0,
        state[1] / 4.0,
        state[2] / (2 * np.pi),
        state[3],
    ], dtype=np.float32)


# ── Q-Network ─────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """Two-hidden-layer Q-network: 4 → 64 → 64 → 4."""

    def __init__(self, state_dim=4, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """DQN with optional experience replay and target network."""

    def __init__(self, state_dim=4, n_actions=4, hidden=64,
                 lr=1e-3, gamma=0.99, buffer_size=10_000,
                 batch_size=64, target_update=100):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_count = 0

        # Flags toggled during ablation study
        self.use_replay = True
        self.use_target_net = True

        self.q_net = QNetwork(state_dim, n_actions, hidden)
        self.target_net = QNetwork(state_dim, n_actions, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state_norm, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state_norm).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done):
        self.step_count += 1

        if self.use_replay:
            self.buffer.push(state, action, reward, next_state, done)
            if len(self.buffer) < self.batch_size:
                return
            states_b, actions_b, rewards_b, next_states_b, dones_b = \
                self.buffer.sample(self.batch_size)
        else:
            # Online: train on single transition only
            states_b = np.array([state], dtype=np.float32)
            actions_b = np.array([action])
            rewards_b = np.array([reward], dtype=np.float32)
            next_states_b = np.array([next_state], dtype=np.float32)
            dones_b = np.array([float(done)], dtype=np.float32)

        states_t = torch.FloatTensor(states_b)
        actions_t = torch.LongTensor(actions_b)
        rewards_t = torch.FloatTensor(rewards_b)
        next_states_t = torch.FloatTensor(next_states_b)
        dones_t = torch.FloatTensor(dones_b)

        # Current Q-values for taken actions
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.use_target_net:
                next_q = self.target_net(next_states_t).max(dim=1)[0]
            else:
                next_q = self.q_net(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically sync target network
        if self.use_target_net and self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ── Training loop ─────────────────────────────────────────────────────────────
def train_dqn(use_replay=True, use_target_net=True,
              n_episodes=2000, epsilon_start=1.0, epsilon_decay=0.985,
              epsilon_min=0.01, max_steps=200, seed=42):
    """Train DQN agent; ablation controlled by flags."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = ContinuousWarehouse()
    agent = DQNAgent()
    agent.use_replay = use_replay
    agent.use_target_net = use_target_net

    epsilon = epsilon_start
    rewards = []

    # Pre-fill replay buffer with random transitions
    if use_replay:
        state = env.reset()
        for _ in range(500):
            action = np.random.randint(env.N_ACTIONS)
            next_state, reward, done = env.step(action)
            s_n = normalize(state)
            ns_n = normalize(next_state)
            agent.buffer.push(s_n, action, reward, ns_n, float(done))
            state = env.reset() if done else next_state

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            s_norm = normalize(state)
            action = agent.select_action(s_norm, epsilon)
            next_state, reward, done = env.step(action)
            ns_norm = normalize(next_state)

            agent.update(s_norm, action, reward, ns_norm, float(done))
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
    N_EP = 2000

    # ── 1. Train all three approaches for comparison ──
    print("Training discretized Q-learning…")
    Q_disc, rewards_disc = train_discretized(n_episodes=N_EP)

    print("Training tile-coded Q-learning…")
    _, rewards_tile = train_tile_coded(n_episodes=N_EP)

    print("Training DQN (full)…")
    agent_full, rewards_dqn = train_dqn(n_episodes=N_EP)

    # Parameter counts
    n_disc = 10 * 10 * 4
    n_tile = 8 * 25 * 4
    n_dqn = sum(p.numel() for p in agent_full.q_net.parameters())
    print(f"\nParameter counts:")
    print(f"  Discretized Q-table : {n_disc}")
    print(f"  Tile-coded          : {n_tile}")
    print(f"  DQN                 : {n_dqn}")

    # Final performance
    for name, rw in [("Disc", rewards_disc), ("Tile", rewards_tile), ("DQN", rewards_dqn)]:
        print(f"  {name} final avg (last 200 eps): {np.mean(rw[-200:]):.3f}")

    # ── 2. Comparison plot ──
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rolling_avg(rewards_disc), label='Discretized Q-Learning', color='#1f77b4', alpha=0.85)
    ax.plot(rolling_avg(rewards_tile), label='Tile-Coded Q-Learning',  color='#ff7f0e', alpha=0.85)
    ax.plot(rolling_avg(rewards_dqn),  label='DQN',                    color='#2ca02c', alpha=0.85)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (rolling avg, w=100)')
    ax.set_title('Comparison: Discretized vs. Tile-Coded vs. DQN')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: comparison_curves.png")

    # ── 3. DQN Ablation study ──
    print("\nTraining DQN without replay…")
    _, rewards_no_replay = train_dqn(use_replay=False, use_target_net=True, n_episodes=N_EP)

    print("Training DQN without target network…")
    _, rewards_no_target = train_dqn(use_replay=True, use_target_net=False, n_episodes=N_EP)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(rolling_avg(rewards_dqn),       label='Full DQN',         color='#2ca02c', alpha=0.85)
    ax2.plot(rolling_avg(rewards_no_replay), label='No Replay',        color='#ff7f0e', alpha=0.85)
    ax2.plot(rolling_avg(rewards_no_target), label='No Target Network', color='#d62728', alpha=0.85)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward (rolling avg, w=100)')
    ax2.set_title('DQN Ablation Study')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dqn_ablation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: dqn_ablation.png")
