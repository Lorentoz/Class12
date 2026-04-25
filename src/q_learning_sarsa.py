import numpy as np
import matplotlib.pyplot as plt
from warehouse_env_rl import WarehouseEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def epsilon_greedy(Q, state, epsilon, n_actions):
    """Choose action via ε-greedy policy over Q."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    q_vals = [Q.get((state, a), 0.0) for a in range(n_actions)]
    return int(np.argmax(q_vals))


def rolling_avg(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode="valid")


# ---------------------------------------------------------------------------
# Q-Learning (off-policy)
# ---------------------------------------------------------------------------

def q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    """Off-policy Q-learning with ε-greedy exploration.

    Returns Q-table (dict) and list of per-episode total rewards.
    """
    Q = {}
    epsilon = epsilon_start
    episode_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy(Q, state, epsilon, env.N_ACTIONS)
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                # No future rewards from terminal state
                target = reward
            else:
                max_q_next = max(Q.get((next_state, a), 0.0) for a in range(env.N_ACTIONS))
                target = reward + gamma * max_q_next

            old_q = Q.get((state, action), 0.0)
            Q[(state, action)] = old_q + alpha * (target - old_q)
            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    return Q, episode_rewards


# ---------------------------------------------------------------------------
# SARSA (on-policy)
# ---------------------------------------------------------------------------

def sarsa(env, n_episodes=1000, alpha=0.1, gamma=0.99,
          epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    """On-policy SARSA with ε-greedy exploration.

    Unlike Q-learning, the update uses the action *actually* chosen at s',
    making it more conservative near hazards.

    Returns Q-table (dict) and list of per-episode total rewards.
    """
    Q = {}
    epsilon = epsilon_start
    episode_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon, env.N_ACTIONS)
        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                target = reward
            else:
                next_action = epsilon_greedy(Q, next_state, epsilon, env.N_ACTIONS)
                target = reward + gamma * Q.get((next_state, next_action), 0.0)

            old_q = Q.get((state, action), 0.0)
            Q[(state, action)] = old_q + alpha * (target - old_q)

            if not done:
                state = next_state
                action = next_action
            else:
                state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    return Q, episode_rewards


# ---------------------------------------------------------------------------
# Policy extraction & comparison
# ---------------------------------------------------------------------------

def extract_policy(Q, states, n_actions):
    """Return greedy policy dict {state: best_action}."""
    return {s: int(np.argmax([Q.get((s, a), 0.0) for a in range(n_actions)]))
            for s in states}


# Optimal policy from value iteration (0=N,1=S,2=W,3=E)
OPTIMAL_POLICY = {
    (0, 0): 0, (1, 0): 3, (2, 0): 3, (3, 0): 0,
    (0, 1): 0, (1, 1): 0, (2, 1): 0, (3, 1): 2,
    (0, 2): 0, (1, 2): 0, (2, 2): 0,
    (0, 3): 3, (1, 3): 3, (2, 3): 3,
}


def compare_to_optimal(policy, all_states, label):
    matches = sum(1 for s in all_states if policy.get(s) == OPTIMAL_POLICY.get(s))
    print(f"{label}: {matches}/{len(all_states)} states match optimal policy")
    return matches


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

ARROW_MAP = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}


def plot_policy(ax, policy, title):
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    for x in range(5):
        ax.axvline(x - 0.5, color="gray", linewidth=0.5)
    for y in range(5):
        ax.axhline(y - 0.5, color="gray", linewidth=0.5)

    # Goal & hazard patches
    ax.add_patch(plt.Rectangle((2.5, 2.5), 1, 1, color="green", alpha=0.3))
    ax.text(3, 3, "+1", ha="center", va="center", fontsize=11,
            fontweight="bold", color="green")
    ax.add_patch(plt.Rectangle((2.5, 1.5), 1, 1, color="red", alpha=0.3))
    ax.text(3, 2, "-1", ha="center", va="center", fontsize=11,
            fontweight="bold", color="red")

    for state, action in policy.items():
        dx, dy = ARROW_MAP[action]
        ax.annotate("", xy=(state[0] + dx, state[1] + dy),
                    xytext=(state[0], state[1]),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))

    ax.set_xticks(range(4))
    ax.set_xticklabels([f"x={i+1}" for i in range(4)])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"y={i+1}" for i in range(4)])
    ax.set_title(title)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    env = WarehouseEnv()
    all_states = env.get_all_states()

    # ---- Task 2 & 3: Train Q-learning and SARSA ----
    print("Training Q-Learning …")
    Q_ql, rewards_ql = q_learning(env, n_episodes=1000)

    print("Training SARSA …")
    Q_sarsa, rewards_sarsa = sarsa(env, n_episodes=1000)

    # ---- Task 4: Learning curves ----
    # Approximate optimal expected return from value iteration
    OPTIMAL_RETURN = 0.62

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rolling_avg(rewards_ql),    label="Q-Learning", alpha=0.85)
    ax.plot(rolling_avg(rewards_sarsa), label="SARSA",      alpha=0.85)
    ax.axhline(OPTIMAL_RETURN, color="black", linestyle="--",
               linewidth=1.2, label=f"Optimal return ≈ {OPTIMAL_RETURN}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (rolling avg, w=50)")
    ax.set_title("Q-Learning vs. SARSA – Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curves.svg", bbox_inches="tight")
    plt.show()

    # ---- Task 5: Policy comparison ----
    policy_ql    = extract_policy(Q_ql,    all_states, env.N_ACTIONS)
    policy_sarsa = extract_policy(Q_sarsa, all_states, env.N_ACTIONS)

    compare_to_optimal(policy_ql,    all_states, "Q-Learning")
    compare_to_optimal(policy_sarsa, all_states, "SARSA     ")

    print("\nDifferences (state: VI / Q-L / SARSA):")
    names = env.ACTION_NAMES
    for s in sorted(all_states):
        opt = OPTIMAL_POLICY.get(s, -1)
        ql  = policy_ql[s]
        sa  = policy_sarsa[s]
        if ql != opt or sa != opt:
            label = f"({s[0]+1},{s[1]+1})"
            print(f"  {label}: VI={names[opt]}, Q-L={names[ql]}, SARSA={names[sa]}")

    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_policy(axes[0], OPTIMAL_POLICY, "Value Iteration (optimal)")
    plot_policy(axes[1], policy_ql,      "Q-Learning")
    plot_policy(axes[2], policy_sarsa,   "SARSA")
    plt.tight_layout()
    plt.savefig("policy_comparison.svg", bbox_inches="tight")
    plt.show()

    # ---- Task 6: Hyperparameter sensitivity ----
    alphas = [0.01, 0.1, 0.5]
    rhos   = [0.99, 0.995, 0.999]

    fig3, axes3 = plt.subplots(len(alphas), len(rhos),
                               figsize=(14, 10), sharex=True, sharey=True)

    for i, alpha in enumerate(alphas):
        for j, rho in enumerate(rhos):
            np.random.seed(42)
            _, rw = q_learning(env, n_episodes=1000, alpha=alpha, epsilon_decay=rho)
            ax = axes3[i][j]
            ax.plot(rolling_avg(rw, 50), color="#1f77b4", alpha=0.85)
            ax.axhline(OPTIMAL_RETURN, color="red", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.set_title(rf"$\alpha$={alpha}, $\rho$={rho}", fontsize=10)
            if j == 0:
                ax.set_ylabel("Reward")
            if i == len(alphas) - 1:
                ax.set_xlabel("Episode")

    fig3.suptitle("Q-Learning: Hyperparameter Sensitivity", fontsize=13)
    plt.tight_layout()
    plt.savefig("hyperparameter_sensitivity.svg", bbox_inches="tight")
    plt.show()

    print("\nDone. Saved: learning_curves.svg, policy_comparison.svg, "
          "hyperparameter_sensitivity.svg")


if __name__ == "__main__":
    main()
