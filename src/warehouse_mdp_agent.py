"""
Problem 4.4 — Building an MDP Agent (Guided)

Implements the complete pipeline for the 4×4 warehouse MDP:
  1. MDP definition  (states, actions, rewards, transitions)
  2. Value iteration  → optimal value function V*
  3. Policy extraction → optimal policy π*
  4. Environment simulator  simulate_step(state, action)
  5. Agent loop  run_episode(policy, start, max_steps)
  6. Experiments:
       Task 2 – 1000 episodes with optimal policy
       Task 3 – comparison with naive greedy policy
       Task 4 – discount factor sweep γ ∈ {0.1, 0.5, 0.9, 0.99}
       Task 5 – harder warehouse with second hazard at (2, 3)

Coordinate system: (x, y) with x = column (1–4), y = row (1–4).
  y=4 is the top row, y=1 is the bottom row.
"""

import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# 4.4.1  MDP Definition
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 4, 4

GOAL   = (4, 4)
HAZARD = (4, 3)
TERMINALS = {GOAL: +1.0, HAZARD: -1.0}

LIVING_REWARD = -0.04

# All grid cells (x, y)
STATES = [(x, y) for x in range(1, WIDTH + 1) for y in range(1, HEIGHT + 1)]

# Actions and their (dx, dy) displacement vectors
ACTIONS = {
    "North": (0,  1),
    "South": (0, -1),
    "East":  (1,  0),
    "West":  (-1, 0),
}

ARROWS = {"North": "↑", "South": "↓", "East": "→", "West": "←"}


def reward(state, terminals=None):
    """R(s): immediate reward for being in state s."""
    t = terminals if terminals is not None else TERMINALS
    if state in t:
        return t[state]
    return LIVING_REWARD


# ---------------------------------------------------------------------------
# 4.4.2  Transition Model
# ---------------------------------------------------------------------------

def get_perpendicular(action):
    """Return the two actions perpendicular to the given action."""
    if action in ("North", "South"):
        return ["West", "East"]
    return ["North", "South"]


def attempt_move(state, action):
    """
    Return the state reached by attempting to move in *action* direction.
    If the move would leave the grid the robot stays in place.
    """
    dx, dy = ACTIONS[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 1 <= nx <= WIDTH and 1 <= ny <= HEIGHT:
        return (nx, ny)
    return state          # wall / boundary → stay


def transitions(state, action, terminals=None):
    """
    T(s' | s, a): return a dict {s': probability} for all reachable
    next states when taking *action* in *state*.

    Stochastic model:
      • 0.8  – intended direction
      • 0.1  – 90° left (perpendicular 1)
      • 0.1  – 90° right (perpendicular 2)
    If a move hits a wall the robot stays; probabilities accumulate
    correctly (e.g. two blocked moves both add to the current cell).
    """
    t = terminals if terminals is not None else TERMINALS
    if state in t:
        return {}           # no transitions from terminal states

    outcomes = {}
    # Intended direction (80 %)
    intended = attempt_move(state, action)
    outcomes[intended] = outcomes.get(intended, 0) + 0.8
    # Two perpendicular drifts (10 % each)
    for perp in get_perpendicular(action):
        drifted = attempt_move(state, perp)
        outcomes[drifted] = outcomes.get(drifted, 0) + 0.1
    return outcomes


# ---------------------------------------------------------------------------
# 4.4.3  Value Iteration
# ---------------------------------------------------------------------------

def value_iteration(gamma=0.99, epsilon=1e-6, terminals=None, living_reward=None):
    """
    Run value iteration and return (V, num_iterations).

    V is a dict {state: value} converged to within *epsilon*.
    Uses synchronous (batch) updates so every state reads from Vk,
    not a mix of Vk and Vk+1.
    """
    t  = terminals     if terminals     is not None else TERMINALS
    lr = living_reward if living_reward is not None else LIVING_REWARD

    def rew(s):
        return t[s] if s in t else lr

    V = {s: 0.0 for s in STATES}
    iteration = 0

    while True:
        V_new = {}
        delta = 0

        for s in STATES:
            if s in t:
                V_new[s] = rew(s)   # terminal values are fixed
                continue

            # Bellman update: V(s) = R(s) + γ · max_a Σ T(s'|s,a) V(s')
            best_value = float("-inf")
            for a in ACTIONS:
                expected = sum(
                    prob * V[s_next]
                    for s_next, prob in transitions(s, a, t).items()
                )
                best_value = max(best_value, expected)

            V_new[s] = rew(s) + gamma * best_value
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        iteration += 1

        if delta < epsilon:
            break

    return V, iteration


# ---------------------------------------------------------------------------
# 4.4.4  Extract Policy
# ---------------------------------------------------------------------------

def extract_policy(V, gamma=0.99, terminals=None):
    """
    Compute the optimal policy from a converged value function V.
    π*(s) = argmax_a Σ T(s'|s,a) V(s')
    """
    t = terminals if terminals is not None else TERMINALS
    policy = {}
    for s in STATES:
        if s in t:
            policy[s] = None
            continue
        best_action = None
        best_value  = float("-inf")
        for a in ACTIONS:
            expected = sum(
                prob * V[s_next]
                for s_next, prob in transitions(s, a, t).items()
            )
            value = reward(s, t) + gamma * expected
            if value > best_value:
                best_value  = value
                best_action = a
        policy[s] = best_action
    return policy


# ---------------------------------------------------------------------------
# 4.4.5  Display Helper
# ---------------------------------------------------------------------------

def display_grid(V, policy, terminals=None):
    """Print V* and the optimal policy as grids (top row = y=4)."""
    t = terminals if terminals is not None else TERMINALS
    print("Value function V*:")
    for y in range(HEIGHT, 0, -1):
        row = []
        for x in range(1, WIDTH + 1):
            row.append(f"{V[(x, y)]:7.3f}")
        print("  " + "  ".join(row))

    print("\nOptimal policy:")
    for y in range(HEIGHT, 0, -1):
        row = []
        for x in range(1, WIDTH + 1):
            s = (x, y)
            if s == GOAL:
                row.append(" GOAL ")
            elif s in t and s != GOAL:
                row.append(" HAZD ")
            else:
                arrow = ARROWS.get(policy[s], "?")
                row.append(f"  {arrow}   ")
        print("  " + "".join(row))
    print()


# ---------------------------------------------------------------------------
# Task 1 — Environment Simulator
# ---------------------------------------------------------------------------

def simulate_step(state, action, terminals=None):
    """
    Sample a next state from T(s' | state, action).

    Uses random.choices to draw one outcome according to the
    transition probabilities.  Returns the sampled next state.
    """
    t = terminals if terminals is not None else TERMINALS
    dist = transitions(state, action, t)
    if not dist:
        return state    # terminal state – no movement
    states = list(dist.keys())
    probs  = list(dist.values())
    return random.choices(states, weights=probs, k=1)[0]


def verify_simulator(n=10_000):
    """
    Verify simulate_step by checking empirical frequencies for
    simulate_step((3, 1), "North") over n samples.
    Expected: North→(3,2)≈80%, West→(2,1)≈10%, East→(4,1)≈10%
    """
    counts = defaultdict(int)
    for _ in range(n):
        s_next = simulate_step((3, 1), "North")
        counts[s_next] += 1
    print("Verification of simulate_step((3,1), 'North'):")
    for s, c in sorted(counts.items()):
        print(f"  → {s}: {c/n:.3f}  (expected: "
              f"{'0.800' if s == (3,2) else '0.100'})")
    print()


# ---------------------------------------------------------------------------
# Task 2 — Agent Loop
# ---------------------------------------------------------------------------

def run_episode(policy, start=(1, 1), max_steps=100, terminals=None):
    """
    Simulate a single episode following *policy* from *start*.

    Returns:
      trajectory  – list of states visited (including start)
      total_reward – undiscounted sum of rewards collected
      outcome      – "goal" | "hazard" | "timeout"
    """
    t = terminals if terminals is not None else TERMINALS
    state        = start
    trajectory   = [state]
    total_reward = 0.0

    for _ in range(max_steps):
        if state in t:
            break                       # terminal state reached

        action   = policy[state]
        state    = simulate_step(state, action, t)
        total_reward += reward(state, t)
        trajectory.append(state)

    # Determine outcome
    if state == GOAL:
        outcome = "goal"
    elif state in t:
        outcome = "hazard"
    else:
        outcome = "timeout"

    return trajectory, total_reward, outcome


def run_episodes(policy, n=1_000, start=(1, 1), terminals=None):
    """
    Run *n* episodes and return summary statistics.
    Returns (goal_rate, hazard_rate, avg_reward).
    """
    goals   = 0
    hazards = 0
    total   = 0.0

    for _ in range(n):
        _, r, outcome = run_episode(policy, start=start, terminals=terminals)
        total += r
        if outcome == "goal":
            goals   += 1
        elif outcome == "hazard":
            hazards += 1

    return goals / n, hazards / n, total / n


# ---------------------------------------------------------------------------
# Task 3 — Naive Greedy Policy
# ---------------------------------------------------------------------------

def greedy_policy(goal=GOAL):
    """
    A naive policy that always moves toward the goal:
      • East  if goal is to the right
      • West  if goal is to the left
      • North if goal is above   (and same column)
      • South if goal is below   (and same column)
    Does not consider hazards at all.
    """
    policy = {}
    for s in STATES:
        if s in TERMINALS:
            policy[s] = None
            continue
        gx, gy = goal
        sx, sy = s
        if gx > sx:
            policy[s] = "East"
        elif gx < sx:
            policy[s] = "West"
        elif gy > sy:
            policy[s] = "North"
        else:
            policy[s] = "South"
    return policy


# ---------------------------------------------------------------------------
# Task 4 — Discount Factor Experiment
# ---------------------------------------------------------------------------

def discount_experiment(gammas=(0.1, 0.5, 0.9, 0.99), n=1_000):
    """
    For each γ in *gammas*, compute the optimal policy and run *n* episodes.
    Prints goal-reach rate, hazard rate, and average reward.
    Returns results as a list of dicts.
    """
    print("=" * 60)
    print("Task 4 — Discount Factor Experiment")
    print("=" * 60)
    print(f"{'γ':>6}  {'iters':>6}  {'goal%':>7}  {'hazard%':>8}  {'avg_reward':>11}")
    print("-" * 50)

    results = []
    for g in gammas:
        V, iters = value_iteration(gamma=g)
        policy   = extract_policy(V, gamma=g)
        gr, hr, ar = run_episodes(policy, n=n)
        print(f"{g:>6.2f}  {iters:>6}  {gr*100:>6.1f}%  {hr*100:>7.1f}%  {ar:>11.4f}")
        results.append({"gamma": g, "iters": iters,
                         "goal_rate": gr, "hazard_rate": hr, "avg_reward": ar})

    print()
    # Find first γ where hazard rate drops to ~0
    for r in results:
        if r["hazard_rate"] < 0.01:
            print(f"  → Agent first consistently avoids hazard at γ = {r['gamma']}")
            break
    print()
    return results


# ---------------------------------------------------------------------------
# Task 5 — Harder Warehouse (second hazard at (2, 3))
# ---------------------------------------------------------------------------

def harder_warehouse_experiment(n=1_000):
    """
    Add a second hazard at (2, 3) with reward -1.
    Recompute optimal policy, run n episodes, display results.
    """
    hard_terminals = {GOAL: +1.0, HAZARD: -1.0, (2, 3): -1.0}

    V, iters = value_iteration(terminals=hard_terminals)
    policy   = extract_policy(V, terminals=hard_terminals)

    print("=" * 60)
    print("Task 5 — Harder Warehouse (second hazard at (2,3))")
    print("=" * 60)
    print(f"Value iteration converged in {iters} iterations.\n")
    display_grid(V, policy, terminals=hard_terminals)

    gr, hr, ar = run_episodes(policy, n=n, terminals=hard_terminals)
    print(f"Results over {n} episodes:")
    print(f"  Goal reached : {gr*100:.1f}%")
    print(f"  Hazard hit   : {hr*100:.1f}%")
    print(f"  Avg reward   : {ar:.4f}")
    print()
    print("Analysis:")
    print("  Adding a second hazard at (2,3) blocks the safe left-column route.")
    print("  The agent must find a narrower corridor to reach the goal, which")
    print("  slightly reduces goal-reach rate and pushes average reward lower.")
    print("  The policy adapts by routing through (1,y) or (3,y) more carefully,")
    print("  trading efficiency for continued hazard avoidance.")
    print()


# ---------------------------------------------------------------------------
# Main — Run All Experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    # ── Baseline: solve the standard MDP ──────────────────────────────────
    print("=" * 60)
    print("Solving the 4×4 Warehouse MDP  (γ=0.99, ε=1e-6)")
    print("=" * 60)
    V, iters = value_iteration(gamma=0.99)
    policy   = extract_policy(V, gamma=0.99)
    print(f"Converged in {iters} iterations.\n")
    display_grid(V, policy)

    # ── Task 1: Verify simulator ───────────────────────────────────────────
    print("=" * 60)
    print("Task 1 — Simulator Verification")
    print("=" * 60)
    verify_simulator(n=10_000)

    # ── Task 2: 1000 episodes with optimal policy ─────────────────────────
    print("=" * 60)
    print("Task 2 — 1000 Episodes with Optimal MDP Policy")
    print("=" * 60)
    gr, hr, ar = run_episodes(policy, n=1_000)
    print(f"  Goal reached : {gr*100:.1f}%")
    print(f"  Hazard hit   : {hr*100:.1f}%")
    print(f"  Avg reward   : {ar:.4f}")
    print()

    # ── Task 3: Comparison with naive greedy policy ───────────────────────
    print("=" * 60)
    print("Task 3 — Comparison: Optimal vs. Naive Greedy Policy")
    print("=" * 60)
    naive = greedy_policy()
    gr_n, hr_n, ar_n = run_episodes(naive, n=1_000)
    print(f"{'Policy':<16}  {'Goal%':>7}  {'Hazard%':>8}  {'Avg reward':>11}")
    print("-" * 48)
    print(f"{'MDP-optimal':<16}  {gr*100:>6.1f}%  {hr*100:>7.1f}%  {ar:>11.4f}")
    print(f"{'Naive greedy':<16}  {gr_n*100:>6.1f}%  {hr_n*100:>7.1f}%  {ar_n:>11.4f}")
    print()
    print("Analysis:")
    print("  The naive greedy policy always moves directly toward (4,4), which")
    print("  routes through column 4 — directly past the hazard at (4,3).")
    print("  Stochastic drift causes frequent hazard hits, dragging down the")
    print("  average reward. The MDP-optimal policy avoids column 4 until it")
    print("  reaches row 4, trading a slightly longer path for near-zero hazard rate.")
    print()

    # ── Task 4: Discount factor sweep ─────────────────────────────────────
    discount_experiment(gammas=(0.1, 0.5, 0.9, 0.99), n=1_000)

    # ── Task 5: Harder warehouse ───────────────────────────────────────────
    harder_warehouse_experiment(n=1_000)
