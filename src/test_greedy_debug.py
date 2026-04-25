from warehouse_env import WarehouseEnv
from warehouse_agent_greedy import GreedyManhattanAgent


def run_episode(seed=None, max_steps=200, randomize=False):
    env = WarehouseEnv()
    obs = env.reset(randomize=randomize)
    agent = GreedyManhattanAgent(seed=seed)
    agent.reset()
    logs = []
    for step in range(max_steps):
        pos = obs["robot_pos"]
        has_item = obs["has_item"]
        pickup = obs["pickup_pos"]
        dropoff = obs["dropoff_pos"]
        act = agent.select_action(obs, env)
        obs, r, done, trunc, info = env.step(act)
        logs.append((step, pos, has_item, pickup, dropoff, act))
        # If agent is on dropoff but not carrying, log it
        if pos == dropoff and not has_item:
            print(f"Step {step}: Agent on DROP at {pos} but not carrying; action: {act}")
        if done or trunc:
            break
    return logs


if __name__ == "__main__":
    # Run a few episodes with different seeds to try to reproduce the behavior.
    for s in [0, 1, 2, 3, 4, None]:
        print("=== Episode seed", s, "===")
        run_episode(seed=s)
