import sys
from pathlib import Path

# Ensure src is importable when running tests from project root
SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from warehouse_env import WarehouseEnv
from warehouse_agent_greedy import GreedyManhattanAgent


def test_move_off_dropoff():
    env = WarehouseEnv()
    obs = env.reset()
    agent = GreedyManhattanAgent(seed=0)

    # Put robot on dropoff without an item
    env.state.robot_pos = obs["dropoff_pos"]
    env.state.has_item = False
    obs = env._observe()

    act = agent.select_action(obs, env)
    assert act in {"N", "E", "S", "W"}, "Agent should choose a cardinal move to leave dropoff"

    # The move must be valid (not into a wall)
    dr, dc = env.MOVE_DELTAS[act]
    nr, nc = env.state.robot_pos[0] + dr, env.state.robot_pos[1] + dc
    assert not env._is_wall(nr, nc)


def test_loop_detection_triggers_escape():
    env = WarehouseEnv()
    obs = env.reset()
    agent = GreedyManhattanAgent(seed=1)

    pos = obs["robot_pos"]
    # Fill recent history with the current position to simulate a loop.
    for _ in range(agent.loop_history_size):
        agent.recent.append(pos)

    act = agent.select_action(obs, env)
    # After selecting action while in recent history, escape_counter should be set.
    assert agent.escape_counter > 0
    assert act in {"N", "E", "S", "W"}
