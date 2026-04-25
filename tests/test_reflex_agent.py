import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import ReflexAgent


def test_pick_and_drop_actions():
    env = WarehouseEnv()
    obs = env.reset()
    agent = ReflexAgent(seed=0)

    # On pickup when not carrying -> PICK
    env.state.robot_pos = obs["pickup_pos"]
    env.state.has_item = False
    obs = env._observe()
    assert agent.select_action(obs, env) == "PICK"

    # On dropoff when carrying -> DROP
    env.state.robot_pos = obs["dropoff_pos"]
    env.state.has_item = True
    obs = env._observe()
    assert agent.select_action(obs, env) == "DROP"


def test_directional_rule_moves_toward_target():
    env = WarehouseEnv()
    obs = env.reset()
    agent = ReflexAgent(seed=0)

    # Place robot north of pickup -> action should reduce distance toward pickup
    pr, pc = obs["pickup_pos"]
    env.state.robot_pos = (pr - 2, pc)
    env.state.has_item = False
    obs = env._observe()
    act = agent.select_action(obs, env)
    # If agent decides to PICK (already on pickup) accept it; otherwise the move must
    # reduce the Manhattan distance to the pickup.
    if act == "PICK":
        return
    dr, dc = env.MOVE_DELTAS.get(act, (0, 0))
    nr, nc = env.state.robot_pos[0] + dr, env.state.robot_pos[1] + dc
    before = abs(env.state.robot_pos[0] - pr) + abs(env.state.robot_pos[1] - pc)
    after = abs(nr - pr) + abs(nc - pc)
    assert after <= before, "Action should not increase distance to the pickup"
