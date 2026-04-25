"""Run a single episode with the Greedy Manhattan agent and visualize results.

Usage (from project root):
    python src/run_episode.py --episodes 1 --seed 0 --randomize --replay

The script:
- Creates a WarehouseEnv and GreedyManhattanAgent
- Resets environment (optionally randomize pickup/dropoff/start)
- Steps until done or truncated
- Collects frames and metrics (rewards, battery, distances)
- Replays the animation using `warehouse_viz.replay_animation` (when available)
- Optionally saves frames as SVGs

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure local src package files are importable when running this script directly
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from warehouse_env import WarehouseEnv
from warehouse_agent_greedy import GreedyManhattanAgent

try:
    from warehouse_viz import replay_animation, save_frames_to_svg
except Exception:
    replay_animation = None  # type: ignore
    save_frames_to_svg = None  # type: ignore


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_episode(env: WarehouseEnv, agent: GreedyManhattanAgent, randomize: bool = False, max_steps: int | None = None) -> Dict:
    """Run one episode and return frames and metrics."""
    obs = env.reset(randomize=randomize)
    agent.reset()

    frames: List[List[List[str]]] = [env.render_grid()]

    # Initialize metrics so their lengths match `frames` (one entry per frame).
    init_pick = manhattan(obs["robot_pos"], obs["pickup_pos"]) if obs.get("pickup_pos") is not None else 0
    init_drop = manhattan(obs["robot_pos"], obs["dropoff_pos"]) if obs.get("dropoff_pos") is not None else 0
    metrics = {
        "rewards": [0.0],
        "battery": [obs["battery"]],
        "dist_pickup": [init_pick],
        "dist_dropoff": [init_drop],
    }

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    limit = max_steps or env.max_steps
    while steps < limit:
        act = agent.select_action(obs, env)
        obs, r, terminated, truncated, info = env.step(act)

        frames.append(env.render_grid())
        total_reward += r
        metrics["rewards"].append(r)
        metrics["battery"].append(obs["battery"])
        # Distances
        if obs.get("pickup_pos") is not None:
            metrics["dist_pickup"].append(manhattan(obs["robot_pos"], obs["pickup_pos"]))
        else:
            metrics["dist_pickup"].append(0)
        if obs.get("dropoff_pos") is not None:
            metrics["dist_dropoff"].append(manhattan(obs["robot_pos"], obs["dropoff_pos"]))
        else:
            metrics["dist_dropoff"].append(0)

        steps += 1
        if terminated or truncated:
            break

    return {
        "frames": frames,
        "metrics": metrics,
        "total_reward": total_reward,
        "steps": steps,
        "final_battery": env.state.battery,
        "terminated": terminated,
        "truncated": truncated,
        "env": env,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--randomize", action="store_true", help="Randomize pickup/dropoff and start pos")
    p.add_argument("--replay", action="store_true", help="Replay the animation (requires matplotlib)")
    p.add_argument("--save-svg", type=str, default="", help="Directory to save SVG frames for each episode")
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args(argv)

    results = []
    for i in range(args.episodes):
        env = WarehouseEnv()
        agent = GreedyManhattanAgent(seed=(args.seed + i) if args.seed is not None else None)
        print(f"Episode {i}: randomize={args.randomize} seed={args.seed + i}")
        res = run_episode(env, agent, randomize=args.randomize, max_steps=args.max_steps)
        results.append(res)
        print(f"  Steps: {res['steps']}, Total reward: {res['total_reward']:.2f}, Battery: {res['final_battery']}, terminated={res['terminated']}, truncated={res['truncated']}")

        if args.save_svg:
            outdir = Path(args.save_svg) / f"episode_{i:03d}"
            outdir.mkdir(parents=True, exist_ok=True)
            if save_frames_to_svg:
                save_frames_to_svg(res["frames"], str(outdir))
                print(f"  Saved frames to {outdir}")
            else:
                print("  matplotlib not available; skipping SVG export.")

        if args.replay:
            if replay_animation:
                print("  Replaying animation... (press space to pause)")
                replay_animation(res["frames"], metrics=res["metrics"])  # interactive
            else:
                print("  matplotlib not available; cannot replay animation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
