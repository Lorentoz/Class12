# Warehouse Pathfinding: UCS vs A* Search Algorithms

## Overview

This project implements and compares two fundamental search algorithms for optimal pathfinding in the warehouse environment:

- **Uniform Cost Search (UCS)**: Expands nodes in order of increasing path cost `g(n)`, guaranteeing optimality for non-negative edge costs.
- **A* Search**: Combines path cost `g(n)` and heuristic estimate `h(n)`, using evaluation function `f(n) = g(n) + h(n)`. Expands nodes in order of lowest `f(n)`.

## Key Files

### Core Implementation

- **`src/ucs_pathfinder.py`** — Uniform Cost Search implementation
  - Returns: `(path, statistics)` where statistics include nodes expanded, path length, computation time
  - Uses priority queue ordered by path cost `g(n)`

- **`src/astar_pathfinder.py`** — A* Search implementation
  - Uses **Manhattan distance heuristic**: `h(n) = |x_n - x_goal| + |y_n - y_goal|`
  - Returns: `(path, statistics)` with same format as UCS
  - Admissible heuristic guarantees optimality

- **`src/compare_search.py`** — Comparison script
  - Runs UCS and A* on 10 randomized warehouse configurations
  - Each trial: finds paths start→pickup and pickup→dropoff
  - Generates comparison plots and summary statistics

### Tests

- **`tests/test_search_algorithms.py`** — Comprehensive search tests
  - Verifies both algorithms find valid paths
  - Confirms optimality: both find paths of identical length
  - Validates A* efficiency: expands fewer (or equal) nodes than UCS

## Quick Start

### Run a single comparison trial

```bash
python src/compare_search.py --trials 10 --seed 0 --randomize --save out/search_compare.png
```

Expected output:
```
Running 10 comparison trials...
  Trial 0: UCS nodes=53, A* nodes=16
  Trial 1: UCS nodes=44, A* nodes=21
  ...
  
A* efficiency gain: ~50% fewer nodes expanded than UCS
```

### Run all tests

```bash
python scripts/run_tests.py
```

## Results Summary (Typical)

### 10-Trial Comparison (20 search tasks: 2 per trial)

| Metric | UCS | A* | Gain |
|--------|-----|-----|------|
| Mean nodes expanded | 22.3 | 10.0 | **55.2% fewer** |
| Mean time (ms) | 0.069 | 0.033 | **51.9% faster** |
| Mean path length | 7.2 | 7.2 | **Identical (optimal)** |

### Key Observations

1. **Optimality**: Both UCS and A* find paths of identical length (both optimal).
2. **Efficiency**: A* expands ~50% fewer nodes than UCS due to admissible Manhattan heuristic.
3. **Speed**: A* is roughly 2× faster in wall-clock time on these problems.

## Algorithm Details

### Uniform Cost Search (UCS)

```
Expand: lowest path cost g(n) first
Frontier: priority queue keyed by g(n)
Explored set: tracks visited nodes with lowest cost found
Optimality: ✓ Complete and optimal for non-negative costs
```

### A* Search

```
Expand: lowest evaluation f(n) = g(n) + h(n) first
Heuristic: Manhattan distance (admissible, never overestimates)
Explored set: tracks visited nodes
Optimality: ✓ Complete and optimal with admissible heuristic
Efficiency: ✓ Expands far fewer nodes than UCS
```

## Visualizations

Run the comparison to generate:

1. **Nodes Expanded**: Bar chart comparing mean nodes expanded by each algorithm.
2. **Computation Time**: Bar chart of wall-clock time (ms) per search task.
3. **Path Length**: Bar chart of mean steps in solution paths (should be identical).

Output saved to: `out/search_compare.png`

## Notes

- **State space**: Positions (r, c) in the warehouse grid
- **Actions**: Cardinal moves (N, S, E, W), cost 1 each
- **Goal test**: Reach target position
- **Grid**: Default 7×12 warehouse with walls, obstacles, pickup/dropoff locations

## Extensions

Future improvements:
- Implement Weighted A* for speed/quality trade-offs
- Add more sophisticated heuristics (e.g., pattern databases)
- Compare against greedy best-first search
- Test on larger grids / complex layouts
