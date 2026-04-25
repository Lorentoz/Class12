"""
Knowledge-Based Agent for the Hazardous Warehouse (Propositional Z3)

Uses Z3's SMT solver with grounded propositional variables to reason
about safety and navigate the warehouse to retrieve the package.

This agent implements the TELL/ASK loop:
  1. TELL the solver about percepts (solver.add)
  2. ASK via entailment check (push/Not(query)/check/pop)
  3. Plan a path through safe squares toward the goal
  4. Execute actions and repeat

The knowledge base encodes the physics of the warehouse using one Bool
variable per square per predicate:
  - Creaking at (x,y) iff damaged floor in an adjacent square
  - Rumbling at (x,y) iff forklift in an adjacent square
  - A square is safe iff it has no damaged floor and no forklift
"""

from collections import deque

from z3 import Bool, Or, And, Not, Solver, unsat
from hazardous_warehouse_env import (
    HazardousWarehouseEnv,
    Action,
    Direction,
)


# ---------------------------------------------------------------------------
# Z3 Entailment Check
# ---------------------------------------------------------------------------

def z3_entails(solver, query):
    """Check whether the solver's current assertions entail *query*.

    Uses the refutation method: push a checkpoint, assert Not(query),
    and check satisfiability.  If unsat, the negated query is
    inconsistent with the KB --- meaning the KB entails the query.
    Pop restores the solver to its previous state.
    """
    solver.push()
    solver.add(Not(query))
    result = solver.check() == unsat
    solver.pop()
    return result


# ---------------------------------------------------------------------------
# Propositional Variable Helpers
# ---------------------------------------------------------------------------

def damaged(x, y):
    """Z3 Bool variable: damaged floor at (x, y)."""
    return Bool(f'D_{x}_{y}')


def forklift_at(x, y):
    """Z3 Bool variable: forklift at (x, y)."""
    return Bool(f'F_{x}_{y}')


def creaking_at(x, y):
    """Z3 Bool variable: creaking perceived at (x, y)."""
    return Bool(f'C_{x}_{y}')


def rumbling_at(x, y):
    """Z3 Bool variable: rumbling perceived at (x, y)."""
    return Bool(f'R_{x}_{y}')


def safe(x, y):
    """Z3 Bool variable: square (x, y) is safe to enter."""
    return Bool(f'OK_{x}_{y}')


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def get_adjacent(x, y, width=4, height=4):
    """Return the list of (x, y) positions adjacent to (x, y)."""
    result = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 1 <= nx <= width and 1 <= ny <= height:
            result.append((nx, ny))
    return result


# ---------------------------------------------------------------------------
# Knowledge-Base Construction
# ---------------------------------------------------------------------------

def build_warehouse_kb(width=4, height=4):
    """Build a Z3 Solver populated with the physics of the warehouse.

    The solver contains three kinds of constraints for every square (x, y):

    1. Creaking biconditional
       C_x_y == Or(D_a1_b1, D_a2_b2, ...)
       where (a_i, b_i) are the squares adjacent to (x, y).

    2. Rumbling biconditional
       R_x_y == Or(F_a1_b1, F_a2_b2, ...)

    3. Safety biconditional
       OK_x_y == And(Not(D_x_y), Not(F_x_y))

    Z3's native == operator handles biconditionals directly ---
    no manual CNF conversion is needed.

    It also encodes the initial knowledge that the starting square (1, 1)
    has no damaged floor and no forklift.
    """
    solver = Solver()

    # The starting square is safe.
    solver.add(Not(damaged(1, 1)))
    solver.add(Not(forklift_at(1, 1)))

    for x in range(1, width + 1):
        for y in range(1, height + 1):
            adj = get_adjacent(x, y, width, height)

            # --- Creaking rule ---
            # Creaking iff at least one adjacent square has damaged floor
            solver.add(creaking_at(x, y) == Or([damaged(a, b) for a, b in adj]))

            # --- Rumbling rule ---
            # Rumbling iff at least one adjacent square has the forklift
            solver.add(rumbling_at(x, y) == Or([forklift_at(a, b) for a, b in adj]))

            # --- Safety rule ---
            # A square is safe iff it has no damaged floor and no forklift
            solver.add(
                safe(x, y) == And(Not(damaged(x, y)), Not(forklift_at(x, y)))
            )

    return solver


# ---------------------------------------------------------------------------
# Turning Helpers
# ---------------------------------------------------------------------------

_DIRECTION_ORDER = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]


def _direction_index(d):
    return _DIRECTION_ORDER.index(d)


def turns_between(current, target):
    """Return a list of TURN_LEFT / TURN_RIGHT actions to face *target*.

    Chooses the shortest rotation direction.
    """
    if current == target:
        return []
    ci = _direction_index(current)
    ti = _direction_index(target)
    right_steps = (ti - ci) % 4   # clockwise
    left_steps = (ci - ti) % 4    # counter-clockwise
    if right_steps <= left_steps:
        return [Action.TURN_RIGHT] * right_steps
    else:
        return [Action.TURN_LEFT] * left_steps


def delta_to_direction(dx, dy):
    """Map a movement delta to the Direction enum."""
    return {
        (0, 1): Direction.NORTH,
        (0, -1): Direction.SOUTH,
        (1, 0): Direction.EAST,
        (-1, 0): Direction.WEST,
    }[(dx, dy)]


# ---------------------------------------------------------------------------
# Knowledge-Based Agent
# ---------------------------------------------------------------------------

class WarehouseKBAgent:
    """A knowledge-based agent for the Hazardous Warehouse.

    The agent maintains:
      - A Z3 Solver with physics rules and accumulated percepts
      - Sets of known-safe and known-dangerous squares
      - A queue of planned actions
      - Its own position, direction, and inventory state

    Decision strategy (in priority order):
      1. If the beacon is detected, GRAB the package.
      2. If carrying the package, navigate to (1,1) and EXIT.
      3. Otherwise, explore the nearest safe unvisited square.
      4. If no safe unvisited square is reachable, return to (1,1) and EXIT.
    """

    def __init__(self, env):
        self.env = env
        self.solver = build_warehouse_kb(env.width, env.height)
        self.x = 1
        self.y = 1
        self.direction = Direction.EAST
        self.has_package = False
        self.visited = {(1, 1)}
        self.known_safe = {(1, 1)}
        self.known_dangerous = set()
        self.action_queue = []
        self.step_count = 0

    # ----- Percepts ----------------------------------------------------------

    def tell_percepts(self, percept):
        """Translate a Percept into Z3 assertions and TELL the solver."""
        x, y = self.x, self.y
        if percept.creaking:
            self.solver.add(creaking_at(x, y))
        else:
            self.solver.add(Not(creaking_at(x, y)))
        if percept.rumbling:
            self.solver.add(rumbling_at(x, y))
        else:
            self.solver.add(Not(rumbling_at(x, y)))

    # ----- Safety queries ----------------------------------------------------

    def update_safety(self):
        """ASK the solver about every square whose status is still unknown."""
        for x in range(1, self.env.width + 1):
            for y in range(1, self.env.height + 1):
                pos = (x, y)
                if pos in self.known_safe or pos in self.known_dangerous:
                    continue
                if z3_entails(self.solver, safe(x, y)):
                    self.known_safe.add(pos)
                elif z3_entails(self.solver, Not(safe(x, y))):
                    self.known_dangerous.add(pos)

    # ----- Path planning -----------------------------------------------------

    def plan_path(self, start, goal_set):
        """BFS through known-safe squares from *start* to any cell in *goal_set*.

        Returns a list of (x, y) positions forming the path (including
        *start* and the reached goal), or None if no path exists.
        """
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            (cx, cy), path = queue.popleft()
            if (cx, cy) in goal_set:
                return path
            for nx, ny in get_adjacent(cx, cy, self.env.width, self.env.height):
                if (nx, ny) not in seen and (nx, ny) in self.known_safe:
                    seen.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return None

    def path_to_actions(self, path):
        """Convert a position path into a sequence of Actions.

        Returns (actions, final_direction) where *actions* is the list of
        TURN_LEFT / TURN_RIGHT / FORWARD actions and *final_direction* is
        the direction the robot faces after executing them all.
        """
        actions = []
        direction = self.direction
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            target_dir = delta_to_direction(dx, dy)
            actions.extend(turns_between(direction, target_dir))
            actions.append(Action.FORWARD)
            direction = target_dir
        return actions, direction

    # ----- Decision logic ----------------------------------------------------

    def choose_action(self, percept):
        """Select the next action based on the current state of knowledge."""
        # Execute queued actions first (from a multi-step plan).
        if self.action_queue:
            return self.action_queue.pop(0)

        # 1. If the beacon is on, grab the package.
        if percept.beacon and not self.has_package:
            return Action.GRAB

        # 2. If carrying the package, navigate home and exit.
        if self.has_package:
            if (self.x, self.y) == (1, 1):
                return Action.EXIT
            path = self.plan_path((self.x, self.y), {(1, 1)})
            if path and len(path) > 1:
                actions, _ = self.path_to_actions(path)
                self.action_queue = actions[1:]
                return actions[0]
            # Already at (1,1) or can't find path — just exit.
            return Action.EXIT

        # 3. Explore the nearest safe unvisited square.
        safe_unvisited = self.known_safe - self.visited
        if safe_unvisited:
            path = self.plan_path((self.x, self.y), safe_unvisited)
            if path and len(path) > 1:
                actions, _ = self.path_to_actions(path)
                self.action_queue = actions[1:]
                return actions[0]

        # 4. Nothing left to explore — go home and exit.
        if (self.x, self.y) == (1, 1):
            return Action.EXIT
        path = self.plan_path((self.x, self.y), {(1, 1)})
        if path and len(path) > 1:
            actions, _ = self.path_to_actions(path)
            self.action_queue = actions[1:]
            self.action_queue.append(Action.EXIT)
            return actions[0]
        return Action.EXIT

    # ----- Execution ---------------------------------------------------------

    def execute_action(self, action):
        """Send *action* to the environment and update internal bookkeeping."""
        percept, reward, done, info = self.env.step(action)

        if action == Action.FORWARD and not percept.bump:
            dx, dy = self.direction.delta()
            self.x += dx
            self.y += dy
            self.visited.add((self.x, self.y))
        elif action == Action.TURN_LEFT:
            self.direction = self.direction.turn_left()
        elif action == Action.TURN_RIGHT:
            self.direction = self.direction.turn_right()
        elif action == Action.GRAB and info.get("grabbed"):
            self.has_package = True

        self.step_count += 1
        return percept, reward, done, info

    # ----- Main loop ---------------------------------------------------------

    def run(self, verbose=True):
        """Run the full perceive-tell-ask-act loop until the episode ends."""
        # Process the initial percept at (1, 1).
        percept = self.env._last_percept
        self.tell_percepts(percept)
        self.update_safety()

        if verbose:
            print(f"Start at ({self.x},{self.y}) facing {self.direction.name}")
            print(f"  Percept: {percept}")
            print(f"  Known safe: {sorted(self.known_safe)}")

        while True:
            action = self.choose_action(percept)
            percept, reward, done, info = self.execute_action(action)

            if verbose:
                print(f"\nStep {self.step_count}: {action.name}")
                print(f"  Position: ({self.x},{self.y}), Facing: {self.direction.name}")
                print(f"  Percept: {percept}")
                print(f"  Info: {info}")

            if done:
                if verbose:
                    print(f"\n{'=' * 40}")
                    print(f"Episode ended.  Reward: {self.env.total_reward:.0f}")
                    print(f"Steps taken: {self.step_count}")
                    success = info.get("exit") == "success"
                    print(f"Success: {success}")
                return

            # After moving to a new square, tell percepts and re-query safety.
            if action == Action.FORWARD and not percept.bump:
                self.tell_percepts(percept)
                self.update_safety()
                if verbose:
                    print(f"  Known safe: {sorted(self.known_safe)}")
                    print(f"  Known dangerous: {sorted(self.known_dangerous)}")


# ---------------------------------------------------------------------------
# Main — run on the example layout from the textbook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from hazardous_warehouse_viz import configure_rn_example_layout

    env = HazardousWarehouseEnv(seed=0)
    configure_rn_example_layout(env)

    print("True state (hidden from the agent):")
    print(env.render(reveal=True))
    print()

    agent = WarehouseKBAgent(env)
    agent.run(verbose=True)
