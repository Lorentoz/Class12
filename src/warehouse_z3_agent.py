"""
FOL Agent for the Hazardous Warehouse (Z3 Version)

Uses Z3's SMT solver with quantified first-order logic to reason about
safety and navigate the warehouse to retrieve the package.

This agent implements the same TELL/ASK loop as warehouse_kb_agent.py
but replaces the propositional ground encoding with Z3's FirstOrder Logic solver.
The physics rules are expressed as single quantified FOL sentences:

  ForAll L, Creaking(L) <=> Exists L', Adjacent(L,L') & Damaged(L')
  ForAll L, Rumbling(L) <=> Exists L', Adjacent(L,L') & Forklift(L')
  ForAll L, Safe(L) <=> ~Damaged(L) & ~Forklift(L)

This eliminates redundancy: same 3 rules for any grid size, instead of
48 biconditionals for 4x4, 300 for 10x10, etc.
"""

from collections import deque
from z3 import (
    Or, And, Not, Solver, unsat,
    DeclareSort, Function, BoolSort, Const, ForAll, Exists, Distinct,
)
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
    and check satisfiability. If unsat, the negated query is
    inconsistent with the KB --- meaning the KB entails the query.
    Pop restores the solver to its previous state.
    """
    solver.push()
    solver.add(Not(query))
    result = solver.check() == unsat
    solver.pop()
    return result


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
# Knowledge-Base Construction (Quantified FOL Encoding)
# ---------------------------------------------------------------------------

def build_warehouse_kb_fol(width=4, height=4, use_domain_closure=True):
    """Build a Z3 Solver using quantified first-order logic.

    The physics rules are expressed as single quantified sentences ---
    three rules total, independent of grid size:

        ForAll L, Creaking(L) == Exists L', Adjacent(L,L') & Damaged(L')
        ForAll L, Rumbling(L) == Exists L', Adjacent(L,L') & Forklift(L')
        ForAll L, Safe(L) == And(Not(Damaged(L)), Not(Forklift(L)))

    Structural facts (adjacency, domain closure) require enumeration,
    but these encode topology, not physics.

    Args:
        width, height: Grid dimensions
        use_domain_closure: If True, add domain closure axiom. Set to False
                          to investigate why domain closure is essential.

    Returns:
        (solver, loc, predicates) where:
          - solver: Z3 Solver with all constraints
          - loc: dict mapping (x,y) to Z3 Location constants
          - predicates: dict mapping predicate names to Z3 Function objects
    """
    # --- Declare Location sort and predicates ---
    Location = DeclareSort('Location')

    Damaged_fn = Function('Damaged', Location, BoolSort())
    Forklift_fn = Function('Forklift', Location, BoolSort())
    Creaking_fn = Function('Creaking', Location, BoolSort())
    Rumbling_fn = Function('Rumbling', Location, BoolSort())
    Safe_fn = Function('Safe', Location, BoolSort())
    Adjacent_fn = Function('Adjacent', Location, Location, BoolSort())

    # --- Location constants: one per grid square ---
    loc = {}
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            loc[(x, y)] = Const(f'L_{x}_{y}', Location)

    solver = Solver()

    # --- Domain closure (optional, but essential for correct reasoning) ---
    # Without this, ForAll(L, ...) ranges over phantom locations that don't
    # exist on the grid. These phantoms can "absorb" the blame for damage,
    # breaking process-of-elimination reasoning.
    L = Const('L', Location)  # Define L early for use in quantified rules
    if use_domain_closure:
        # Every location is one of our grid constants
        # Build the location list first to avoid scoping issues
        loc_list = [loc[(x, y)] for x in range(1, width + 1) for y in range(1, height + 1)]
        solver.add(ForAll(L,
            Or([L == loc_const for loc_const in loc_list])
        ))
        # All location constants are distinct
        solver.add(Distinct(list(loc.values())))

    # --- Adjacency facts (closed-world) ---
    # Enumerate all pairs: either adjacent or not.
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            adj_set = set(get_adjacent(x, y, width, height))
            for x2 in range(1, width + 1):
                for y2 in range(1, height + 1):
                    if (x2, y2) in adj_set:
                        solver.add(Adjacent_fn(loc[(x, y)], loc[(x2, y2)]))
                    else:
                        solver.add(Not(Adjacent_fn(loc[(x, y)], loc[(x2, y2)])))

    # --- Quantified physics rules ---
    # One sentence per rule, regardless of grid size.
    Lp = Const('Lp', Location)

    # Creaking iff damaged in adjacent square
    solver.add(ForAll(L,
        Creaking_fn(L) == Exists(Lp, And(Adjacent_fn(L, Lp), Damaged_fn(Lp)))
    ))

    # Rumbling iff forklift in adjacent square
    solver.add(ForAll(L,
        Rumbling_fn(L) == Exists(Lp, And(Adjacent_fn(L, Lp), Forklift_fn(Lp)))
    ))

    # Safe iff no damaged floor and no forklift
    solver.add(ForAll(L,
        Safe_fn(L) == And(Not(Damaged_fn(L)), Not(Forklift_fn(L)))
    ))

    # --- Initial knowledge ---
    solver.add(Safe_fn(loc[(1, 1)]))

    predicates = {
        'Creaking': Creaking_fn,
        'Rumbling': Rumbling_fn,
        'Safe': Safe_fn,
        'Damaged': Damaged_fn,
        'Forklift': Forklift_fn,
        'Adjacent': Adjacent_fn,
    }

    return solver, loc, predicates


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
# Z3 FOL Knowledge-Based Agent
# ---------------------------------------------------------------------------

class WarehouseZ3Agent:
    """A knowledge-based agent using Z3's quantified FOL for warehouse reasoning.

    Mirrors WarehouseKBAgent from warehouse_kb_agent.py with identical
    decision logic, path planning, and action conversion. The key difference
    is the reasoning engine: Z3 with quantified FOL rules instead of
    propositional ground encoding.

    Decision strategy (in priority order):
      1. If the beacon is detected, GRAB the package.
      2. If carrying the package, navigate to (1,1) and EXIT.
      3. Otherwise, explore the nearest safe unvisited square.
      4. If no safe unvisited square is reachable, return to (1,1) and EXIT.
    """

    def __init__(self, env, use_domain_closure=True):
        self.env = env
        self.solver, self.loc, self.preds = build_warehouse_kb_fol(
            env.width, env.height, use_domain_closure=use_domain_closure
        )
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
        """Translate a Percept into Z3 FOL assertions and add to solver."""
        L = self.loc[(self.x, self.y)]
        if percept.creaking:
            self.solver.add(self.preds['Creaking'](L))
        else:
            self.solver.add(Not(self.preds['Creaking'](L)))
        if percept.rumbling:
            self.solver.add(self.preds['Rumbling'](L))
        else:
            self.solver.add(Not(self.preds['Rumbling'](L)))

    # ----- Safety queries ----------------------------------------------------

    def update_safety(self):
        """Check entailment for every square whose status is still unknown."""
        for x in range(1, self.env.width + 1):
            for y in range(1, self.env.height + 1):
                pos = (x, y)
                if pos in self.known_safe or pos in self.known_dangerous:
                    continue
                L = self.loc[pos]
                if z3_entails(self.solver, self.preds['Safe'](L)):
                    self.known_safe.add(pos)
                elif z3_entails(self.solver, Not(self.preds['Safe'](L))):
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

    agent = WarehouseZ3Agent(env)
    agent.run(verbose=True)
