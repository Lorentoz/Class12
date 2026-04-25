# Problem 3.6  Propositional vs. First-Order Expressiveness

This document works through the four parts of Problem 3.6.  The goal is to
understand the expressiveness gap between propositional and first-order
logic in the context of the hazardous warehouse domain.

---

## Part A: Propositional Encoding Cost

Consider an `n \times n` warehouse grid.  We wish to count the number of
propositional sentences required to encode the following elements.

1. **Adjacency relation**
   - In a grid each square has up to four neighbors (north, south, east, west).
   - For each ordered pair of distinct locations `(i,j)` and `(k,l)` we must
     assert either `adjacent_i_j_k_l` or its negation.  A simpler method is to
     assert positive adjacency only for actual neighbours and assert negative
     cases implicitly by closed-world assumption, but the KB construction in
     `warehouse_kb_agent.py` writes both positive and negative facts.  That
     yields `O(n^4)` sentences because there are `n^2` locations and each pair
     of locations is considered.
   - With only positive facts, the count is proportional to the number of
     adjacent pairs: roughly `2n(n-1)` (horizontal plus vertical edges), which is
     `O(n^2)`.  Either way, adjacency natural scaling is polynomial in the
     area or its square.

2. **Creaking rule**
   - For each location `L` we need a biconditional
     ``creaking_L <-> (damaged_adj1 or damaged_adj2 or ...)``
   - There are `n^2` locations, and each rule mentions up to four adjacent
     positions, but the number of separate sentences is exactly `n^2`.
   - Hence the cost is `f(n)=n^2`.

3. **Safety rule**
   - Similarly, for each location `L` a sentence `safe_L <-> (¬damaged_L and
     ¬forklift_L)` is required.
   - Again, one sentence per location: `f(n)=n^2`.

### Growth to 100

- For `n=100`, the grid contains `10,000` locations.  
- **Adjacency facts:** if enumerating every ordered pair, `10,000^2 = 100
  million` sentences; if only positive neighbor relations, about `2 * 100 * 99
  = 19,800` sentences.  
- **Creaking rules:** 10,000 sentences.
- **Safety rules:** 10,000 sentences.

Propositional KB size thus grows quadratically or quartically depending on
encoding choices; either way, `n=100` is burdensome, though the physics rules
remain manageable (10k each).  Larger warehouses quickly become impractical.

---

## Part B: First-Order Logic Comparison

With FOL we can express the same rules without grounding over locations.

- **Adjacency relation** cannot be written as a single quantified sentence that
  enumerates all pairs unless we introduce a predicate `Adjacent(x,y)` and then
  add axioms describing its properties or enumerate only the true pairs.  The
  natural FOL encoding still requires one sentence per adjacency fact, so
  adjacency remains `O(n^2)` positive assertions plus negative closure if we add
  them explicitly.  It does not improve asymptotic complexity, though it allows
  symbolic reasoning about adjacency (e.g. `forall x. Adjacent(x,x) -> False`)
  if desirable.

- **Creaking rule** becomes a single formula:
  ```
  ForAll L. Creaking(L) <-> Exists L'. Adjacent(L, L') & Damaged(L')
  ```
  Regardless of grid size, this is one sentence.

- **Safety rule** likewise is one sentence:
  ```
  ForAll L. Safe(L) <-> (¬Damaged(L) & ¬Forklift(L))
  ```

Thus for any `n`, the number of FOL sentences required for physics rules is
constant (three statements including the two rules and optionally the domain
closure axiom).  Adjacency still scales with the number of true/false facts but
not with the rules themselves.

---

## Part C: Translation Challenge

### FOL Statement

The sentence "There is exactly one damaged floor section in the entire
warehouse" can be expressed concisely in FOL as:

```text
Exists L. Damaged(L) & ForAll M. (Damaged(M) -> M = L)
```

or equivalently with a two-part formulation using a uniqueness predicate:

```text
Exists L. Damaged(L) & ForAll M,N. ((Damaged(M) & Damaged(N)) -> M = N)
```

### Propositional Encoding for 3×3 Grid

We must introduce a propositional atom for each location, say
`damaged_1_1` ... `damaged_3_3`.  The statement then becomes:

```
(damaged_1_1 ∨ ... ∨ damaged_3_3)                           # at least one
and
for every distinct pair (i,j) ≠ (k,l), ¬(damaged_i_j & damaged_k_l)  # at most one
```
which expands to:
```
**(1)** 9 disjuncts
**(2)** choose(9,2) = 36 pairwise exclusion clauses
```

So the propositional encoding uses `9 + 36 = 45` clauses for the 3×3 case.
For an `n×n` grid with `m = n^2` locations this generalizes to
`m` disjunctions plus `m(m-1)/2` exclusion clauses, i.e. `O(n^4)` sentences.
Thus the complexity scales quartically: extremely expensive once `n` grows
beyond a handful.

It is possible to express the uniqueness at the propositional level, but it
requires explicit enumeration of all potential pairs of distinct locations, and
there is no succinct general formula other than writing out the combinations.
This explosion exemplifies the expressiveness gap: FOL can state uniqueness in
three symbols, but propositional demands combinatorial blow‑up.

---

## Part D: Practical Implications

### 100×100 Warehouse

- **Propositional feasibility:** 10,000 variables and tens of millions of
  sentences (especially if adjacency is fully enumerated) make KB construction
  and SAT solving very slow.  Purely propositional encoding is impractical for
  real-time reasoning in such a large space; memory and solver time would grow
  prohibitively.  One could restrict to local regions or use SAT modulo
  theories, but the direct encoding is not workable.

- **FOL restrictions for tractability:** To keep reasoning tractable in first-
  order logic, we would limit quantification and avoid function symbols.  Using
  a domain‑restricted form (e.g. a finite Herbrand universe) and decidable
  fragments such as the effectively propositional (EPR) fragment or guarded
  fragments helps.  Restricting to monadic predicates or limiting variable
  arity also improves performance.  In practice, we can combine FOL rules with
  domain closure and use a solver like Z3 that handles quantifiers efficiently
  on finite domains.  Caching entailment results and incremental solving help
  manage complexity.

- **Hybrid Approach:**
  1. **FOL for general rules:** Express generic physics rules once using
     quantifiers, keeping the KB size constant regardless of grid size.
  2. **Propositionalization for local reasoning:** When the agent enters a
     particular neighborhood, ground the relevant predicates (damage, safety,
     adjacency) for that region only, constructing a small propositional KB for
     SAT/graph search.  This local grounding avoids large global SAT problems.
  3. **Incremental updates:** Use FOL solver to maintain global invariants and
     derive new facts; when a new percept is observed, add a few ground facts
     to a propositional sub-KB and resolve locally.

Such an architecture leverages FOL’s expressiveness to keep the specification
concise and uses propositional reasoning where performance matters most.  The
current project already follows this hybrid pattern: physics rules are FOL,
adjacency is grounded, and the agent queries safety by grounding a single
location at a time.

---

*End of Problem 3.6 analysis.*
