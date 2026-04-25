# Class5 Knowledge-Based Agent Project - Deliverables Index

## Quick Start

To run the knowledge-based agent:
```bash
cd "c:\Users\XITRI\Desktop\SMU-Codes\Class5"
python src/warehouse_kb_agent.py
```

To run comprehensive tests (Tasks 1-3):
```bash
python test_kb_agent_tasks.py
```

Expected results:
- Tests: All assertions pass ✓
- Agent: Success=True, Reward=978, Steps=23 ✓

---

## Project Files

### Source Code (in `src/` directory)

#### 1. **warehouse_kb_agent.py** (15 KB, 404 lines)
   - Core knowledge-based agent implementation
   - Z3 entailment checking function
   - Propositional variable helpers (damaged, forklift_at, creaking_at, etc.)
   - Knowledge base construction with physics rules
   - Path planning via BFS
   - Complete TELL/ASK/PLAN/ACT agent loop
   - **Status**: ✓ Fully functional

#### 2. **hazardous_warehouse_env.py** (16 KB, 384 lines)
   - Warehouse environment simulation
   - Direction, Action, Percept enums
   - HazardousWarehouseEnv class with step logic
   - Reward tracking and state management
   - ASCII rendering for visualization
   - **Status**: ✓ Fully functional (restored)

### Test & Validation (in root directory)

#### 3. **test_kb_agent_tasks.py** (5.1 KB, 179 lines)
   - Comprehensive test suite for Tasks 1-3
   - Tests Z3 biconditional creation
   - Validates z3_entails() function
   - Verifies propositional symbol helpers
   - Tests warehouse KB construction  
   - Replicates textbook reasoning walkthrough (Section 3.2)
   - **Status**: ✓ All 13 assertions pass

### Documentation (in root directory)

#### 4. **KB_AGENT_RESULTS.md** (20 KB)
   - Complete results for all 6 tasks
   - Detailed Task 1-3 implementation and test results
   - Manual reasoning trace with Z3 queries and entailment checks
   - Task 4 architecture description
   - Task 5 agent execution results (978 reward, 23 steps)
   - Task 6 reflection on conservative behavior, stuck scenarios, and improvements
   - Installation instructions
   - Architecture overview
   - Complexity analysis
   - References to AIMA textbook
   - **Status**: ✓ Comprehensive documentation

#### 5. **SUBMISSION_SUMMARY.md** (This document summary)
   - Checklist of all completed tasks
   - Validation results
   - File listing and descriptions
   - How to run instructions
   - Key statistics
   - **Status**: ✓ Complete

#### 6. **README.md** (in root, auto-generated)
   - Project overview and setup guide

---

## Task Completion Status

| Task | Component | Status | Evidence |
|------|-----------|--------|----------|
| 1 | Z3 biconditionals & entailment | ✓ COMPLETE | test_kb_agent_tasks.py passes |
| 2 | Propositional symbols & KB | ✓ COMPLETE | KB builds, is satisfiable |
| 3 | Manual reasoning (textbook) | ✓ COMPLETE | Exact reasoning trace matches |
| 4 | Complete agent loop | ✓ COMPLETE | warehouse_kb_agent.py (404 lines) |
| 5 | Testing on example layout | ✓ COMPLETE | Agent reward=978, success=True |
| 6 | Reflection & analysis | ✓ COMPLETE | KB_AGENT_RESULTS.md (detailed) |

---

## Test Results Summary

### Task 1-3 Tests

```
Task 1: Z3 Setup and Exploration
  ✓ Boolean variables with biconditionals  
  ✓ z3_entails() refutation method
  ✓ Entailment queries (True/False)

Task 2: Propositional Symbols and Warehouse KB  
  ✓ All 5 symbol helpers defined
  ✓ Adjacency computation
  ✓ KB construction and satisfiability

Task 3: Manual Reasoning - Textbook Walkthrough
  ✓ Step 1: Adjacent squares proven safe
  ✓ Step 2: Ambiguous squares identified as UNKNOWN
  ✓ Step 3: Chain of inference identifies hazard locations
  
Result: All 13 assertions PASSED [SUCCESS]
```

### Task 5 - Full Agent Execution

```
Configuration:
  - Grid: 4×4
  - Damaged floors: (3,1), (3,3)
  - Forklift: (2,3)
  - Package: (2,3)

Agent Performance:
  - Episode succeeded: ✓ TRUE
  - Package retrieved: ✓ TRUE  
  - Exited successfully: ✓ TRUE
  - Total reward: 978
  - Steps taken: 23
  - Success rate: 100% (1/1 runs)
```

---

## Key Implementation Details

### Architecture
```
┌─ Knowledge Base (Z3 Solver)
├─ Physics Rules (creaking, rumbling, safety)
├─ Accumulated Percepts (TELL operations)
├─ Entailment Checking (z3_entails)
├─ Safety Classification (SAFE/DANGEROUS/UNKNOWN)  
├─ Path Planning (BFS through safe squares)
├─ Decision Logic (priority-based action selection)
└─ Action Execution (send to environment)
```

### Z3 Integration
- **Variables per square**: 4 (damaged, forklift, creaking, rumbling, safe)
- **Total variables (4×4 grid)**: 64
- **Biconditional rules**: 48
- **Method**: Push/pop refutation for entailment
- **Completeness**: Sound and complete for propositional logic

### Performance
- **Computer setup time**: ~2 seconds
- **Agent runtime per episode**: ~0.5 seconds  
- **KB reasoning time**: Dominated by Z3 satisfiability checks
- **Scalability**: Currently limited to small grids (propositional encoding)

---

## How It Works

1. **Knowledge Base Construction**
   - Encodes warehouse physics as Z3 biconditionals
   - Starting square guaranteed safe
   - Ready for agent observations

2. **Perception ↔ Reasoning Loop**
   - Agent observes creaking and/or rumbling
   - Tells solver about observations (TELL)
   - Queries solver for safety status (ASK)
   - Updates known safe/dangerous sets

3. **Decision Making**
   - If beacon detected: GRAB
   - If has package: Navigate home (via safe squares)
   - Else: Explore nearest safe unvisited square
   - Last resort: Return home and EXIT

4. **Path Planning**
   - BFS only through known-safe squares
   - Guarantees safe navigation
   - Finds shortest path to goal

5. **Episode Completion**
   - Execute planned sequence of actions
   - Update position, orientation, inventory
   - Track cumulative reward
   - Stop on EXIT or death

---

## Assumptions & Limitations

### Assumptions Made
- Closed-world assumption: Unknown = possibly unsafe
- Propositional logic is sufficient for current domain size
- BFS shortest path is appropriate for safety-first exploration

### Limitations
1. **Conservative behavior**: Won't explore unproven territories
2. **Propositional scaling**: O(w × h) variables becomes impractical for large grids
3. **No probability**: Can't express partial/uncertain knowledge
4. **No learning**: Knowledge doesn't transfer between episodes

### Improvements Needed
- Probabilistic reasoning (Bayesian networks)
- First-order logic for scalability (Section 3.7)
- Risk-aware exploration with heuristics (A* search)
- Machine learning for policy improvement

---

## Requirements

- **Python**: 3.13+
- **Z3**: `uv pip install z3-solver` (v4.15.8.0 tested)
- **System**: Windows/macOS/Linux
- **Dependencies**: Only Z3 (all other libraries are stdlib)

---

## References

- **AIMA Section 3.6**: "Building a Knowledge-Based Agent"
- **AIMA Section 3.2**: "The Hazardous Warehouse Environment"
- **AIMA Section 3.3**: "Propositional Logic"
- **Z3 Documentation**: [https://github.com/Z3Prover/z3](https://github.com/Z3Prover/z3)
- **De Moura & Bjørler (2008)**: Z3 SMT Solver paper

---

## Bonus Feature (Not Implemented)

The assignment included an optional bonus: extend the agent to use the emergency shutdown device. This would require:
1. Additional Z3 variables for shutdown state
2. Line-of-sight logic to detect forklift
3. Decision logic to fire device when beneficial
4. Expected improvement: Enable elimination of forklift hazards for safer navigation

---

**Project Status**: ✅ **COMPLETE & SUBMISSION READY**

All deliverables are fully implemented, tested, and documented.
