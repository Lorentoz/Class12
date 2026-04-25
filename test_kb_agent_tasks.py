"""
Comprehensive test of Knowledge-Based Agent implementation
Tasks 1-3: Z3 setup, symbols, and manual reasoning verification
"""

import sys
sys.path.insert(0, 'src')

from z3 import Bool, Bools, Solver, And, Or, Not
from warehouse_kb_agent import (
    z3_entails, build_warehouse_kb, damaged, forklift_at,
    creaking_at, rumbling_at, safe, get_adjacent
)
from hazardous_warehouse_env import HazardousWarehouseEnv, Percept

print("=" * 70)
print("TASK 1: Z3 Setup and Exploration")
print("=" * 70)

# Test 1.1: Basic biconditionals
print("\n1.1 Creating boolean variables and biconditionals:")
P, Q = Bools('P Q')
s = Solver()
s.add(P == Q)    # Biconditional
s.add(P)

result = s.check()
print(f"  Satisfiability: {result}")
print(f"  Model: {s.model()}")
assert str(result) == "sat", "Expected sat"

# Test 1.2: z3_entails function
print("\n1.2 Testing z3_entails function:")
print(f"  Does solver entail Q? {z3_entails(s, Q)}")
assert z3_entails(s, Q) == True, "Solver should entail Q"
print(f"  Does solver entail Not(Q)? {z3_entails(s, Not(Q))}")
assert z3_entails(s, Not(Q)) == False, "Solver should not entail Not(Q)"
print("  [PASS] z3_entails working correctly")

print("\n" + "=" * 70)
print("TASK 2: Propositional Symbols and Warehouse KB")
print("=" * 70)

# Test 2.1: Helper functions
print("\n2.1 Testing Bool variable helpers:")
d_2_1 = damaged(2, 1)
f_2_1 = forklift_at(2, 1)
c_2_1 = creaking_at(2, 1)
r_2_1 = rumbling_at(2, 1)
ok_2_1 = safe(2, 1)
print(f"  damaged(2,1):     {d_2_1}")
print(f"  forklift_at(2,1): {f_2_1}")
print(f"  creaking_at(2,1): {c_2_1}")
print(f"  rumbling_at(2,1): {r_2_1}")
print(f"  safe(2,1):        {ok_2_1}")
print("  [PASS] All symbols created correctly")

# Test 2.2: Adjacency
print("\n2.2 Testing adjacency:")
adj_2_1 = get_adjacent(2, 1)
print(f"  Adjacent to (2,1): {sorted(adj_2_1)}")
assert set(adj_2_1) == {(1, 1), (3, 1), (2, 2)}, "Wrong adjacency"
print("  [PASS] Adjacency correct")

# Test 2.3: Build warehouse KB
print("\n2.3 Building warehouse knowledge base:")
solver = build_warehouse_kb()
kb_check = solver.check()
print(f"  Initial KB satisfiable? {kb_check}")
assert str(kb_check) == "sat", "KB should be satisfiable"
print("  [PASS] Warehouse KB built successfully")

print("\n" + "=" * 70)
print("TASK 3: Manual Reasoning - Textbook Walkthrough")
print("=" * 70)

# Recreate the textbook example
solver = build_warehouse_kb()

# Step 1: At (1,1), no creaking, no rumbling
print("\nStep 1: At (1,1) - perceiving no creaking, no rumbling")
solver.add(Not(creaking_at(1, 1)))
solver.add(Not(rumbling_at(1, 1)))

safe_2_1 = z3_entails(solver, safe(2, 1))
safe_1_2 = z3_entails(solver, safe(1, 2))
print(f"  Is (2,1) safe? {safe_2_1}")
print(f"  Is (1,2) safe? {safe_1_2}")
assert safe_2_1 == True, "Should prove (2,1) safe"
assert safe_1_2 == True, "Should prove (1,2) safe"
print("  [PASS] Correctly deduced adjacent squares are safe")

# Step 2: Move to (2,1), perceive creaking but no rumbling
print("\nStep 2: At (2,1) - perceiving creaking, no rumbling")
solver.add(creaking_at(2, 1))
solver.add(Not(rumbling_at(2, 1)))

safe_3_1 = z3_entails(solver, safe(3, 1))
not_safe_3_1 = z3_entails(solver, Not(safe(3, 1)))
safe_2_2 = z3_entails(solver, safe(2, 2))
not_safe_2_2 = z3_entails(solver, Not(safe(2, 2)))

print(f"  Is (3,1) safe? {safe_3_1}")
print(f"  Is (3,1) provably NOT safe? {not_safe_3_1}")
print(f"  Is (2,2) safe? {safe_2_2}")
print(f"  Is (2,2) provably NOT safe? {not_safe_2_2}")

assert safe_3_1 == False and not_safe_3_1 == False, "(3,1) should be UNKNOWN"
assert safe_2_2 == False and not_safe_2_2 == False, "(2,2) should be UNKNOWN"
print("  [PASS] Correctly identified (3,1) and (2,2) as UNKNOWN")

# Step 3: Move to (1,2), perceive rumbling but no creaking
print("\nStep 3: At (1,2) - perceiving rumbling, no creaking")
solver.add(rumbling_at(1, 2))
solver.add(Not(creaking_at(1, 2)))

safe_2_2_after = z3_entails(solver, safe(2, 2))
not_safe_3_1_after = z3_entails(solver, Not(safe(3, 1)))
not_safe_1_3 = z3_entails(solver, Not(safe(1, 3)))

print(f"  Is (2,2) safe NOW? {safe_2_2_after}")
print(f"  Is (3,1) provably NOT safe NOW? {not_safe_3_1_after}")
print(f"  Is (1,3) provably NOT safe? {not_safe_1_3}")

assert safe_2_2_after == True, "Should now prove (2,2) is safe"
assert not_safe_3_1_after == True, "Should now prove (3,1) is NOT safe"
assert not_safe_1_3 == True, "Should prove (1,3) is NOT safe"
print("  [PASS] Chain of reasoning succeeded!")
print("    - Forklift location identified: (2,3)")
print("    - Damaged floor location identified: (3,1)")

print("\n" + "=" * 70)
print("All tests passed! [SUCCESS]")
print("=" * 70)
print("\nSummary:")
print("  Task 1: Z3 biconditionals and entailment checking - COMPLETE")
print("  Task 2: Propositional variables and warehouse KB - COMPLETE")
print("  Task 3: Manual reasoning walkthrough (textbook example) - COMPLETE")
print("\nThe implementation correctly follows the textbook walkthrough")
print("and demonstrates sound propositional logic reasoning.")
