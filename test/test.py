from typing import List, Tuple, Set, Optional, Dict
import itertools
from ortools.sat.python import cp_model
from z3 import *
import time

class EfficientConstraintSorter:
    def __init__(self, elements: List[str]):
        self.elements = elements
        self.n = len(elements)
        self.element_to_idx = {name: i for i, name in enumerate(elements)}
        # Store original constraints with a unique ID for conflict reporting
        self.constraints = []
        self.next_constraint_id = 0
        self.maximize_distance: List[Tuple[str, str]] = []
        self.last_violations: Optional[List[str]] = None

    def _add_constraint(self, type: str, data: dict):
        """Internal method to add a constraint with a unique ID."""
        self.constraints.append({'id': self.next_constraint_id, 'type': type, 'data': data})
        self.next_constraint_id += 1

    def add_forbidden_constraint(self, x: str, y: str, intervals: List[Tuple[float, float]]):
        """Add a constraint that element x cannot be placed in specified intervals around element y."""
        data = {'x': x, 'y': y, 'intervals': [(float(s), float(e)) for s, e in intervals]}
        self._add_constraint('forbidden', data)

    def add_forbidden_constraint_any_y(self, x: str, y_list: List[str], intervals: List[Tuple[float, float]]):
        """Adds a constraint that x's relative position to AT LEAST ONE element in y_list
        must fall OUTSIDE the specified forbidden intervals."""
        data = {'x': x, 'y_list': y_list, 'intervals': [(float(s), float(e)) for s, e in intervals]}
        self._add_constraint('disjunctive', data)

    def add_maximize_distance_constraint(self, x: str, y: str):
        self.maximize_distance.append((x, y))

    def add_group_maximize(self, index_set: Set[int]):
        names = [self.elements[i] for i in index_set if i < len(self.elements)]
        for u, v in itertools.combinations(names, 2):
            self.add_maximize_distance_constraint(u, v)
            
    def _get_constraint_description(self, constraint: dict) -> str:
        """Returns a human-readable string for a constraint."""
        c_type = constraint['type']
        c_data = constraint['data']
        if c_type == 'forbidden':
            return f"ID {constraint['id']}: FORBIDDEN pos({c_data['x']}) - pos({c_data['y']}) in {c_data['intervals']}"
        if c_type == 'disjunctive':
            return f"ID {constraint['id']}: REQUIRED pos({c_data['x']}) - pos(y) is valid for at least one y in {c_data['y_list']}"
        return f"ID {constraint['id']}: Unknown constraint type"


    ## OR-Tools Solver with Conflict Detection
    
    def solve_with_ortools(self, time_limit_seconds: int = 300) -> Optional[List[str]]:
        """Solve using Google OR-Tools CP-SAT solver."""
        model = cp_model.CpModel()
        position = {elem: model.NewIntVar(0, self.n - 1, f'pos_{elem}') for elem in self.elements}
        model.AddAllDifferent([position[elem] for elem in self.elements])

        # Add all constraints to the model
        for constraint in self.constraints:
            self._add_single_ortools_constraint(model, position, constraint)

        # First, try to solve the model as is
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        
        # Handle optimization if required
        if self.maximize_distance:
            result = self._solve_lexicographic_ortools(model, position, time_limit_seconds)
            if result is None and self.last_violations:
                print("Lexicographic optimization failed. Error details:")
                for i, violation in enumerate(self.last_violations, 1):
                    print(f"  {i}. {violation}")
            return result

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [''] * self.n
            for elem in self.elements:
                pos = solver.Value(position[elem])
                solution[pos] = elem
            return solution
        elif status == cp_model.INFEASIBLE:
            print("Model is infeasible. Finding conflicting constraints...")
            self.last_violations = self._find_conflicting_constraints_ortools()
            return None
        else:
            self.last_violations = [f"OR-Tools solver status: {solver.StatusName(status)}"]
            return None

    def _find_conflicting_constraints_ortools(self) -> List[str]:
        """Uses assumptions to find a sufficient set of conflicting constraints (IIS)."""
        model = cp_model.CpModel()
        position = {elem: model.NewIntVar(0, self.n - 1, f'pos_{elem}') for elem in self.elements}
        model.AddAllDifferent([position[elem] for elem in self.elements])

        assumptions = []
        # *** FIX: This dictionary is crucial for mapping solver indices back to our constraints. ***
        assumption_index_to_constraint = {} 

        for constraint in self.constraints:
            assumption_lit = model.NewBoolVar(f"assumption_{constraint['id']}")
            assumptions.append(assumption_lit)
            
            # Populate the map: solver's internal index -> our constraint object
            assumption_index_to_constraint[assumption_lit.Index()] = constraint
            
            self._add_single_ortools_constraint(model, position, constraint, assumption_lit)

        model.AddAssumptions(assumptions)
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.INFEASIBLE:
            infeasible_assumption_indices = solver.SufficientAssumptionsForInfeasibility()
            conflicting_constraints = []
            
            for index in infeasible_assumption_indices:
                # *** FIX: Use the dictionary to look up the constraint by its internal solver index. ***
                # This correctly avoids the IndexError.
                original_constraint = assumption_index_to_constraint[index]
                conflicting_constraints.append(self._get_constraint_description(original_constraint))
            
            return conflicting_constraints if conflicting_constraints else ["Conflict found, but could not map to specific constraints."]
        
        return ["Solver found the model feasible during conflict analysis, which is unexpected."]

    def _add_single_ortools_constraint(self, model, position, constraint_data, enforce_if: Optional[cp_model.BoolVarT] = None):
        """Helper to add one constraint to an OR-Tools model, potentially guarded by a literal."""
        c_type = constraint_data['type']
        data = constraint_data['data']
        
        # This function encapsulates the logic you already had for adding constraints.
        # The key change is adding `.OnlyEnforceIf(enforce_if)` to each top-level `model.Add`.
        # If `enforce_if` is None, the constraint is added normally.
        
        def add(constraint):
            if enforce_if is not None:
                constraint.OnlyEnforceIf(enforce_if)
            return constraint

        if c_type == 'forbidden':
            x, y, intervals = data['x'], data['y'], data['intervals']
            if x not in self.elements or y not in self.elements: return
            
            # The entire forbidden constraint for (x, y) is one logical unit.
            # We need to enforce that for this pair, ALL interval restrictions apply.
            all_conditions_for_pair = []
            for start, end in intervals:
                diff = position[x] - position[y]
                # Create a boolean that is true if this specific interval is NOT violated
                is_valid_for_interval = model.NewBoolVar('')
                
                # Logic to define when is_valid_for_interval is true
                if start == float('-inf') and end == float('inf'):
                    model.Add(is_valid_for_interval == 0) # always invalid
                elif start == float('-inf'):
                    model.Add(diff > int(end)).OnlyEnforceIf(is_valid_for_interval)
                    model.Add(diff <= int(end)).OnlyEnforceIf(is_valid_for_interval.Not())
                elif end == float('inf'):
                    model.Add(diff < int(start)).OnlyEnforceIf(is_valid_for_interval)
                    model.Add(diff >= int(start)).OnlyEnforceIf(is_valid_for_interval.Not())
                else:
                    # diff < start OR diff > end
                    is_less = model.NewBoolVar('')
                    is_greater = model.NewBoolVar('')
                    model.AddBoolOr([is_less, is_greater]).OnlyEnforceIf(is_valid_for_interval)
                    model.AddBoolAnd([is_less.Not(), is_greater.Not()]).OnlyEnforceIf(is_valid_for_interval.Not())
                    model.Add(diff < int(start)).OnlyEnforceIf(is_less)
                    model.Add(diff > int(end)).OnlyEnforceIf(is_greater)

                all_conditions_for_pair.append(is_valid_for_interval)

            # Enforce that all these conditions must be met
            if all_conditions_for_pair:
                add(model.AddBoolAnd(all_conditions_for_pair))

        elif c_type == 'disjunctive':
            x, y_list, intervals = data['x'], data['y_list'], data['intervals']
            if x not in self.elements: return

            valid_y_list = [y for y in y_list if y in self.elements]
            if not valid_y_list: return

            # We need at least one `y` in `y_list` to satisfy the condition.
            # `is_satisfied_for_y` is true if `pos(x)-pos(y)` is OUTSIDE all forbidden intervals.
            satisfied_options = []
            for y in valid_y_list:
                is_satisfied_for_y = model.NewBoolVar(f'satisfied_{x}_{y}')
                
                violations = [] # A list of booleans, each is true if pos(x)-pos(y) is IN a forbidden interval
                for start, end in intervals:
                    diff = position[x] - position[y]
                    is_in_interval = model.NewBoolVar('')
                    if start == float('-inf') and end == float('inf'):
                        model.Add(is_in_interval == 1)
                    elif start == float('-inf'):
                        model.Add(diff <= int(end)).OnlyEnforceIf(is_in_interval)
                        model.Add(diff > int(end)).OnlyEnforceIf(is_in_interval.Not())
                    elif end == float('inf'):
                        model.Add(diff >= int(start)).OnlyEnforceIf(is_in_interval)
                        model.Add(diff < int(start)).OnlyEnforceIf(is_in_interval.Not())
                    else:
                        model.AddLinearExpressionInDomain(diff, cp_model.Domain(int(start), int(end))).OnlyEnforceIf(is_in_interval)
                        model.AddLinearExpressionNotInDomain(diff, cp_model.Domain(int(start), int(end))).OnlyEnforceIf(is_in_interval.Not())
                    violations.append(is_in_interval)
                
                # `is_satisfied_for_y` is true IFF there are NO violations.
                if violations:
                    model.AddBoolAnd([v.Not() for v in violations]).OnlyEnforceIf(is_satisfied_for_y)
                    model.AddBoolOr(violations).OnlyEnforceIf(is_satisfied_for_y.Not())
                else:
                    model.Add(is_satisfied_for_y == 1) # No intervals to violate

                satisfied_options.append(is_satisfied_for_y)
            
            if satisfied_options:
                add(model.AddBoolOr(satisfied_options))

    def _solve_lexicographic_ortools(self, base_model, position, time_limit_seconds):
        """Solve with lexicographic optimization: maximize distances in sorted order."""
        print("Starting lexicographic optimization with OR-Tools...")

        # Collect valid pairs
        valid_pairs = [(x, y) for x, y in self.maximize_distance if x in self.elements and y in self.elements]
        k = len(valid_pairs)

        if not valid_pairs:
            # No valid pairs, solve base model
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = time_limit_seconds
            status = solver.Solve(base_model)
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                solution = [''] * self.n
                for elem in self.elements:
                    pos = solver.Value(position[elem])
                    solution[pos] = elem
                return solution
            elif status == cp_model.INFEASIBLE:
                print("Base model is infeasible during lexicographic optimization. Finding conflicting constraints...")
                self.last_violations = self._find_conflicting_constraints_ortools()
                return None
            else:
                self.last_violations = [f"OR-Tools solver status: {solver.StatusName(status)}"]
                return None

        # First, check if the base constraints are feasible
        print("Checking feasibility of base constraints...")
        test_model = cp_model.CpModel()
        test_position = {elem: test_model.NewIntVar(0, self.n - 1, f'pos_{elem}') for elem in self.elements}
        test_model.AddAllDifferent([test_position[elem] for elem in self.elements])
        self._add_constraints_to_model(test_model, test_position)
        
        test_solver = cp_model.CpSolver()
        test_solver.parameters.max_time_in_seconds = min(30, time_limit_seconds / 10)  # Quick feasibility check
        test_status = test_solver.Solve(test_model)
        
        if test_status == cp_model.INFEASIBLE:
            print("Base constraints are infeasible. Finding conflicting constraints...")
            self.last_violations = self._find_conflicting_constraints_ortools()
            return None
        elif test_status != cp_model.OPTIMAL and test_status != cp_model.FEASIBLE:
            self.last_violations = [f"Base constraints feasibility check failed: {test_solver.StatusName(test_status)}"]
            return None
        
        print("Base constraints are feasible. Proceeding with optimization...")

        # Create the model
        model = cp_model.CpModel()
        model_position = {elem: model.NewIntVar(0, self.n - 1, f'pos_{elem}') for elem in self.elements}
        model.AddAllDifferent([model_position[elem] for elem in self.elements])
        self._add_constraints_to_model(model, model_position)

        # Define distance variables
        dist = {}
        for i, (x, y) in enumerate(valid_pairs):
            dist[i] = model.NewIntVar(0, self.n - 1, f'dist_{x}_{y}')
            model.AddAbsEquality(dist[i], model_position[x] - model_position[y])

        # Define sorted distance variables
        sorted_dist = [model.NewIntVar(0, self.n - 1, f'sorted_dist_{j}') for j in range(k)]
        for j in range(k - 1):
            model.Add(sorted_dist[j] <= sorted_dist[j + 1])

        # Boolean variables to map dist[i] to sorted_dist[j]
        b = [[model.NewBoolVar(f'b_{i}_{j}') for j in range(k)] for i in range(k)]
        for i in range(k):
            model.Add(sum(b[i][j] for j in range(k)) == 1)  # Each dist[i] maps to one sorted_dist[j]
        for j in range(k):
            model.Add(sum(b[i][j] for i in range(k)) == 1)  # Each sorted_dist[j] maps to one dist[i]
        for i in range(k):
            for j in range(k):
                model.Add(sorted_dist[j] == dist[i]).OnlyEnforceIf(b[i][j])

        # Sequential maximization
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds / max(1, k)  # Distribute time
        optimal_values = []

        for j in range(k):
            print(f"Maximizing sorted distance {j + 1} of {k}...")
            model.Maximize(sorted_dist[j])
            status = solver.Solve(model)
            if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
                if status == cp_model.INFEASIBLE:
                    print(f"Model became infeasible while maximizing sorted_dist[{j}]. This suggests the optimization constraints conflict with base constraints.")
                    # Try to find conflicts in the optimization model
                    print("Attempting to find conflicting constraints in optimization model...")
                    self.last_violations = [f"Optimization became infeasible at step {j+1}/{k}. The lexicographic optimization constraints may be too restrictive."]
                    # Additional diagnostic info
                    if j > 0:
                        self.last_violations.append(f"Successfully optimized {j} distance(s) with values: {optimal_values}")
                        self.last_violations.append("This suggests the conflict arises from the combination of:")
                        self.last_violations.append("1. Base positioning constraints")
                        self.last_violations.append("2. Distance maximization requirements")
                        self.last_violations.append(f"3. Previously fixed distance constraints: sorted_dist[0..{j-1}] >= {optimal_values}")
                else:
                    self.last_violations = [f"Failed to maximize sorted_dist[{j}]: {solver.StatusName(status)}"]
                return None
            s_j = solver.Value(sorted_dist[j])
            optimal_values.append(s_j)
            model.Add(sorted_dist[j] >= s_j)  # Fix this distance for subsequent steps
            print(f"  sorted_dist[{j}] maximized to {s_j}")

        # Extract final solution
        solution = [''] * self.n
        for elem in self.elements:
            pos = solver.Value(model_position[elem])
            solution[pos] = elem

        # Report final distances
        actual_distances = [abs(solution.index(x) - solution.index(y)) for x, y in valid_pairs]
        print(f"Final distances: {sorted(actual_distances)}")
        return solution

    def _add_constraints_to_model(self, model, position):
        """Helper method to add all original constraints to a model."""
        # Add forbidden constraints
        for constraint in self.constraints:
            c_type = constraint['type']
            data = constraint['data']
            
            if c_type == 'forbidden':
                x, y, intervals = data['x'], data['y'], data['intervals']
                if x not in self.elements or y not in self.elements:
                    continue
                
                for start, end in intervals:
                    if start == float('-inf') and end == float('inf'):
                        continue
                    elif start == float('-inf'):
                        if end == float('inf'):
                            continue
                        end_int = int(end)
                        model.Add(position[x] - position[y] > end_int)
                    elif end == float('inf'):
                        if start == float('-inf'):
                            continue
                        start_int = int(start)
                        model.Add(position[x] - position[y] < start_int)
                    else:
                        start_int, end_int = int(start), int(end)
                        is_less = model.NewBoolVar(f'less_{x}_{y}_{start_int}_{end_int}')
                        is_greater = model.NewBoolVar(f'greater_{x}_{y}_{start_int}_{end_int}')
                        model.AddBoolOr([is_less, is_greater])
                        model.Add(position[x] - position[y] < start_int).OnlyEnforceIf(is_less)
                        model.Add(position[x] - position[y] > end_int).OnlyEnforceIf(is_greater)
            
            elif c_type == 'disjunctive':
                x, y_list, intervals = data['x'], data['y_list'], data['intervals']
                if x not in self.elements:
                    continue
                
                valid_y_list = [y for y in y_list if y in self.elements]
                if not valid_y_list:
                    continue
                
                satisfied_with_y = []
                
                for y in valid_y_list:
                    y_satisfied = model.NewBoolVar(f'satisfied_{x}_{y}')
                    satisfied_with_y.append(y_satisfied)
                    
                    interval_violations = []
                    
                    for start, end in intervals:
                        if start == float('-inf') and end == float('inf'):
                            always_violated = model.NewBoolVar(f'always_violated_{x}_{y}')
                            model.Add(always_violated == 1)
                            interval_violations.append(always_violated)
                        elif start == float('-inf'):
                            if end == float('inf'):
                                continue
                            end_int = int(end)
                            in_interval = model.NewBoolVar(f'in_neg_inf_interval_{x}_{y}_{end_int}')
                            interval_violations.append(in_interval)
                            model.Add(position[x] - position[y] <= end_int).OnlyEnforceIf(in_interval)
                            model.Add(position[x] - position[y] > end_int).OnlyEnforceIf(in_interval.Not())
                        elif end == float('inf'):
                            if start == float('-inf'):
                                continue
                            start_int = int(start)
                            in_interval = model.NewBoolVar(f'in_pos_inf_interval_{x}_{y}_{start_int}')
                            interval_violations.append(in_interval)
                            model.Add(position[x] - position[y] >= start_int).OnlyEnforceIf(in_interval)
                            model.Add(position[x] - position[y] < start_int).OnlyEnforceIf(in_interval.Not())
                        else:
                            start_int, end_int = int(start), int(end)
                            in_interval = model.NewBoolVar(f'in_interval_{x}_{y}_{start_int}_{end_int}')
                            interval_violations.append(in_interval)
                            model.Add(position[x] - position[y] >= start_int).OnlyEnforceIf(in_interval)
                            model.Add(position[x] - position[y] <= end_int).OnlyEnforceIf(in_interval)
                            model.Add(position[x] - position[y] < start_int).OnlyEnforceIf(in_interval.Not())
                            model.Add(position[x] - position[y] > end_int).OnlyEnforceIf(in_interval.Not())
                    
                    if interval_violations:
                        model.AddBoolAnd([iv.Not() for iv in interval_violations]).OnlyEnforceIf(y_satisfied)
                        model.AddBoolOr(interval_violations).OnlyEnforceIf(y_satisfied.Not())
                    else:
                        model.Add(y_satisfied == 1)
                
                if satisfied_with_y:
                    model.AddBoolOr(satisfied_with_y)

    ## Z3 Solver with Conflict Detection
    
    def solve_with_z3(self, time_limit_ms: int = 300000) -> Optional[List[str]]:
        """Solve using Z3 with unsat core for conflict detection."""
        s = Solver()
        s.set("timeout", time_limit_ms)

        pos = {elem: Int(f'pos_{elem}') for elem in self.elements}
        
        # Basic permutation constraints
        for elem in self.elements:
            s.add(And(pos[elem] >= 0, pos[elem] < self.n))
        s.add(Distinct([pos[elem] for elem in self.elements]))

        # Add constraints and track them for unsat core
        for constraint in self.constraints:
            prop = Bool(self._get_constraint_description(constraint)) # Use description as tracking literal
            c_type, data = constraint['type'], constraint['data']
            
            if c_type == 'forbidden':
                x, y, intervals = data['x'], data['y'], data['intervals']
                if x in pos and y in pos:
                    conditions = []
                    for start, end in intervals:
                        diff = pos[x] - pos[y]
                        if start == float('-inf'):
                            conditions.append(diff > int(end))
                        elif end == float('inf'):
                            conditions.append(diff < int(start))
                        else:
                            conditions.append(Or(diff < int(start), diff > int(end)))
                    s.assert_and_track(And(conditions), prop)

            elif c_type == 'disjunctive':
                x, y_list, intervals = data['x'], data['y_list'], data['intervals']
                if x in pos:
                    valid_y_list = [y for y in y_list if y in pos]
                    if valid_y_list:
                        or_conditions = []
                        for y in valid_y_list:
                            diff = pos[x] - pos[y]
                            and_conditions = []
                            for start, end in intervals:
                                if start == float('-inf'):
                                    and_conditions.append(diff > int(end))
                                elif end == float('inf'):
                                    and_conditions.append(diff < int(start))
                                else:
                                    and_conditions.append(Or(diff < int(start), diff > int(end)))
                            or_conditions.append(And(and_conditions))
                        s.assert_and_track(Or(or_conditions), prop)

        # Optimization goal (if any)
        # Z3's standard solver doesn't optimize. For that, you'd use z3.Optimize()
        # and add soft constraints. For finding hard conflicts, z3.Solver is perfect.

        if s.check() == sat:
            m = s.model()
            solution = [''] * self.n
            for elem in self.elements:
                p = m.eval(pos[elem]).as_long()
                solution[p] = elem
            return solution
        else:
            print("Model is unsat. Finding conflicting constraints (unsat core)...")
            core = s.unsat_core()
            self.last_violations = [str(c) for c in core]
            return None

    def solve(self, method: str = "ortools", time_limit: int = 300) -> Optional[List[str]]:
        self.last_violations = []
        if method.lower() == "ortools":
            return self.solve_with_ortools(time_limit)
        elif method.lower() == "z3":
            return self.solve_with_z3(time_limit * 1000)
        else:
            raise ValueError("Method must be 'ortools' or 'z3'")

# --- Example Usage ---

def test_conflict():
    """Test conflict detection with a clear contradiction."""
    print("--- Testing Conflict Detection ---")
    elements = ["A", "B", "C"]
    sorter = EfficientConstraintSorter(elements)

    # Contradictory constraints: A must be after B, and B must be after A.
    # pos(A) - pos(B) > 0  AND  pos(B) - pos(A) > 0
    sorter.add_forbidden_constraint("A", "B", [(-float('inf'), 0)]) # pos(A) - pos(B) must be > 0
    sorter.add_forbidden_constraint("B", "A", [(-float('inf'), 0)]) # pos(B) - pos(A) must be > 0
    
    # A third, non-conflicting constraint to show it's not part of the IIS
    sorter.add_forbidden_constraint("C", "A", [(1, 1)]) # C cannot be immediately after A

    print("\n[OR-Tools] Attempting to solve with contradictory constraints...")
    result_ortools = sorter.solve(method="ortools", time_limit=10)
    
    if not result_ortools:
        print("\n[OR-Tools] Solution not found. Identified conflicting constraints:")
        for v in sorter.last_violations:
            print(f"  - {v}")
    
    print("\n" + "="*40 + "\n")

    print("[Z3] Attempting to solve with contradictory constraints...")
    result_z3 = sorter.solve(method="z3", time_limit=10)

    if not result_z3:
        print("\n[Z3] Solution not found. Identified conflicting constraints (unsat core):")
        for v in sorter.last_violations:
            print(f"  - {v}")

def test_solvers():
    """Test both solvers with a sample problem."""
    elements = [f"elem_{i}" for i in range(20)]  # Start with smaller test
    
    sorter = EfficientConstraintSorter(elements)
    
    # Add some test constraints
    sorter.add_forbidden_constraint("elem_0", "elem_3", [(1, float('inf'))])
    sorter.add_forbidden_constraint("elem_2", "elem_3", [(1, float('inf'))])
    sorter.add_forbidden_constraint("elem_2", "elem_3", [(-float('inf'), -3)])
    sorter.add_forbidden_constraint("elem_5", "elem_6", [(-float('inf'), -2)])
    sorter.add_forbidden_constraint("elem_5", "elem_6", [(1, float('inf'))])
    sorter.add_group_maximize({0, 2, 3, 5, 6})
    
    print("Testing OR-Tools solver...")
    start_time = time.time()
    result_ortools = sorter.solve(method="ortools", time_limit=60)
    ortools_time = time.time() - start_time
    
    print("Testing Z3 solver...")
    start_time = time.time()
    result_z3 = sorter.solve(method="z3", time_limit=60)
    z3_time = time.time() - start_time
    
    print(f"OR-Tools result: {'Found' if result_ortools else 'Not found'} (Time: {ortools_time:.2f}s)")
    print(f"Z3 result: {'Found' if result_z3 else 'Not found'} (Time: {z3_time:.2f}s)")
    
    return result_ortools, result_z3

if __name__ == "__main__":
    test_conflict()
    print("\n" + "="*60 + "\n")
    test_solvers()