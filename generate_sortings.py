import re
import random
from typing import List, Tuple, Set, Optional, Dict
import itertools
import string
from ortools.sat.python import cp_model
from z3 import *
from config import PLAYLISTS_PATH

ROLES = ['path', 'names', 'attributes', 'dependencies', 'sprawl']
PATTERN_DISTANCE = r'^(?P<prefix>as far as possible from )(?P<any>any)?((?P<number>\d+)|(?P<name>.+))(?P<suffix>)$'
PATTERN_AREAS = r'^(?P<prefix>.*\|)(?P<any>any)?((?P<number>\d+)|(?P<name>.+))(?P<suffix>\|.*)$'

class instr_struct:
    def __init__(self, instr_type: int, any: bool, numbers: List[int], intervals: List[Tuple[int, int]] = None):
        self.instr_type = instr_type
        self.any = any
        self.numbers = numbers
        self.intervals = intervals

    def __eq__(self, other):
        if not isinstance(other, instr_struct):
            return False
        return (self.instr_type == other.instr_type and
                self.any == other.any and
                sorted(self.numbers) == sorted(other.numbers) and
                sorted(self.intervals) == sorted(other.intervals))

    def __hash__(self):
        return hash((
            self.instr_type,
            self.any,
            tuple(sorted(self.numbers)),
            tuple(sorted(self.intervals))
        ))

class EfficientConstraintSorter:
    def __init__(self, elements: List[str]):
        self.elements = elements
        self.n = len(elements)
        self.forbidden_constraints: List[Tuple[str, str, List[Tuple[float, float]]]] = []
        self.required_disjunctive_constraints: List[Tuple[str, List[str], List[Tuple[float, float]]]] = []
        self.maximize_distance: List[Tuple[str, str]] = []
        self.last_violations: Optional[List[str]] = None
        
    def add_forbidden_constraint(self, x: str, y: str, intervals: List[Tuple[int, int]]):
        """Add a constraint that element x cannot be placed in specified intervals around element y."""
        self.forbidden_constraints.append((x, y, [(float(s), float(e)) for s, e in intervals]))
    
    def add_forbidden_constraint_any_y(self, x: str, y_list: List[str], intervals: List[Tuple[int, int]]):
        """Adds a constraint that x's relative position to AT LEAST ONE element in y_list
        must fall OUTSIDE the specified forbidden intervals."""
        self.required_disjunctive_constraints.append((x, y_list, [(float(s), float(e)) for s, e in intervals]))
    
    def add_maximize_distance_constraint(self, x: str, y: str):
        self.maximize_distance.append((x, y))
    
    def add_group_maximize(self, index_set: Set[int]):
        names = [self.elements[i] for i in index_set]
        for u, v in itertools.combinations(names, 2):
            self.add_maximize_distance_constraint(u, v)
    
    def solve_with_ortools(self, time_limit_seconds: int = 300) -> Optional[List[str]]:
        """Solve using Google OR-Tools CP-SAT solver with lexicographic optimization."""
        model = cp_model.CpModel()
        
        # Decision variables: position[i] represents the position of element i
        position = {}
        for i, elem in enumerate(self.elements):
            position[elem] = model.NewIntVar(0, self.n - 1, f'pos_{elem}')
        
        # All elements must be at different positions (permutation constraint)
        model.AddAllDifferent([position[elem] for elem in self.elements])
        
        # Add forbidden constraints
        for x, y, intervals in self.forbidden_constraints:
            if x not in self.elements or y not in self.elements:
                continue
            
            # For each forbidden interval, create constraint: NOT (start <= pos[x] - pos[y] <= end)
            for start, end in intervals:
                # Handle infinite intervals
                if start == float('-inf') and end == float('inf'):
                    # This interval covers everything, so the constraint is unsatisfiable
                    # We can't forbid all relative positions
                    continue
                elif start == float('-inf'):
                    # Interval is (-inf, end], so we need pos[x] - pos[y] > end
                    if end == float('inf'):
                        continue  # This would be (-inf, inf) which we already handled
                    end_int = int(end)
                    model.Add(position[x] - position[y] > end_int)
                elif end == float('inf'):
                    # Interval is [start, inf), so we need pos[x] - pos[y] < start
                    if start == float('-inf'):
                        continue  # This would be (-inf, inf) which we already handled
                    start_int = int(start)
                    model.Add(position[x] - position[y] < start_int)
                else:
                    # Finite interval [start, end]
                    start_int, end_int = int(start), int(end)
                    
                    # pos[x] - pos[y] should NOT be in [start_int, end_int]
                    # This means: pos[x] - pos[y] < start_int OR pos[x] - pos[y] > end_int
                    
                    # Create boolean variables for the disjunction
                    is_less = model.NewBoolVar(f'less_{x}_{y}_{start_int}_{end_int}')
                    is_greater = model.NewBoolVar(f'greater_{x}_{y}_{start_int}_{end_int}')
                    
                    # At least one must be true
                    model.AddBoolOr([is_less, is_greater])
                    
                    # If is_less is true, then pos[x] - pos[y] < start_int
                    model.Add(position[x] - position[y] < start_int).OnlyEnforceIf(is_less)
                    
                    # If is_greater is true, then pos[x] - pos[y] > end_int
                    model.Add(position[x] - position[y] > end_int).OnlyEnforceIf(is_greater)
        
        # Add required disjunctive constraints
        for x, y_list, intervals in self.required_disjunctive_constraints:
            if x not in self.elements:
                continue
            
            valid_y_list = [y for y in y_list if y in self.elements]
            if not valid_y_list:
                continue
            
            # For each y in y_list, create boolean indicating if constraint is satisfied with that y
            satisfied_with_y = []
            
            for y in valid_y_list:
                y_satisfied = model.NewBoolVar(f'satisfied_{x}_{y}')
                satisfied_with_y.append(y_satisfied)
                
                # y_satisfied is true if x is NOT in forbidden intervals relative to y
                interval_violations = []
                
                for start, end in intervals:
                    # Handle infinite intervals in disjunctive constraints
                    if start == float('-inf') and end == float('inf'):
                        # This interval covers everything, so it's always violated
                        # Create a boolean that's always true
                        always_violated = model.NewBoolVar(f'always_violated_{x}_{y}')
                        model.Add(always_violated == 1)
                        interval_violations.append(always_violated)
                    elif start == float('-inf'):
                        if end == float('inf'):
                            continue  # Already handled above
                        end_int = int(end)
                        # Violated if pos[x] - pos[y] <= end_int
                        in_interval = model.NewBoolVar(f'in_neg_inf_interval_{x}_{y}_{end_int}')
                        interval_violations.append(in_interval)
                        
                        model.Add(position[x] - position[y] <= end_int).OnlyEnforceIf(in_interval)
                        model.Add(position[x] - position[y] > end_int).OnlyEnforceIf(in_interval.Not())
                    elif end == float('inf'):
                        if start == float('-inf'):
                            continue  # Already handled above
                        start_int = int(start)
                        # Violated if pos[x] - pos[y] >= start_int
                        in_interval = model.NewBoolVar(f'in_pos_inf_interval_{x}_{y}_{start_int}')
                        interval_violations.append(in_interval)
                        
                        model.Add(position[x] - position[y] >= start_int).OnlyEnforceIf(in_interval)
                        model.Add(position[x] - position[y] < start_int).OnlyEnforceIf(in_interval.Not())
                    else:
                        # Finite interval
                        start_int, end_int = int(start), int(end)
                        
                        # Create boolean for whether x is in this forbidden interval relative to y
                        in_interval = model.NewBoolVar(f'in_interval_{x}_{y}_{start_int}_{end_int}')
                        interval_violations.append(in_interval)
                        
                        # in_interval is true iff start_int <= pos[x] - pos[y] <= end_int
                        model.Add(position[x] - position[y] >= start_int).OnlyEnforceIf(in_interval)
                        model.Add(position[x] - position[y] <= end_int).OnlyEnforceIf(in_interval)
                        model.Add(position[x] - position[y] < start_int).OnlyEnforceIf(in_interval.Not())
                        model.Add(position[x] - position[y] > end_int).OnlyEnforceIf(in_interval.Not())
                
                # y_satisfied is true iff none of the intervals are violated
                if interval_violations:
                    model.AddBoolAnd([iv.Not() for iv in interval_violations]).OnlyEnforceIf(y_satisfied)
                    model.AddBoolOr(interval_violations).OnlyEnforceIf(y_satisfied.Not())
                else:
                    # No intervals to violate, so always satisfied
                    model.Add(y_satisfied == 1)
            
            # At least one y must satisfy the constraint
            if satisfied_with_y:
                model.AddBoolOr(satisfied_with_y)
        
        # Lexicographic optimization: first maximize minimum distance, then maximize sum
        if self.maximize_distance:
            return self._solve_lexicographic_ortools(model, position, time_limit_seconds)
        else:
            # No distance constraints, just solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = time_limit_seconds
            
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # Extract solution
                solution = [''] * self.n
                for elem in self.elements:
                    pos = solver.Value(position[elem])
                    solution[pos] = elem
                return solution
            else:
                self.last_violations = [f"OR-Tools solver status: {solver.StatusName(status)}"]
                return None
    
    def _solve_lexicographic_ortools(self, base_model, position, time_limit_seconds):
        """Solve with lexicographic optimization: first max-min, then max-sum."""
        
        # Phase 1: Find the maximum possible minimum distance
        print("Phase 1: Finding maximum minimum distance...")
        
        # Create distance variables
        distance_vars = []
        valid_pairs = []
        for x, y in self.maximize_distance:
            if x not in self.elements or y not in self.elements:
                continue
            valid_pairs.append((x, y))
        
        if not valid_pairs:
            # No valid pairs, just solve base model
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = time_limit_seconds
            status = solver.Solve(base_model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                solution = [''] * self.n
                for elem in self.elements:
                    pos = solver.Value(position[elem])
                    solution[pos] = elem
                return solution
            else:
                self.last_violations = [f"OR-Tools solver status: {solver.StatusName(status)}"]
                return None
        
        # Binary search for maximum minimum distance
        low, high = 1, self.n - 1
        best_min_distance = 0
        best_solution = None
        
        while low <= high:
            mid = (low + high) // 2
            
            # Create model for testing minimum distance = mid
            test_model = cp_model.CpModel()
            
            # Copy all constraints from base model
            test_position = {}
            for elem in self.elements:
                test_position[elem] = test_model.NewIntVar(0, self.n - 1, f'pos_{elem}')
            
            # Add all constraints from base model (we need to reconstruct them)
            test_model.AddAllDifferent([test_position[elem] for elem in self.elements])
            
            # Re-add all original constraints
            self._add_constraints_to_model(test_model, test_position)
            
            # Add minimum distance constraint
            for x, y in valid_pairs:
                abs_distance = test_model.NewIntVar(0, self.n - 1, f'abs_dist_{x}_{y}')
                test_model.AddAbsEquality(abs_distance, test_position[x] - test_position[y])
                test_model.Add(abs_distance >= mid)
            
            # Test if this minimum distance is achievable
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = max(1, time_limit_seconds // 10)  # Use fraction of time
            
            status = solver.Solve(test_model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                best_min_distance = mid
                # Extract solution
                solution = [''] * self.n
                for elem in self.elements:
                    pos = solver.Value(test_position[elem])
                    solution[pos] = elem
                best_solution = solution
                low = mid + 1
                print(f"  Minimum distance {mid} is achievable")
            else:
                high = mid - 1
                print(f"  Minimum distance {mid} is not achievable")
        
        if best_solution is None:
            self.last_violations = ["Could not find any solution with distance constraints"]
            return None
        
        print(f"Phase 1 complete: Maximum minimum distance = {best_min_distance}")
        
        # Phase 2: Among solutions with optimal minimum distance, maximize sum
        print("Phase 2: Maximizing sum of distances...")
        
        final_model = cp_model.CpModel()
        final_position = {}
        for elem in self.elements:
            final_position[elem] = final_model.NewIntVar(0, self.n - 1, f'pos_{elem}')
        
        final_model.AddAllDifferent([final_position[elem] for elem in self.elements])
        
        # Re-add all original constraints
        self._add_constraints_to_model(final_model, final_position)
        
        # Add minimum distance constraint (must equal best_min_distance)
        distance_vars = []
        for x, y in valid_pairs:
            abs_distance = final_model.NewIntVar(0, self.n - 1, f'abs_dist_{x}_{y}')
            final_model.AddAbsEquality(abs_distance, final_position[x] - final_position[y])
            final_model.Add(abs_distance >= best_min_distance)
            distance_vars.append(abs_distance)
        
        # Maximize sum of distances
        final_model.Maximize(sum(distance_vars))
        
        # Solve final model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(1, time_limit_seconds - time_limit_seconds // 5)  # Use remaining time
        
        status = solver.Solve(final_model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [''] * self.n
            for elem in self.elements:
                pos = solver.Value(final_position[elem])
                solution[pos] = elem
            
            # Calculate and report final metrics
            min_dist = min(abs(solution.index(x) - solution.index(y)) for x, y in valid_pairs)
            sum_dist = sum(abs(solution.index(x) - solution.index(y)) for x, y in valid_pairs)
            print(f"Phase 2 complete: Minimum distance = {min_dist}, Sum of distances = {sum_dist}")
            
            return solution
        else:
            # Fallback to Phase 1 solution
            print("Phase 2 failed, returning Phase 1 solution")
            return best_solution
    
    def _add_constraints_to_model(self, model, position):
        """Helper method to add all original constraints to a model."""
        # Add forbidden constraints
        for x, y, intervals in self.forbidden_constraints:
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
        
        # Add required disjunctive constraints
        for x, y_list, intervals in self.required_disjunctive_constraints:
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

    def solve_with_z3(self, time_limit_ms: int = 300000) -> Optional[List[str]]:
        """Solve using Z3 SMT solver with lexicographic optimization."""
        # Create Z3 variables
        position = {}
        for elem in self.elements:
            position[elem] = Int(f'pos_{elem}')
        
        # Create solver
        solver = Solver()
        solver.set("timeout", time_limit_ms)
        
        # Domain constraints: all positions are in [0, n-1]
        for elem in self.elements:
            solver.add(And(position[elem] >= 0, position[elem] < self.n))
        
        # All different constraint (permutation)
        solver.add(Distinct([position[elem] for elem in self.elements]))
        
        # Add forbidden constraints
        for x, y, intervals in self.forbidden_constraints:
            if x not in self.elements or y not in self.elements:
                continue
            
            for start, end in intervals:
                if start == float('inf') or end == float('inf'):
                    continue
                
                start_int, end_int = int(start), int(end)
                
                # NOT (start_int <= pos[x] - pos[y] <= end_int)
                # Equivalent to: pos[x] - pos[y] < start_int OR pos[x] - pos[y] > end_int
                solver.add(Or(
                    position[x] - position[y] < start_int,
                    position[x] - position[y] > end_int
                ))
        
        # Add required disjunctive constraints
        for x, y_list, intervals in self.required_disjunctive_constraints:
            if x not in self.elements:
                continue
            
            valid_y_list = [y for y in y_list if y in self.elements]
            if not valid_y_list:
                continue
            
            # For each y in y_list, check if constraint is satisfied
            satisfied_conditions = []
            
            for y in valid_y_list:
                # x satisfies constraint with y if it's outside all forbidden intervals
                outside_all_intervals = []
                
                for start, end in intervals:
                    if start == float('inf') or end == float('inf'):
                        continue
                    
                    start_int, end_int = int(start), int(end)
                    
                    # Outside this interval: pos[x] - pos[y] < start_int OR pos[x] - pos[y] > end_int
                    outside_interval = Or(
                        position[x] - position[y] < start_int,
                        position[x] - position[y] > end_int
                    )
                    outside_all_intervals.append(outside_interval)
                
                if outside_all_intervals:
                    # Satisfied with y if outside ALL intervals
                    satisfied_with_y = And(outside_all_intervals)
                    satisfied_conditions.append(satisfied_with_y)
            
            # At least one y must satisfy the constraint
            if satisfied_conditions:
                solver.add(Or(satisfied_conditions))
        
        # Check satisfiability first
        if solver.check() != sat:
            self.last_violations = ["Z3 solver could not find a satisfiable solution"]
            return None
        
        # If we have distance maximization constraints, use lexicographic optimization
        if self.maximize_distance:
            return self._solve_lexicographic_z3(position, solver, time_limit_ms)
        else:
            # No distance constraints, return current solution
            model = solver.model()
            solution = [''] * self.n
            for elem in self.elements:
                pos = model[position[elem]].as_long()
                solution[pos] = elem
            return solution
    
    def _solve_lexicographic_z3(self, position, base_solver, time_limit_ms):
        """Solve with lexicographic optimization using Z3."""
        valid_pairs = [(x, y) for x, y in self.maximize_distance 
                      if x in self.elements and y in self.elements]
        
        if not valid_pairs:
            # No valid pairs, return base solution
            model = base_solver.model()
            solution = [''] * self.n
            for elem in self.elements:
                pos = model[position[elem]].as_long()
                solution[pos] = elem
            return solution
        
        print("Z3 Phase 1: Finding maximum minimum distance...")
        
        # Binary search for maximum minimum distance
        low, high = 1, self.n - 1
        best_min_distance = 0
        best_solution = None
        
        while low <= high:
            mid = (low + high) // 2
            
            # Create test solver
            test_solver = Solver()
            test_solver.set("timeout", max(1000, time_limit_ms // 20))  # Use fraction of time
            
            # Add all base constraints
            for constraint in base_solver.assertions():
                test_solver.add(constraint)
            
            # Add minimum distance constraints
            for x, y in valid_pairs:
                test_solver.add(Or(
                    position[x] - position[y] >= mid,
                    position[y] - position[x] >= mid
                ))
            
            if test_solver.check() == sat:
                best_min_distance = mid
                model = test_solver.model()
                solution = [''] * self.n
                for elem in self.elements:
                    pos = model[position[elem]].as_long()
                    solution[pos] = elem
                best_solution = solution
                low = mid + 1
                print(f"  Z3: Minimum distance {mid} is achievable")
            else:
                high = mid - 1
                print(f"  Z3: Minimum distance {mid} is not achievable")
        
        if best_solution is None:
            self.last_violations = ["Z3 could not find any solution with distance constraints"]
            return None
        
        print(f"Z3 Phase 1 complete: Maximum minimum distance = {best_min_distance}")
        print("Z3 Phase 2: Maximizing sum of distances...")
        
        # Phase 2: Optimize sum while maintaining minimum distance
        # For Z3, we'll use iterative improvement since it doesn't have built-in optimization
        current_solution = best_solution
        current_sum = sum(abs(current_solution.index(x) - current_solution.index(y)) 
                         for x, y in valid_pairs)
        
        # Try to find better solutions by increasing the target sum
        for target_sum in range(current_sum + 1, current_sum + 50):  # Limited search
            opt_solver = Solver()
            opt_solver.set("timeout", max(1000, time_limit_ms // 10))
            
            # Add base constraints
            for constraint in base_solver.assertions():
                opt_solver.add(constraint)
            
            # Add minimum distance constraint
            for x, y in valid_pairs:
                opt_solver.add(Or(
                    position[x] - position[y] >= best_min_distance,
                    position[y] - position[x] >= best_min_distance
                ))
            
            # Add sum constraint (approximation)
            sum_expr = 0
            for x, y in valid_pairs:
                # Approximate absolute value with two variables
                diff_pos = Int(f'diff_pos_{x}_{y}')
                diff_neg = Int(f'diff_neg_{x}_{y}')
                
                opt_solver.add(diff_pos >= 0)
                opt_solver.add(diff_neg >= 0)
                opt_solver.add(diff_pos >= position[x] - position[y])
                opt_solver.add(diff_neg >= position[y] - position[x])
                opt_solver.add(Or(diff_pos == 0, diff_neg == 0))
                
                sum_expr += diff_pos + diff_neg
            
            opt_solver.add(sum_expr >= target_sum)
            
            if opt_solver.check() == sat:
                model = opt_solver.model()
                solution = [''] * self.n
                for elem in self.elements:
                    pos = model[position[elem]].as_long()
                    solution[pos] = elem
                current_solution = solution
                current_sum = target_sum
            else:
                break  # No better solution found
        
        # Calculate final metrics
        min_dist = min(abs(current_solution.index(x) - current_solution.index(y)) 
                      for x, y in valid_pairs)
        sum_dist = sum(abs(current_solution.index(x) - current_solution.index(y)) 
                      for x, y in valid_pairs)
        print(f"Z3 Phase 2 complete: Minimum distance = {min_dist}, Sum of distances = {sum_dist}")
        
        return current_solution
    
    def solve(self, method: str = "ortools", time_limit: int = 300) -> Optional[List[str]]:
        """
        Solve the constraint satisfaction problem with lexicographic optimization.
        
        Args:
            method: "ortools" or "z3"
            time_limit: Time limit in seconds for OR-Tools, milliseconds for Z3
        """
        self.last_violations = []
        
        if method.lower() == "ortools":
            return self.solve_with_ortools(time_limit)
        elif method.lower() == "z3":
            return self.solve_with_z3(time_limit * 1000)  # Convert to milliseconds
        else:
            raise ValueError("Method must be 'ortools' or 'z3'")

def get_intervals(interval_str):
    # First, parse the positions of intervals
    intervals = [[], []]
    neg_pos = interval_str.split('|')
    positive = 0
    for neg_pos_part in [neg_pos[0], neg_pos[2]]:
        parts = re.split(r'_', neg_pos_part)
        for part in parts:
            if not part:
                intervals[positive].append((None, None))
            elif ":" in part:
                start, end = part.split(':')
                try:
                    start = int(start)
                except:
                    start = float('inf')
                try:
                    end = int(end)
                except:
                    end = float('inf')
                if not positive:
                    start = -start
                    end = -end
                intervals[positive].append((start, end))
            else:
                intervals[positive].append((int(part), int(part)))
        positive = 1

    # Now calculate underscore intervals
    result = []
    positive = 0
    for neg_pos_part in intervals:
        for i in range(len(neg_pos_part) - 1):
            end_of_current = neg_pos_part[i][1]
            start_of_next = neg_pos_part[i+1][0]
            if end_of_current is None:
                if not positive:
                    end_of_current = -float('inf')
                elif result and result[-1][1] == -1:
                    end_of_current = result[-1][0] - 1
                    del result[-1]
                else:
                    end_of_current = 0
            if start_of_next is None:
                if not positive:
                    start_of_next = 0
                else:
                    start_of_next = float('inf')
            if start_of_next - end_of_current <= 1:
                raise ValueError("Invalid interval: overlapping or adjacent intervals found.")
            result.append((end_of_current + 1, start_of_next - 1))
        positive = 1
    
    return result

def generate_unique_strings(n):
    charset = string.ascii_lowercase  # you can expand this (e.g. add digits or uppercase)
    result = []
    length = 1

    while len(result) < n:
        for combo in itertools.product(charset, repeat=length):
            result.append(''.join(combo))
            if len(result) == n:
                return result
        length += 1

def go(strings, instructions, sorter):
    for idx, inst_list in enumerate(instructions):
        current = strings[idx]
        for inst in inst_list:
            targets = []
            for number in inst.numbers:
                targets.append(strings[number])
            # Forbidden constraint
            if inst.instr_type:
                if inst.any:
                    sorter.add_forbidden_constraint_any_y(current, targets, inst.intervals)
                else:
                    for target in targets:
                        if target != current:
                            sorter.add_forbidden_constraint(current, target, inst.intervals)
            else:
                # Maximizing distance constraint
                for target in targets:
                    sorter.add_maximize_distance_constraint(current, target)

def order_table(res, table, roles, dep_pattern):
    new_pos = [0] * len(table)
    for i, old_index in enumerate(res):
        new_pos[old_index] = i
    updated_roles = [False] * len(roles)
    new_table = [table[0]]
    for n in res[1:]:
        row = table[n]
        for j, cell in enumerate(row):
            if roles[j] == 'dependencies':
                if cell:
                    updated_cell = cell.split('; ')
                    for d, instr in enumerate(list(updated_cell)):
                        instr_split = instr.split('.')
                        if dep_pattern[j]:
                            instr = dep_pattern[j][0] + ''.join([instr_split[i]+dep_pattern[j][i+1] for i in range(len(instr_split))])
                        match = re.match(PATTERN_DISTANCE, instr)
                        if not match:
                            match = re.match(PATTERN_AREAS, instr)
                        if match.group("number"):
                            number = int(match.group("number"))
                            new_instr = f"{match.group('prefix')}{"any" if match.group('any') else ''}{new_pos[number]}{match.group('suffix')}"
                            prefix = f"{match.group('prefix')}{"any" if match.group('any') else ''}"
                            str_len = 0
                            prev_str_len = 0
                            id = 0
                            id2 = 0
                            while id < len(new_pos) and str_len <= len(prefix):
                                prev_str_len = str_len
                                if id == id2:
                                    str_len += len(dep_pattern[j][id])
                                    id += 1
                                else:
                                    str_len += len(instr_split[id2])
                                    id2 += 1
                            if id == id2:
                                instr_split[id2-1] = new_instr[prev_str_len:str_len]
                                updated_cell[d] = '.'.join(instr_split)
                            elif not updated_roles[j]:
                                dep_pattern[j][id-1] = new_instr[prev_str_len:str_len]
                                updated_roles[j] = True
                    row[j] = '; '.join(updated_cell)
        new_table.append(row)
    return new_table

def accumulate_dependencies(graph):
    def dfs(node, visited, path, prev_neighbors):
        if node in path:
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            raise ValueError(f"Cycle detected: {' -> '.join(cycle)}")
        # if node in result:
        #     return result[node]

        path.append(node)
        neighbors = {}
        accumulated = {}
        for neighbor in graph[node]:
            if neighbor in prev_neighbors:
                if prev_neighbors[neighbor][1] == 0 or graph[node][neighbor] == 1:
                    warnings.append(f"Warning: {neighbor!r} has a redundant dependency {prev_neighbors[neighbor][0]!r} given by {' <- '.join([str(x) for x in path[path.index(prev_neighbors[neighbor][0]):] + [neighbor]])}")
            neighbors[neighbor] = [node, graph[node][neighbor]]
            accumulated[neighbor] = graph[node][neighbor]
        new_neighbors = dict(neighbors)
        new_neighbors.update(prev_neighbors)
        for neighbor in neighbors:
            if neighbor in graph:
                accumulated.update(dfs(neighbor, visited, path.copy(), new_neighbors))
        
        path.pop()
        # result[node] = accumulated
        return accumulated

    result = {}
    for node in graph:
        result[node] = dfs(node, set(), [], {})
    return result

def sorter(table_original, errors, warnings):
    roles = table_original[0]
    table = table_original[1:]
    alph = generate_unique_strings(max(len(roles), len(table)))
    interm_roles = [role.lower() for role in roles]
    path_index = interm_roles.index('path') if 'path' in interm_roles else -1
    if path_index == -1:
        errors.append("Error: 'path' role not found in roles")
        return table_original
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            cells = cell.split('; ')
            for k, c in enumerate(cells):
                cells[k] = c.strip()
                if j != path_index:
                    cells[k] = cells[k].lower()
            row[j] = '; '.join(cells)
    for i, row in enumerate(table[1:]):
        cell = row[path_index]
        if cell:
            if not re.match(r'^(https?://|file://)', cell):
                warnings.append(f"Warning in row {i+1}, column {alph[path_index]}: {cell!r} is not a valid URL or local path")
    names = {}
    for i, row in enumerate(table[1:], start=1):
        for j, cell in enumerate(row):
            if roles[j] == 'names' and cell:
                cell_list = cell.split('; ')
                for name in cell_list:
                    try:
                        int(name)
                        errors.append(f"Error in row {i}, column {alph[j]}: {name!r} is not a valid name")
                        return table_original
                    except ValueError:
                        pass
                    if "_" in name or ":" in name or "|" in name:
                        errors.append(f"Error in row {i}, column {alph[j]}: {name!r} contains invalid characters (_ : |)")
                    if name in names:
                        errors.append(f"Error in row {i}, column {alph[j]}: name {name!r} already exists in row {names[name]}")
                    names[name] = i
    if errors:
        return table_original
    attributes = {}
    for i, row in enumerate(table[1:], start=1):
        for j, cell in enumerate(row):
            is_sprawl = roles[j] == 'sprawl'
            if roles[j] == 'attributes' or is_sprawl:
                if not cell:
                    continue
                cell_list = cell.split('; ')
                for instr in cell_list:
                    if not instr:
                        errors.append(f"Error in row {i}, column {alph[j]}: empty attribute name")
                        return table_original
                    try:
                        k = int(instr)
                        if k < 1 or k > len(table)-1:
                            errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} points to an invalid row {k}")
                            return table_original
                    except ValueError:
                        k = -1
                        if instr in names:
                            k = names[instr]
                    if k == -1:
                        k = instr
                    if k not in attributes:
                        attributes[k] = {}
                    if i not in attributes[k]:
                        attributes[k][i] = is_sprawl
                    else:
                        warnings.append(f"Redundant attribute {instr!r} in row {i}, column {alph[j]}")

    attributes = accumulate_dependencies(attributes)
    valid_row_indexes = []
    new_indexes = list(range(len(table)))
    to_old_indexes = []
    cat_rows = []
    new_index = 0
    for i, row in enumerate(table[1:], start=1):
        if row[path_index]:
            valid_row_indexes.append(i)
            new_indexes[i] = new_index
            new_index += 1
            to_old_indexes.append(i)
        else:
            cat_rows.append(i)
    if not valid_row_indexes:
        errors.append("Error: No valid rows found in the table!")
        return table_original
    for cat in list(attributes.keys()):
        if type(cat) is not int:
            continue
        attributes[cat] = {k: v for k, v in attributes[cat].items() if table[k][path_index]}
        if not attributes[cat]:
            del attributes[cat]
    attributes_table = [set() for _ in range(len(table))]
    for attr, deps in attributes.items():
        for dep in deps:
            attributes_table[dep].add(attr)
    instr_table = [[] for _ in range(len(table))]
    dep_pattern = [cell.split('.') for cell in table[0]]
    for i, row in enumerate(table[1:], start=1):
        if not table[i][path_index] and i not in attributes:
            continue
        for j, cell in enumerate(row):
            if roles[j] == 'dependencies' and cell:
                cell_list = cell.split('; ')
                for instr in cell_list:
                    if instr:
                        instr_split = instr.split('.')
                        if len(instr_split) != len(dep_pattern[j])-1 and len(dep_pattern[j]) > 1:
                            errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} does not match dependencies pattern {dep_pattern[j]!r}")
                            return table_original
                        if len(dep_pattern[j]) > 1:
                            instr = dep_pattern[j][0] + ''.join([instr_split[i]+dep_pattern[j][i+1] for i in range(len(instr_split))])
                        match = re.match(PATTERN_DISTANCE, instr)
                        intervals = []
                        if instr_type := not match:
                            match = re.match(PATTERN_AREAS, instr)
                            if not match:
                                errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} does not match expected format")
                                return table_original
                            intervals = get_intervals(instr)
                        numbers = []
                        if i==8:
                            print(f"Processing instruction: {instr!r} in row {i}, column {alph[j]}")
                        if match.group("number"):
                            number = int(match.group("number"))
                            if number == 0 or number > len(table):
                                errors.append(f"Error in row {i}, column {alph[j]}: invalid number.")
                                return table_original
                            if table[number][path_index]:
                                numbers.append(number)
                            name = number
                        elif not (name := match.group("name")):
                            errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} does not match expected format")
                            return table_original
                        if name in attributes:
                            for r in attributes[name].keys():
                                numbers.append(r)
                        elif match.group("name"):
                            if name not in names:
                                errors.append(f"Error in row {i}, column {alph[j]}: attribute {name!r} does not exist")
                                return table_original
                            if table[names[name]][path_index]:
                                numbers.append(names[name])
                            if names[name] in attributes:
                                for r in attributes[names[name]].keys():
                                    numbers.append(r)
                        numbers = list(map(lambda x: new_indexes[x], numbers))
                        instr_table[i].append(instr_struct(instr_type, match.group("any"), numbers, intervals))
    for i in valid_row_indexes:
        for j in attributes_table[i]:
            if type(j) is int:
                instr_table[i] = list(set(instr_table[i] + instr_table[j]))
    instr_table_int = []
    for i in valid_row_indexes:
        instr_table_int.append(instr_table[i])
    instr_table = instr_table_int
    # detect cycles in instr_table
    def has_cycle(instr_table, visited, stack, node, after=True):
        visited.add(node)
        stack.append(node)
        for neighbor in instr_table[node]:
            if neighbor.any or not neighbor.instr_type or (neighbor.intervals[0] != (-float("inf"), -1) if after else neighbor.intervals[-1] != (1, float("inf"))):
                continue
            for target in neighbor.numbers:
                if target not in visited:
                    if has_cycle(instr_table, visited, stack, target, after):
                        return True
                elif target in stack and len(stack) > 1:
                    return True
        stack.pop()
        return False
    for p in [0, 1]:
        stack = []
        visited = set()
        for i in range(len(instr_table)):
            if has_cycle(instr_table, visited, stack, i, p):
                stack.append(i)
                errors.append(f"Cycle detected: {(' after ' if p else ' before ').join([str(to_old_indexes[k]) for k in stack])}")
                return table_original
    sorter = EfficientConstraintSorter(alph[:len(valid_row_indexes)])
    go(alph, instr_table, sorter)
    for cat, rows in attributes.items():
        category = [k for k, v in rows.items() if v]
        if len(category) > 1:
            sorter.add_group_maximize(set(map(lambda x: new_indexes[x], category)))

    # Solve the problem
    print("Solving constraint-based sorting problem...")
    solution = sorter.solve(method="ortools", time_limit=300)
    
    if not solution:
        errors.append("No valid solution found!")
        return table_original
    elif type(solution) is string:
        errors.append(f"Error when sorting: {solution!r}")
        return table_original
    print(f"Solution found: {solution}")
    
    # Show positions for clarity
    print("\nPositions:")
    for i, elem in enumerate(solution):
        print(f"Position {i}: {elem}")

    res = [0] + [to_old_indexes[alph.index(elem)] for elem in solution]
    i = 1
    while i < len(res):
        d = 0
        while d < len(cat_rows):
            e = cat_rows[d]
            if e in attributes_table[res[i]]:
                res.insert(i, e)
                i += 1
                del cat_rows[d]
            else:
                d += 1
        i += 1
    res.extend(cat_rows)
    new_table = order_table(res, table, roles, dep_pattern)
    with open(f'{PLAYLISTS_PATH}/{table[0][0]}.txt', 'w') as f:
        for i in res[1:]:
            if table[i][path_index]:
                f.write(f"{table[i][path_index]}\n")
    return [roles] + new_table

if __name__ == "__main__":
    try:
        import pyperclip
        clipboard_content = pyperclip.paste()
        table = [line.split('\t') for line in re.split(r'\r?\n', clipboard_content)]
        crop_line = len(table)
        crop_column = len(table[0])
        for i in range(len(table) - 1, -1, -1):
            if table[i] != ['']* len(table[0]):
                crop_line = i + 1
                break
        for j in range(len(table[0]) - 1, -1, -1):
            if any(row[j] for row in table):
                crop_column = j + 1
                break
        table = table[1:crop_line]
        for i in range(len(table)):
            table[i] = table[i][1:crop_column]
        warnings = []
        errors = []
        roles = table[0]
        for i, role in enumerate(roles):
            if role not in ROLES:
                errors.append(f"Error: Invalid role {role!r} found in roles")
                result = table
                break
        else:
            result = sorter(table, errors, warnings)
        if errors:
            print("Errors found:")
            for error in errors:
                print(f"- {error}")
        if warnings:
            print("Warnings found:")
            for warning in warnings:
                print(f"- {warning}")
        new_clipboard_content = '\n'.join(['\t'.join(row) for row in result])
        pyperclip.copy(new_clipboard_content)
        input("Sorted table copied to clipboard. Press Enter to exit.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     input("Press Enter to exit.")
    except pyperclip.PyperclipException:
        print("Clipboard access error. Please ensure you have the pyperclip module installed and your clipboard is accessible.")
        input("Press Enter to exit.")