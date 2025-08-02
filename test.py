from typing import List, Tuple, Set, Optional, Dict
import itertools
from ortools.sat.python import cp_model
from z3 import *
import time

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
        """Solve using Google OR-Tools CP-SAT solver."""
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
        
        # Objective: maximize distances for maximize_distance constraints
        if self.maximize_distance:
            distance_vars = []
            for x, y in self.maximize_distance:
                if x not in self.elements or y not in self.elements:
                    continue
                
                # Create variable for absolute distance
                abs_distance = model.NewIntVar(0, self.n - 1, f'abs_dist_{x}_{y}')
                distance_vars.append(abs_distance)
                
                # abs_distance = |pos[x] - pos[y]|
                model.AddAbsEquality(abs_distance, position[x] - position[y])
            
            if distance_vars:
                model.Maximize(sum(distance_vars))
        
        # Solve
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
    def solve_with_z3(self, time_limit_ms: int = 300000) -> Optional[List[str]]:
        """Solve using Z3 SMT solver."""
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
        
        # Check satisfiability
        if solver.check() == sat:
            model = solver.model()
            
            # Extract solution
            solution = [''] * self.n
            for elem in self.elements:
                pos = model[position[elem]].as_long()
                solution[pos] = elem
            
            # If we have distance maximization constraints, try to optimize
            if self.maximize_distance:
                return self._optimize_z3_solution(solution, position, solver)
            
            return solution
        else:
            self.last_violations = ["Z3 solver could not find a satisfiable solution"]
            return None
    
    def _optimize_z3_solution(self, initial_solution: List[str], position: Dict, base_solver: Solver) -> List[str]:
        """Optimize Z3 solution for distance maximization using iterative improvement."""
        current_solution = initial_solution.copy()
        
        def calculate_total_distance(solution: List[str]) -> int:
            pos_map = {elem: i for i, elem in enumerate(solution)}
            total = 0
            for x, y in self.maximize_distance:
                if x in pos_map and y in pos_map:
                    total += abs(pos_map[x] - pos_map[y])
            return total
        
        current_distance = calculate_total_distance(current_solution)
        
        # Try to improve by adding distance constraints iteratively
        for target_distance in range(current_distance + 1, current_distance + 100):
            optimizer = Solver()
            optimizer.set("timeout", 10000)  # 10 second timeout for each optimization attempt
            
            # Copy all constraints from base solver
            for constraint in base_solver.assertions():
                optimizer.add(constraint)
            
            # Add distance optimization constraint
            distance_sum = 0
            for x, y in self.maximize_distance:
                if x in self.elements and y in self.elements:
                    # Use absolute value approximation
                    abs_var = Int(f'abs_{x}_{y}')
                    optimizer.add(abs_var >= position[x] - position[y])
                    optimizer.add(abs_var >= position[y] - position[x])
                    distance_sum += abs_var
            
            optimizer.add(distance_sum >= target_distance)
            
            if optimizer.check() == sat:
                model = optimizer.model()
                solution = [''] * self.n
                for elem in self.elements:
                    pos = model[position[elem]].as_long()
                    solution[pos] = elem
                current_solution = solution
                current_distance = target_distance
            else:
                break  # No better solution found
        
        return current_solution

    def solve(self, method: str = "ortools", time_limit: int = 300) -> Optional[List[str]]:
        """
        Solve the constraint satisfaction problem.
        
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

# Example usage and testing
def test_solvers():
    """Test both solvers with a sample problem."""
    elements = [f"elem_{i}" for i in range(20)]  # Start with smaller test
    
    sorter = EfficientConstraintSorter(elements)
    
    # Add some test constraints
    sorter.add_forbidden_constraint("elem_0", "elem_1", [(-2, 2)])
    sorter.add_forbidden_constraint("elem_2", "elem_3", [(1, float('inf'))])
    sorter.add_forbidden_constraint("elem_2", "elem_3", [(-float('inf'), -1)])
    sorter.add_maximize_distance_constraint("elem_0", "elem_19")
    
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
    test_solvers()