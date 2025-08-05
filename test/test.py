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
            self.last_violations = [f"OR-Tools solver status: {solver.StatusName(status)}"]
            return None

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

# Example usage and testing
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
    test_solvers()