import re
import random
from typing import List, Tuple, Set, Optional, Dict
import itertools
import string
from z3 import *
import threading
import pickle
import json
import time
from config import PLAYLISTS_PATH, DEFAULT_PLAYLIST_FILE

ROLES = ['path', 'names', 'attributes', 'dependencies', 'sprawl']
PATTERN_DISTANCE = r'^(?P<prefix>as far as possible from )(?P<any>any)?((?P<number>\d+)|(?P<name>.+))(?P<suffix>)$'
PATTERN_AREAS = r'^(?P<prefix>.*\|)(?P<any>any)?((?P<number>\d+)|(?P<name>.+))(?P<suffix>\|.*)$'

ortools_loaded = threading.Event()
fst_row = None
fst_col = None

class instr_struct:
    def __init__(self, is_constraint: int, any: bool, numbers: List[int], intervals: List[Tuple[int, int]], path: List[int] = []):
        self.is_constraint = is_constraint
        self.any = any
        self.numbers = numbers
        self.intervals = intervals
        self.path = path

    def __eq__(self, other):
        if not isinstance(other, instr_struct):
            return False
        return (self.is_constraint == other.is_constraint and
                self.any == other.any and
                sorted(self.numbers) == sorted(other.numbers) and
                sorted(self.intervals) == sorted(other.intervals))

    def __hash__(self):
        return hash((
            self.is_constraint,
            self.any,
            tuple(sorted(self.numbers)),
            tuple(sorted(self.intervals))
        ))

class EfficientConstraintSorter:
    def __init__(self, elements: List[str], table, urls, to_old_indexes, alph, cat_rows, attributes_table, dep_pattern, errors, path_index):
        self.elements = elements
        self.table = table
        self.urls = urls
        self.to_old_indexes = to_old_indexes
        self.alph = alph
        self.cat_rows = cat_rows
        self.attributes_table = attributes_table
        self.dep_pattern = dep_pattern
        self.errors = errors
        self.path_index = path_index
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

        self.file_path = os.path.join(PLAYLISTS_PATH, self.table[0][0]+".pkl")
        os.makedirs(PLAYLISTS_PATH, exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "wb") as f:
                pickle.dump({
                    "input": {
                        "elements": [],
                        "constraints": [],
                        "maximize_distance": []
                    },
                    "output": {}
                }, f)
        with open(self.file_path, "rb") as f:
            prev_sorting = pickle.load(f)
        new_sorting = {
            "input": {
                "elements": self.elements,
                "constraints": self.constraints,
                "maximize_distance": self.maximize_distance
            },
            "output": {}
        }
        if prev_sorting["input"] == new_sorting["input"]:
            if prev_sorting["output"]["error"]:
                self.errors.append(prev_sorting["output"]["error"])
                return self.table
            elif prev_sorting["output"]["done"]:
                return prev_sorting["output"]["new_table"]
        else:
            new_sorting["input"] = prev_sorting["input"]
        
        # Handle optimization if required
        if self.maximize_distance:
            result = self._solve_lexicographic_ortools(model, prev_sorting, position, time_limit_seconds)
            if result is None and self.last_violations:
                print("Lexicographic optimization failed. Error details:")
                for i, violation in enumerate(self.last_violations, 1):
                    print(f"  {i}. {violation}")
            return result

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [''] * self.n
            new_urls = [''] * self.n
            for i, elem in enumerate(self.elements):
                pos = solver.Value(position[elem])
                solution[pos] = elem
                new_urls[pos] = self.urls[i]
            prev_sorting["output"]["done"] = True
            prev_sorting["output"]["best_solution"] = solution
            prev_sorting["output"]["urls"] = new_urls
            self._save_incremental_solution(prev_sorting)
            return solution
        else:
            if status == cp_model.INFEASIBLE:
                print("Model is infeasible. Finding conflicting constraints...")
                self.last_violations = self._find_conflicting_constraints_ortools()
            else:
                self.last_violations = [f"OR-Tools solver status: {solver.StatusName(status)}"]
            if result is None and self.last_violations:
                print("Lexicographic optimization failed. Error details:")
                for i, violation in enumerate(self.last_violations, 1):
                    print(f"  {i}. {violation}")
            prev_sorting["output"]["error"] = self.last_violations
            self._save_incremental_solution(prev_sorting)
            return self.table
            

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

    def _add_single_ortools_constraint(self, model, position, constraint_data, enforce_if = None):
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

    def _solve_lexicographic_ortools(self, model, saved, position, time_limit_seconds):
        """Maximize the sum of all distances in maximize_distance."""
        k = len(self.maximize_distance)
        start_time = time.time()
        elapsed = time.time() - start_time

        # Create distance variables and constraints
        dist_vars = []
        for (x, y) in self.maximize_distance:
            diff = model.NewIntVar(-self.n + 1, self.n - 1, f'diff_{x}_{y}')
            model.Add(diff == position[x] - position[y])
            dist = model.NewIntVar(0, self.n - 1, f'dist_{x}_{y}')
            model.AddAbsEquality(dist, diff)
            dist_vars.append(dist)

        total_dist = model.NewIntVar(0, k * (self.n - 1), 'total_dist')
        model.Add(total_dist == sum(dist_vars))
        model.Maximize(total_dist)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(1, time_limit_seconds - elapsed)
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [''] * self.n
            new_urls = [''] * self.n
            for i, elem in enumerate(self.elements):
                pos = solver.Value(position[elem])
                solution[pos] = elem
                new_urls[pos] = self.urls[i]

            saved["output"]["done"] = True
            # saved["output"]["optimal_values"] = optimal_values
            saved["output"]["best_solution"] = solution
            saved["output"]["urls"] = new_urls
            return self._save_incremental_solution(saved)
        else:
            if status == cp_model.INFEASIBLE:
                self.last_violations = ["Unexpected infeasibility during optimization."]
            else:
                self.last_violations = [f"Optimization failed: {solver.StatusName(status)}"]
            saved["output"]["error"] = self.last_violations
            return self._save_incremental_solution(saved)

    def _save_incremental_solution(self, saved):
        """Save an incremental solution to file with metadata."""
        solution = saved["output"]["best_solution"]
        if not solution:
            errors.append("No valid solution found!")
            return self.table
        print(f"Solution found: {solution}")
        
        # Show positions for clarity
        print("\nPositions:")
        for i, elem in enumerate(solution):
            print(f"Position {i}: {elem}")

        res = [0] + [self.to_old_indexes[self.alph.index(elem)] for elem in solution]
        i = 1
        while i < len(res):
            d = 0
            while d < len(self.cat_rows):
                e = self.cat_rows[d]
                if e in self.attributes_table[res[i]]:
                    for s in self.attributes_table[e]:
                        if s in self.cat_rows:
                            res.insert(i, s)
                            i += 1
                            del self.cat_rows[self.cat_rows.index(s)]
                    res.insert(i, e)
                    i += 1
                    del self.cat_rows[d]
                else:
                    d += 1
            i += 1
        res.extend(self.cat_rows)
        new_table = order_table(res, self.table, roles, self.dep_pattern)
        saved["output"]["new_table"] = new_table
        with open(self.file_path, 'wb') as f:
            pickle.dump(saved, f)
        global fst_row, fst_col
        result = [fst_row, roles] + new_table
        for i in range(len(result)):
            result[i] = [fst_col[i]] + result[i]
        with open(self.file_path.replace(".pkl", "_table.txt"), 'w') as f:
            f.write('\n'.join(['\t'.join(row) for row in result]))
        return new_table
    
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

def preload_ortools():
    global cp_model
    from ortools.sat.python import cp_model as _cp_model
    cp_model = _cp_model
    ortools_loaded.set()

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
            if inst.is_constraint:
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
                        if len(dep_pattern[j]) > 1:
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
    def dfs(node, path):
        if node in path:
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            raise ValueError(f"Cycle detected: {' -> '.join(cycle)}")
        if node in result:
            return result[node]

        path.append(node)
        accumulated = dict(graph[node])
        for neighbor in graph[node]:
            if neighbor not in graph:
                continue
            res = dfs(neighbor, path.copy())
            for key, value in res.items():
                accumulated[key] = tuple(value)
                accumulated[key][1].append(node)
                if key in graph[node]:
                    warnings.append(f"Warning: {key!r} has a redundant dependency {node!r} given by {' -> '.join([str(x) for x in accumulated[key][1]])}")
        path.pop()
        result[node] = accumulated
        return accumulated

    result = {}
    for node in graph:
        dfs(node, [])
    return result

def sorter(table, roles, errors, warnings):
    alph = generate_unique_strings(max(len(roles), len(table)))
    path_index = roles.index('path') if 'path' in roles else -1
    if path_index == -1:
        errors.append("Error: 'path' role not found in roles")
        return table
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            cells = cell.split(';')
            for k, c in enumerate(cells):
                cells[k] = c.strip()
                if j != path_index:
                    cells[k] = cells[k].lower()
                    if roles[j] == 'attributes' or roles[j] == 'sprawl':
                        if cells[k].endswith("-fst"):
                            cells[k] = cells[k][:-4].strip() + " -fst"
                        elif cells[k].endswith("-lst"):
                            cells[k] = cells[k][:-4].strip() + " -lst"
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
                        return table
                    except ValueError:
                        pass
                    if "_" in name or ":" in name or "|" in name or ("-" in name and not name.endswith("-fst") and not name.endswith("-lst")):
                        errors.append(f"Error in row {i}, column {alph[j]}: {name!r} contains invalid characters (_ : | -)")
                    if name in ["fst", "lst"]:
                        errors.append(f"Error in row {i}, column {alph[j]}: {name!r} is a reserved name")
                    if name in names:
                        errors.append(f"Error in row {i}, column {alph[j]}: name {name!r} already exists in row {names[name]}")
                    names[name] = i
    if errors:
        return table
    attributes = {}
    first_element = None
    last_element = None
    fst_cat = {}
    lst_cat = {}
    for i, row in enumerate(table[1:], start=1):
        for j, cell in enumerate(row):
            is_sprawl = roles[j] == 'sprawl'
            if roles[j] == 'attributes' or is_sprawl:
                if not cell:
                    continue
                new_cell_list = []
                cell_list = cell.split('; ')
                for instr in cell_list:
                    if not instr:
                        errors.append(f"Error in row {i}, column {alph[j]}: empty attribute name")
                        return table
                    if is_fst := instr.endswith("-fst"):
                        instr = instr[:-5]
                    elif instr == "fst":
                        first_element = i
                    elif is_lst := instr.endswith("-lst"):
                        instr = instr[:-5]
                    elif instr == "lst":
                        last_element = i
                    elif "-fst" in instr:
                        errors.append(f"Error in row {i}, column {alph[j]}: '-fst' is not at the end of {instr!r}")
                        return table
                    try:
                        k = int(instr)
                        if k < 1 or k > len(table)-1:
                            errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} points to an invalid row {k}")
                            return table
                    except ValueError:
                        k = -1
                        if instr in names:
                            k = names[instr]
                    if k == -1:
                        k = instr
                    if k not in attributes:
                        attributes[k] = {}
                    if i not in attributes[k]:
                        attributes[k][i] = (is_sprawl, [k])
                    else:
                        warnings.append(f"Redundant attribute {instr!r} in row {i}, column {alph[j]}")
                    if is_fst:
                        fst_cat[i] = k
                    elif is_lst:
                        lst_cat[i] = k
                    new_cell_list.append(instr)
                row[j] = '; '.join(new_cell_list)
    attributes = accumulate_dependencies(attributes)
    urls = [(row[i], []) for i in range(len(table))]
    for i, row in enumerate(table[1:], start=1):
        if row[path_index] and i in attributes:
            for k in attributes[i].keys():
                if urls[k][0]:
                    errors.append(f"Error in row {i}, column {alph[path_index]}: a URL given by {' -> '.join([str(x) for x in urls[i][1]])} conflicts with another given by {' -> '.join([str(x) for x in attributes[i][k][1]])}")
                    return table
                urls[k] = (row[path_index], attributes[i][k][1])
    valid_row_indexes = []
    is_valid = [False] * len(table)
    new_indexes = list(range(len(table)))
    to_old_indexes = []
    cat_rows = []
    new_index = 0
    for i, row in enumerate(table[1:], start=1):
        if row[path_index]:
            is_valid[i] = True
            valid_row_indexes.append(i)
            new_indexes[i] = new_index
            new_index += 1
            to_old_indexes.append(i)
        else:
            cat_rows.append(i)
    if not valid_row_indexes:
        errors.append("Error: No valid rows found in the table!")
        return table

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
    for k, v in fst_cat.items():
        if is_valid[k]:
            t = v
            while fst_cat[t]:
                t = fst_cat[t]
            for i in attributes[t].keys():
                instr_table[i].append(instr_struct(True, False, [new_indexes[k]], [(-float("inf"), -1)]))
    for k, v in lst_cat.items():
        if is_valid[k]:
            t = v
            while lst_cat[t]:
                t = lst_cat[t]
            for i in attributes[t].keys():
                instr_table[i].append(instr_struct(True, False, [new_indexes[k]], [(1, float("inf"))]))
    if first_element is not None:
        for i in valid_row_indexes:
            instr_table[i].append(instr_struct(True, False, [new_indexes[first_element]], [(-float("inf"), -1)]))
    if last_element is not None:
        for i in valid_row_indexes:
            instr_table[i].append(instr_struct(True, False, [new_indexes[last_element]], [(1, float("inf"))]))
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
                            return table
                        if len(dep_pattern[j]) > 1:
                            instr = dep_pattern[j][0] + ''.join([instr_split[i]+dep_pattern[j][i+1] for i in range(len(instr_split))])
                        match = re.match(PATTERN_DISTANCE, instr)
                        intervals = []
                        if is_constraint := not match:
                            match = re.match(PATTERN_AREAS, instr)
                            if not match:
                                errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} does not match expected format")
                                return table
                            intervals = get_intervals(instr)
                        numbers = []
                        if match.group("number"):
                            number = int(match.group("number"))
                            if number == 0 or number > len(table):
                                errors.append(f"Error in row {i}, column {alph[j]}: invalid number.")
                                return table
                            if table[number][path_index]:
                                numbers.append(number)
                            name = number
                        elif not (name := match.group("name")):
                            errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} does not match expected format")
                            return table
                        if name in attributes:
                            for r in attributes[name].keys():
                                numbers.append(r)
                        elif match.group("name"):
                            if name not in names:
                                errors.append(f"Error in row {i}, column {alph[j]}: attribute {name!r} does not exist")
                                return table
                            if table[names[name]][path_index]:
                                numbers.append(names[name])
                            if names[name] in attributes:
                                for r in attributes[names[name]].keys():
                                    numbers.append(r)
                        numbers = list(map(lambda x: new_indexes[x], numbers))
                        instr_table[i].append(instr_struct(is_constraint, match.group("any"), numbers, intervals))
    for i in valid_row_indexes:
        for j in attributes_table[i]:
            if type(j) is int:
                for x in instr_table[j]:
                    if x not in instr_table[i]:
                        x.path = attributes[j][i][1] + x.path
                        instr_table[i].append(x)
                    elif len(instr_table[i][instr_table[i].index(x)].path) == 1:
                        warnings.append(f"Redundant instruction {x!r} in row {i}, column {alph[j]} given by {' -> '.join([str(x) for x in attributes[j][i][1] + x.path])}")
    instr_table_int = []
    for i in valid_row_indexes:
        instr_table_int.append(list(set(instr_table[i])))
    instr_table = instr_table_int
    # detect cycles in instr_table
    def has_cycle(instr_table, visited, stack, node, after=True):
        stack.append([to_old_indexes[node]])
        visited.add(node)
        for neighbor in instr_table[node]:
            if neighbor.any or not neighbor.is_constraint or (neighbor.intervals[0] != (-float("inf"), -1) if after else neighbor.intervals[-1] != (1, float("inf"))):
                continue
            for target in neighbor.numbers:
                stack[-1][1:] = neighbor.path
                if target not in visited:
                    if has_cycle(instr_table, visited, stack, target, after):
                        return True
                elif any(target == k[0] for k in stack):
                    return True
        stack.pop()
        return False
    for p in [0, 1]:
        visited = set()
        stack = []
        for i in range(len(instr_table)):
            if has_cycle(instr_table, visited, stack, i, p):
                errors.append(f"Cycle detected: {(' after ' if p else ' before ').join(['->'.join([str(x) for x in k]) for k in stack])}")
                return table
    sorter = EfficientConstraintSorter(alph[:len(valid_row_indexes)], table, urls, to_old_indexes, alph, cat_rows, attributes_table, dep_pattern, errors, path_index)
    go(alph, instr_table, sorter)
    for cat, rows in attributes.items():
        category = [k for k, v in rows.items() if v[0]]
        if len(category) > 1:
            sorter.add_group_maximize(set(map(lambda x: new_indexes[x], category)))

    # Solve the problem
    print("Solving constraint-based sorting problem...")
    ortools_loaded.wait(timeout=30)
    return sorter.solve(time_limit=300000000000)

if __name__ == "__main__":
    try:
        threading.Thread(target=preload_ortools, daemon=True).start()
        import pyperclip
        # clipboard_content = pyperclip.paste()
        with open('data/test.txt', 'r') as f:
            clipboard_content = f.read()
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
        fst_row = table[0]
        fst_col = [row[0] for row in table]
        table = table[1:crop_line]
        for i in range(len(table)):
            table[i] = table[i][1:crop_column]
        warnings = []
        errors = []
        roles = table[0]
        table = table[1:]
        for i, role in enumerate(roles):
            role = role.strip().lower()
            if role not in ROLES:
                errors.append(f"Error: Invalid role {role!r} found in roles")
                result = [roles] + table
                break
            roles[i] = role
        else:
            result = sorter(table, roles, errors, warnings)
        if errors:
            print("Errors found:")
            for error in errors:
                print(f"- {error}")
        if warnings:
            print("Warnings found:")
            for warning in warnings:
                print(f"- {warning}")
        result = [fst_row, roles] + result
        for i in range(len(result)):
            result[i] = [fst_col[i]] + result[i]
        new_clipboard_content = '\n'.join(['\t'.join(row) for row in result])
        pyperclip.copy(new_clipboard_content)
        input("Sorted table copied to clipboard. Press Enter to exit.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     input("Press Enter to exit.")
    except KeyboardInterrupt:
        print("\nSorting interrupted by user.")