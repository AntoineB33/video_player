from datetime import datetime
import re
import sys
from typing import List, Tuple, Set, Optional
import itertools
import string
import os
from z3 import Solver, Int, And, Distinct, Or, Bool, sat
import threading
import multiprocessing
import pickle
import time
import pyperclip
import copy
from config import PLAYLISTS_PATH

ROLES = ['path', 'names', 'attributes', 'dependencies', 'sprawl']
PATTERN_DISTANCE = r'^(?P<prefix>as far as possible from )(?P<any>any)?((?P<number>\d+)|(?P<name>.+))(?P<suffix>)$'
PATTERN_AREAS = r'^(?P<prefix>.*\|)(?P<any>any)?((?P<number>\d+)|(?P<name>.+))(?P<suffix>\|.*)$'

ortools_loaded = threading.Event()
fst_row = None
fst_col = None
user_input = None
stop_requested = False

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
    def __init__(self, saved, preload_thread, errors, existing_pb, to_pyperclip):
        self.saved = saved
        self.elements = saved.get("data", {}).get("elements", [])
        self.table = saved.get("input", {}).get("table", [])
        self.urls = saved.get("data", {}).get("urls", [])
        self.to_old_indexes = saved.get("data", {}).get("to_old_indexes", [])
        self.cat_rows = saved.get("data", {}).get("cat_rows", [])
        self.attributes_table = saved.get("data", {}).get("attributes_table", [])
        self.roles = saved.get("data", {}).get("roles", [])
        self.dep_pattern = saved.get("data", {}).get("dep_pattern", [])
        self.path_index = saved.get("data", {}).get("path_index", 0)
        self.preload_thread = preload_thread
        self.errors = errors
        self.existing_pb = existing_pb
        self.manual = to_pyperclip
        self.n = len(self.elements)
        self.element_to_idx = {name: i for i, name in enumerate(self.elements)}
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

    def _return_result(self, solver, prev_sorting, position, status, to_pyperclip = False):
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [''] * self.n
            new_urls = [''] * self.n
            for i, elem in enumerate(self.elements):
                pos = solver.Value(position[elem])
                solution[pos] = elem
                new_urls[pos] = self.urls[i]
            prev_sorting["output"]["done"] = not self.maximize_distance
            prev_sorting["output"]["best_solution"] = solution
            prev_sorting["output"]["urls"] = new_urls
            self._save_incremental_solution(prev_sorting, to_pyperclip)
        else:
            if status == cp_model.INFEASIBLE:
                print("Model is infeasible. Finding conflicting constraints...")
                self.errors.append(self._find_conflicting_constraints_ortools())
            else:
                self.errors.append(f"OR-Tools solver status: {solver.StatusName(status)}")
            prev_sorting["output"]["error"] = self.errors
            self._save_incremental_solution(prev_sorting, to_pyperclip)

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
        self.preload_thread.join()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = time_limit_seconds

        self.file_path = os.path.join(PLAYLISTS_PATH, self.table[0][0]+".pkl")
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                prev_sorting = pickle.load(f)
        else:
            prev_sorting = {"data": {}}
        prev_sorting["input"] = self.saved["input"]
        if "fst_row" in self.saved:
            prev_sorting["fst_row"] = self.saved["fst_row"]
            prev_sorting["fst_col"] = self.saved["fst_col"]
        to_pyperclip = self.manual
        if self.existing_pb or prev_sorting["data"] == self.saved["data"]:
            if prev_sorting["output"]["error"]:
                self.errors.extend(prev_sorting["output"]["error"])
                return
            self._save_incremental_solution(prev_sorting, to_pyperclip)
            if prev_sorting["output"]["done"]:
                return
        else:
            prev_sorting["data"] = self.saved["data"]
            prev_sorting["output"] = {"error": [], "best_dist": None}
            self.p = multiprocessing.Process(target=input_listener)
            self.p.start()
            status = self.solver.Solve(model)
            self.p.terminate()
            self.p.join()
            self._return_result(self.solver, prev_sorting, position, status, to_pyperclip)
            to_pyperclip = False
        if self.maximize_distance:
            status = self._solve_lexicographic_ortools(model, prev_sorting, position, time_limit_seconds, to_pyperclip)
            self._return_result(self.solver, prev_sorting, position, status)

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
        self.solver = cp_model.CpSolver()
        status = self.solver.Solve(model)

        if status == cp_model.INFEASIBLE:
            infeasible_assumption_indices = self.solver.SufficientAssumptionsForInfeasibility()
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
            if x not in self.elements or y not in self.elements:
                return
            
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
            if x not in self.elements:
                return

            valid_y_list = [y for y in y_list if y in self.elements]
            if not valid_y_list:
                return

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

    def _solve_lexicographic_ortools(self, model, saved, position, time_limit_seconds, to_pyperclip = False):
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

        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = max(1, time_limit_seconds - elapsed)
        progress_cb = ProgressCallback(self, saved, position, total_dist, to_pyperclip)
        self.p = multiprocessing.Process(target=input_listener)
        self.p.start()
        status = self.solver.SolveWithSolutionCallback(model, progress_cb)
        self.p.terminate()
        self.p.join()
        return status

    def order_table(self, res):
        table = self.table
        roles = self.roles
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
                            if len(self.dep_pattern[j]) > 1:
                                instr = self.dep_pattern[j][0] + ''.join([instr_split[i]+self.dep_pattern[j][i+1] for i in range(len(instr_split))])
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
                                        str_len += len(self.dep_pattern[j][id])
                                        id += 1
                                    else:
                                        str_len += len(instr_split[id2])
                                        id2 += 1
                                if id == id2:
                                    instr_split[id2-1] = new_instr[prev_str_len:str_len]
                                    updated_cell[d] = '.'.join(instr_split)
                                elif not updated_roles[j]:
                                    self.dep_pattern[j][id-1] = new_instr[prev_str_len:str_len]
                                    updated_roles[j] = True
                        row[j] = '; '.join(updated_cell)
            new_table.append(row)
        return new_table

    def _save_incremental_solution(self, saved, in_pyperclip=False):
        """Save an incremental solution to file with metadata."""
        solution = saved["output"]["best_solution"]
        if solution:
            print(f"Solution found: {solution}")
            self.cat_rows_copy = list(self.cat_rows)
            res = [0] + [self.to_old_indexes[self.elements.index(elem)] for elem in solution]
            i = 1
            while i < len(res):
                d = 0
                while d < len(self.cat_rows_copy):
                    e = self.cat_rows_copy[d]
                    if e in self.attributes_table[res[i]]:
                        for s in self.attributes_table[e]:
                            if s in self.cat_rows_copy:
                                res.insert(i, s)
                                i += 1
                                del self.cat_rows_copy[self.cat_rows_copy.index(s)]
                        res.insert(i, e)
                        i += 1
                        del self.cat_rows_copy[d]
                    else:
                        d += 1
                i += 1
            res.extend(self.cat_rows_copy)
            new_table = self.order_table(res)
            saved["output"]["table"] = new_table
        else:
            new_table = self.table
        if self.manual:
            fst_row = saved["fst_row"]
            fst_col = saved["fst_col"]
            result = [self.roles] + new_table
            for i in range(len(result)):
                result[i] = [fst_col[i]] + result[i]
            result.insert(0, fst_row)
            for_pyperclip = '\n'.join(['\t'.join(row) for row in result])
            if in_pyperclip:
                pyperclip.copy(for_pyperclip)
            with open(self.file_path.replace(".pkl", "_table.txt"), 'w', encoding='utf-8') as f:
                f.write(for_pyperclip)
        with open(self.file_path, 'wb') as f:
            pickle.dump(saved, f)
        return
    
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
            self.solve_with_ortools(time_limit)
        elif method.lower() == "z3":
            self.solve_with_z3(time_limit * 1000)
        else:
            raise ValueError("Method must be 'ortools' or 'z3'")

def solver(saved, errors, to_pyperclip, preload_thread = None, existing_pb = False):
    if preload_thread is None:
        preload_thread = threading.Thread(target=preload_ortools, daemon=True)
        preload_thread.start()
    sorter = EfficientConstraintSorter(saved, preload_thread, errors, existing_pb, to_pyperclip)
    for idx, inst_list in enumerate(saved["data"]["instructions"]):
        current = saved["data"]["elements"][idx]
        for inst in inst_list:
            targets = []
            for number in inst.numbers:
                targets.append(saved["data"]["elements"][number])
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
    for cat, rows in saved["data"]["attributes"].items():
        category = [k for k, v in rows.items() if v[0]]
        if len(category) > 1:
            sorter.add_group_maximize(set(map(lambda x: saved["data"]["new_indexes"][x], category)))

    # Solve the problem
    print("Solving constraint-based sorting problem...")
    ortools_loaded.wait(timeout=30)
    sorter.solve(time_limit=300000000000)

def input_listener():
    global stop_requested
    while True:
        text = sys.stdin.readline().strip()
        if text.lower() == "q":
            exit(0)
            
def preload_ortools():
    global cp_model, ProgressCallback
    from ortools.sat.python import cp_model as _cp_model
    cp_model = _cp_model
    class ProgressCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self, parent, saved, position, total_dist, to_pyperclip):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._parent = parent
            self.saved = saved
            self._position = position
            self._total_dist = total_dist
            self.to_pyperclip = to_pyperclip
            self.best_dist = saved["output"]["best_dist"]

        def OnSolutionCallback(self):
            current_dist = self.Value(self._total_dist)

            # Keep track of improvement
            if self.best_dist is None or current_dist > self.best_dist:
                self._parent.p.terminate()
                self._parent.p.join()
                self.best_dist = current_dist
                self.saved["output"]["best_dist"] = current_dist
                self._parent._return_result(self, self.saved, self._position, cp_model.OPTIMAL, self.to_pyperclip)
                self.to_pyperclip = False
                self._parent.p = multiprocessing.Process(target=input_listener)
                self._parent.p.start()


            print(f"Time: {datetime.now()}, Current best distance: {current_dist}")
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
                except Exception:
                    start = float('inf')
                try:
                    end = int(end)
                except Exception:
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

def accumulate_dependencies(graph, warnings):
    def dfs(node, path, warnings):
        if node in path:
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            raise ValueError(f"Cycle detected: {' -> '.join(cycle)}")
        if node in result:
            return result[node]

        path.append(node)
        if (node == 28):
            print("Node 28 reached")
        accumulated = dict(graph[node])
        for neighbor in graph[node]:
            if neighbor not in graph:
                continue
            res = dfs(neighbor, path.copy(), warnings)
            for key, value in res.items():
                if node == 28 and key == 4:
                    print(f"Node 28 has a dependency on {key}")
                accumulated[key] = tuple(value)
                accumulated[key][1].append(node)
                if key in graph[node]:
                    warnings.append(f"Warning: {key!r} has a redundant dependency {node!r} given by {' -> '.join([str(x) for x in accumulated[key][1]])}")
        path.pop()
        result[node] = accumulated
        return accumulated

    result = {}
    for node in graph:
        dfs(node, [], warnings)
    return result

def sorter(table, roles, errors, warnings, preload_thread, fst_row, fst_col):
    path_index = roles.index('path') if 'path' in roles else -1
    if path_index == -1:
        errors.append("Error: 'path' role not found in roles")
        return
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
    saved = {"input": {"table": table, "roles": roles}, "output": {"errors": errors}}
    alph = generate_unique_strings(max(len(roles), len(table)))
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
                        return
                    except ValueError:
                        pass
                    match = re.search(r" -(\w+)$", name)
                    if "_" in name or ":" in name or "|" in name or (match and match.group(1) not in ["fst", "lst"]):
                        errors.append(f"Error in row {i}, column {alph[j]}: {name!r} contains invalid characters (_ : | -)")
                    match = re.search(r"(\(\d+\))$", name)
                    if match:
                        errors.append(f"Error in row {i}, column {alph[j]}: {name!r} contains invalid parentheses")
                    if name in ["fst", "lst"]:
                        errors.append(f"Error in row {i}, column {alph[j]}: {name!r} is a reserved name")
                    if name in names:
                        errors.append(f"Error in row {i}, column {alph[j]}: name {name!r} already exists in row {names[name]}")
                    names[name] = i
    if errors:
        return
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
                        return
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
                        return
                    try:
                        k = int(instr)
                        if k < 1 or k > len(table)-1:
                            errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} points to an invalid row {k}")
                            return
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
    attributes = accumulate_dependencies(attributes, warnings)
    urls = [(table[i][path_index], []) for i in range(len(table))]
    for i, row in enumerate(table[1:], start=1):
        if row[path_index] and i in attributes:
            for k in attributes[i].keys():
                if urls[k][0]:
                    path1 = ' -> '.join([f"{table[x][0].split('; ')[0]}({x})" for x in urls[i][1]])
                    path2 = ' -> '.join([f"{table[x][0].split('; ')[0]}({x})" for x in attributes[i][k][1]])
                    errors.append(f"Error in row {i}, column {alph[path_index]}: a URL given by {path1} conflicts with another given by {path2}")
                    return
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
        return

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
            while t in fst_cat:
                t = fst_cat[t]
            for i in attributes[t].keys():
                instr_table[i].append(instr_struct(True, False, [new_indexes[k]], [(-float("inf"), -1)]))
    for k, v in lst_cat.items():
        if is_valid[k]:
            t = v
            while t in lst_cat:
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
                            return
                        if len(dep_pattern[j]) > 1:
                            instr = dep_pattern[j][0] + ''.join([instr_split[i]+dep_pattern[j][i+1] for i in range(len(instr_split))])
                        match = re.match(PATTERN_DISTANCE, instr)
                        intervals = []
                        if is_constraint := not match:
                            match = re.match(PATTERN_AREAS, instr)
                            if not match:
                                errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} does not match expected format")
                                return
                            intervals = get_intervals(instr)
                        numbers = []
                        if match.group("number"):
                            number = int(match.group("number"))
                            if number == 0 or number > len(table):
                                errors.append(f"Error in row {i}, column {alph[j]}: invalid number.")
                                return
                            if table[number][path_index]:
                                numbers.append(number)
                            name = number
                        elif not (name := match.group("name")):
                            errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} does not match expected format")
                            return
                        if name in attributes:
                            for r in attributes[name].keys():
                                numbers.append(r)
                        elif match.group("name"):
                            if name not in names:
                                errors.append(f"Error in row {i}, column {alph[j]}: attribute {name!r} does not exist")
                                return
                            if table[names[name]][path_index]:
                                numbers.append(names[name])
                            if names[name] in attributes:
                                for r in attributes[names[name]].keys():
                                    numbers.append(r)
                        numbers = list(map(lambda x: new_indexes[x], numbers))
                        instr_table[i].append(instr_struct(is_constraint, match.group("any"), numbers, intervals))
    instr_table_ext = [list(x) for x in instr_table]
    for i in valid_row_indexes:
        for j in attributes_table[i]:
            if type(j) is int:
                for x2 in instr_table[j]:
                    x = copy.deepcopy(x2)
                    if x not in instr_table[i]:
                        if i == 4 and 16 in x.numbers:
                            print("h")
                        x.path = attributes[j][i][1] + x.path
                        instr_table_ext[i].append(x)
                    elif len(instr_table[i][instr_table[i].index(x)].path) == 1:
                        warnings.append(f"Redundant instruction {x!r} in row {i}, column {alph[j]} given by {' -> '.join([str(x) for x in attributes[j][i][1] + x.path])}")
    instr_table_int = []
    for i in valid_row_indexes:
        instr_table_int.append(list(set(instr_table_ext[i])))
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
                errors.append(f"Cycle detected: {(' after ' if p else ' before ').join(['->'.join([f"{table[x][0].split('; ')[0]}({x})" for x in k]) for k in stack])}")
                return
    urls = [urls[i][0] for i in valid_row_indexes]
    saved["data"] = {
        "elements": alph[:len(valid_row_indexes)],
        "instructions": instr_table,
        "urls": urls,
        "to_old_indexes": to_old_indexes,
        "cat_rows": cat_rows,
        "attributes_table": attributes_table,
        "roles": roles,
        "dep_pattern": dep_pattern,
        "path_index": path_index,
        "attributes": attributes,
        "new_indexes": new_indexes
    }
    if fst_row is not None:
        saved["fst_row"] = fst_row
        saved["fst_col"] = fst_col
    solver(saved, errors, fst_row is not None, preload_thread)


if __name__ == "__main__":
    preload_thread = threading.Thread(target=preload_ortools, daemon=True)
    preload_thread.start()
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
        if any(row[j] for row in table[:3]):
            crop_column = j + 1
            break
    fst_row = table[0]
    table = table[1:crop_line]
    fst_col = [row[0] for row in table]
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
            result = table
            break
        roles[i] = role
    else:
        result = sorter(table, roles, errors, warnings, preload_thread, fst_row, fst_col)
    if errors:
        print("Errors found:")
        for error in errors:
            print(f"- {error}")
    if warnings:
        print("Warnings found:")
        for warning in warnings:
            print(f"- {warning}")
    input("Press Enter to continue...")