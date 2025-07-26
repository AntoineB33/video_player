import re
import random
from typing import List, Tuple, Set, Optional, Dict
import itertools
import string

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

class ConstraintSorter:
    def __init__(self, elements: List[str]):
        self.elements = elements
        self.n = len(elements)
        self.forbidden_constraints: List[Tuple[str, str, List[Tuple[float, float]]]] = []
        self.required_disjunctive_constraints: List[Tuple[str, List[str], List[Tuple[float, float]]]] = []
        self.maximize_distance: List[Tuple[str, str]] = []
        self.last_violations: Optional[List[str]] = None

    def add_forbidden_constraint(self, x: str, y: str, intervals: List[Tuple[int, int]]):
        """
        Add a constraint that element x cannot be placed in specified intervals around element y.
        Intervals are relative positions: [-10,-6] means x cannot be 10 to 6 positions before y.
        Use float('inf') for 'end of list' in intervals.
        """
        self.forbidden_constraints.append((x, y, [(float(s), float(e)) for s, e in intervals]))

    def add_forbidden_constraint_any_y(self, x: str, y_list: List[str], intervals: List[Tuple[int, int]]):
        """
        Adds a constraint that x's relative position to AT LEAST ONE element in y_list
        must fall OUTSIDE the specified forbidden intervals.
        """
        self.required_disjunctive_constraints.append((x, y_list, [(float(s), float(e)) for s, e in intervals]))

    def add_maximize_distance_constraint(self, x: str, y: str):
        self.maximize_distance.append((x, y))

    def add_group_maximize(self, index_set: Set[int]):
        names = [self.elements[i] for i in index_set]
        for u, v in itertools.combinations(names, 2):
            self.add_maximize_distance_constraint(u, v)

    def is_valid_placement(self, arrangement: List[Optional[str]]) -> bool:
        """Check if a partial or full arrangement satisfies all hard constraints."""
        pos: Dict[str, int] = {elem: i for i, elem in enumerate(arrangement) if elem is not None}

        # 1. Check standard forbidden constraints
        for x, y, intervals in self.forbidden_constraints:
            if x not in pos or y not in pos:
                continue
            relative_pos = pos[x] - pos[y]
            for start, end in intervals:
                if start <= relative_pos <= end:
                    return False

        # 2. Check required disjunctive constraints
        for x, y_list, intervals in self.required_disjunctive_constraints:
            if x not in pos:
                continue

            # This constraint is only active if at least one 'y' is also placed.
            placed_ys = [y_elem for y_elem in y_list if y_elem in pos]
            if not placed_ys:
                continue

            is_satisfied_for_x = False
            for y in placed_ys:
                relative_pos = pos[x] - pos[y]
                is_in_forbidden_zone = any(start <= relative_pos <= end for start, end in intervals)
                
                if not is_in_forbidden_zone:
                    is_satisfied_for_x = True
                    break
            
            if not is_satisfied_for_x:
                return False

        return True

    def calculate_distance_score(self, arrangement: List[str]) -> float:
        """Calculate score based on distance maximization constraints (higher is better)."""
        pos = {elem: i for i, elem in enumerate(arrangement)}
        total_distance = 0
        for x, y in self.maximize_distance:
            if x in pos and y in pos:
                total_distance += abs(pos[x] - pos[y])
        return total_distance

    def local_search_optimization(self, initial_arrangement: List[str], max_iterations: int = 2000) -> List[str]:
        """Improve arrangement using local search while maintaining constraint satisfaction."""
        current = initial_arrangement.copy()
        current_score = self.calculate_distance_score(current)

        for _ in range(max_iterations):
            i, j = random.sample(range(self.n), 2)
            new_arrangement = current.copy()
            new_arrangement[i], new_arrangement[j] = new_arrangement[j], new_arrangement[i]

            if self.is_valid_placement(new_arrangement):
                new_score = self.calculate_distance_score(new_arrangement)
                if new_score > current_score:
                    current = new_arrangement
                    current_score = new_score
        return current

    # Replace your old solve() method with this new one
    def solve(self, max_iterations: int = 2000) -> Optional[List[str]]:
        """
        Finds a valid arrangement using backtracking, then optimizes it for distance.
        Returns the best arrangement found, or None if no valid arrangement exists.
        """
        self.last_violations = []
        
        # Find a single valid arrangement using a systematic backtracking search
        valid_arrangement = self._solve_backtracking()

        if valid_arrangement is None:
            self.last_violations = ["No valid arrangement could be found by the backtracking solver."]
            return None

        # If a valid solution is found, optimize it for the distance score
        if self.maximize_distance:
            optimized_arrangement = self.local_search_optimization(valid_arrangement, max_iterations)
            return optimized_arrangement
        
        return valid_arrangement

    # Add this new private helper method for the backtracking logic
    def _solve_backtracking(self) -> Optional[List[str]]:
        """
        A systematic backtracking search to find one valid arrangement.
        """
        arrangement: List[Optional[str]] = [None] * self.n
        remaining_elements = set(self.elements)

        # A simple heuristic: try to place more constrained elements first.
        constrained_elements_count = {elem: 0 for elem in self.elements}
        all_constraints = (self.forbidden_constraints + self.required_disjunctive_constraints)
        for constraint in all_constraints:
            constrained_elements_count[constraint[0]] += 1
            for y in constraint[1]:
                 constrained_elements_count[y] += 1
        
        # Sort elements to place by how constrained they are, descending.
        elements_to_place = sorted(
            list(self.elements), 
            key=lambda e: constrained_elements_count[e], 
            reverse=True
        )

        return self._backtrack_recursive(arrangement, elements_to_place)

    def _backtrack_recursive(self, arrangement: List[Optional[str]], elements_to_place: List[str]) -> Optional[List[str]]:
        """The recursive core of the backtracking solver."""
        # Base case: If there are no more elements to place, we found a solution.
        if not elements_to_place:
            return [elem for elem in arrangement if elem is not None]

        element_to_try = elements_to_place[0]
        remaining = elements_to_place[1:]

        # Iterate through all available empty slots for the current element
        for i in range(self.n):
            if arrangement[i] is None:
                # Try placing the element in the current slot
                arrangement[i] = element_to_try

                # Check if this partial arrangement is still valid
                if self.is_valid_placement(arrangement):
                    # If it's valid, recurse to place the next element
                    result = self._backtrack_recursive(arrangement, remaining)
                    if result:
                        return result  # Solution found, propagate it up

                # Backtrack: Undo the placement if it didn't lead to a solution
                arrangement[i] = None
        
        # If we tried all positions for this element and none worked, this path is a dead end.
        return None

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
                    updated_cell = cell.split(';')
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
                    row[j] = ';'.join(updated_cell)
        new_table.append(row)
    return new_table

def sorter(table, roles, errors, warnings):
    alph = generate_unique_strings(max(len(roles), len(table)))
    path_index = roles.index('path') if 'path' in roles else -1
    if path_index != -1:
        for i, row in enumerate(table[1:]):
            cell = row[path_index]
            if cell:
                if not cell.strip():
                    warnings.append(f"Warning in row {i+1}, column {alph[path_index]}: only whitespace in cell")
                if not re.match(r'^(https?://|file://)', cell):
                    warnings.append(f"Warning in row {i+1}, column {alph[path_index]}: {cell!r} is not a valid URL or local path")
    pointed_by = [[] for _ in range(len(table))]
    point_to = [[] for _ in range(len(table))]
    names = [[] for _ in range(len(table))]
    for i, row in enumerate(table[1:], start=1):
        for j, cell in enumerate(row):
            if roles[j] == 'names' and cell:
                cell_list = cell.split(';')
                for name in cell_list:
                    if name not in names[i]:
                        names[i].append(name)
                    else:
                        warnings.append(f"Redundant name {name!r} in row {i}, column {alph[j]}")
    for i, row in enumerate(table[1:], start=1):
        for j, cell in enumerate(row):
            if roles[j] == 'pointers':
                if cell:
                    cell_list = cell.split(';')
                    for instr in cell_list:
                        try:
                            k = int(instr)
                            if k < 1 or k > len(table)-1:
                                errors.append(f"Error in row {i}, column {alph[j]}: {instr!r} points to an invalid row {k}")
                                return table
                            else:
                                pointed_by[k].append(i)
                                point_to[i].append(k)
                        except ValueError:
                            for ii, rrow in enumerate(names[1:], start=1):
                                if instr in rrow:
                                    pointed_by[ii].append(i)
                                    point_to[i].append(ii)
                                    break
                            else:
                                errors.append(f"Error in row {i+1}, column {alph[j]}: row {instr!r} does not exist")
                                return table
    # find a cycle
    def dfs(node, visited, stack):
        visited.add(node)
        stack.add(node)
        for neighbor in point_to[node]:
            if neighbor not in visited:
                if dfs(neighbor, visited, stack):
                    return True
            elif neighbor in stack:
                return True
        stack.remove(node)
        return False
    visited = set()
    stack = set()
    for i in range(1, len(table)):
        if i not in visited:
            if dfs(i, visited, stack):
                print(f"Cycle found starting at row {i}")
                return table
    attributes = {}
    attributes_table = [[] for _ in range(len(table))]
    for i, row in enumerate(table[1:], start=1):
        for j, cell in enumerate(row):
            if roles[j] == 'sprawl':
                if not cell:
                    continue
                cell_list = cell.split(';')
                for cat in cell_list:
                    if cat not in attributes:
                        if not cat:
                            errors.append(f"Error in row {i}, column {alph[j]}: empty attribute name")
                            return table
                        attributes[cat] = []
                    if i not in attributes[cat]:
                        attributes[cat].append(i)
                        attributes_table[i].append(cat)
                    else:
                        warnings.append(f"Redundant attribute {cat!r} in row {i}, column {alph[j]}")
    for cat in attributes:
        for i, row in enumerate(names[1:], start=1):
            if cat in row:
                errors.append(f"Error: attribute {cat!r} in row {attributes[cat][0]} conflicts with name in row {i}")
                return table
    pointed_givers = [dict() for _ in range(len(table))]
    pointed_givers_path = [0 for _ in range(len(table))]
    pointed_by_all = [list() for i in range(len(table))]
    point_to_all = [list() for i in range(len(table))]
    for i, row in enumerate(pointed_by_all[1:], start=1):
        to_check = list(pointed_by[i])
        while to_check:
            current = to_check.pop()
            if current not in row:
                row.append(current)
                point_to_all[current].append(i)
                for cat in attributes_table[i]:
                    if cat not in attributes_table[current]:
                        attributes_table[current].append(cat)
                        attributes[cat].append(current)
                        pointed_givers[current][cat] = i
                    elif cat not in pointed_givers[current]:
                        warnings.append(f"Redundant attribute {cat!r} in row {current} already given by row {i}")
                if path_index != -1 and table[i][path_index]:
                    if table[current][path_index] and not pointed_givers_path[current]:
                        warnings.append(f"Warning in row {current}, column {alph[path_index]}: path already given by row {i}")
                    table[current][path_index] = table[i][path_index]
                    pointed_givers_path[current] = i
            to_check.extend(pointed_by[current])
    valid_row_indexes = []
    new_indexes = list(range(len(table)))
    to_old_indexes = []
    staying = [False]
    cat_rows = []
    new_index = 0
    for i, row in enumerate(table[1:], start=1):
        staying.append(False)
        if path_index != -1 and row[path_index]:
            valid_row_indexes.append(i)
            staying[i] = True
            new_indexes[i] = new_index
            new_index += 1
            to_old_indexes.append(i)
        else:
            cat_rows.append(i)
    for cat in list(attributes.keys()):
        attributes[cat] = list(filter(lambda x: staying[x], attributes[cat]))
        if not attributes[cat]:
            del attributes[cat]
        elif len(attributes[cat]) == 1:
            warnings.append(f"Warning: attribute {cat!r} only in row {attributes[cat][0]}, consider removing it")
            del attributes[cat]
    for row in pointed_by_all:
        row[:] = list(filter(lambda x: staying[x], row))
    instr_table = [[] for _ in range(len(table))]
    dep_pattern = [cell.split('.') for cell in table[0]]
    for i, row in enumerate(table[1:], start=1):
        if not staying[i] and not pointed_by[i]:
            continue
        for j, cell in enumerate(row):
            if roles[j] == 'dependencies' and cell:
                cell_list = cell.split(';')
                for instr in cell_list:
                    if instr:
                        instr_split = instr.split('.')
                        if len(instr_split) != len(dep_pattern[j])-1:
                            errors.append(f"Error in row {i+1}, column {alph[j]}: {instr!r} does not match dependencies pattern {dep_pattern[j]!r}")
                            return table
                        if dep_pattern[j]:
                            instr = dep_pattern[j][0] + ''.join([instr_split[i]+dep_pattern[j][i+1] for i in range(len(instr_split))])
                        match = re.match(PATTERN_DISTANCE, instr)
                        intervals = []
                        if instr_type := not match:
                            match = re.match(PATTERN_AREAS, instr)
                            if not match:
                                errors.append(f"Error in row {i+1}, column {alph[j]}: {instr!r} does not match expected format")
                                return table
                            intervals = get_intervals(instr)
                        numbers = []
                        if match.group("number"):
                            number = int(match.group("number"))
                            if number == 0 or number > len(table):
                                errors.append(f"Error in row {i}, column {alph[j]}: invalid number.")
                                return table
                            if staying[number]:
                                numbers.append(number)
                            for pointer in pointed_by_all[number]:
                                numbers.append(pointer)
                        elif name := match.group("name"):
                            if name in attributes:
                                for r in attributes[name]:
                                    numbers.append(r)
                            else:
                                for ii, rrow in enumerate(names[1:], start=1):
                                    if name in rrow:
                                        number = ii
                                        if staying[number]:
                                            numbers.append(number)
                                        for pointer in pointed_by_all[number]:
                                            numbers.append(pointer)
                                        break
                                else:
                                    errors.append(f"Error in row {i+1}, column {alph[j]}: attribute {name!r} does not exist")
                                    return table
                        else:
                            errors.append(f"Error in row {i+1}, column {alph[j]}: {instr!r} does not match expected format")
                            return table
                        numbers = list(map(lambda x: new_indexes[x], numbers))
                        instr_table[i].append(instr_struct(instr_type, match.group("any"), numbers, intervals))
    for i in valid_row_indexes:
        for j in point_to_all[i]:
            instr_table[i] = list(set(instr_table[i] + instr_table[j]))
    # detect cycles in instr_table
    def has_cycle(instr_table, visited, stack, node, after=True):
        visited.add(node)
        stack.add(node)
        for neighbor in instr_table[node]:
            if neighbor.any and not neighbor.instr_type and neighbor.intervals[0] != (-float("inf"), -1) if after else neighbor.intervals[-1] != (1, float("inf")):
                continue
            for target in neighbor.numbers:
                if target not in visited:
                    if has_cycle(instr_table, visited, stack, target, after):
                        return True
                elif target in stack and len(stack) > 1:
                    return True
        stack.remove(node)
        return False
    visited = set()
    stack = set()
    for p in [0, 1]:
        for i in valid_row_indexes:
            if has_cycle(instr_table, visited, stack, i, p):
                errors.append(f"Cycle detected: {(' after ' if p else ' before ').join([str(to_old_indexes[i])]+[str(to_old_indexes[k]) for k in stack])}")
                return table
    instr_table_int = []
    for i in valid_row_indexes:
        instr_table_int.append(instr_table[i])
    sorter = ConstraintSorter(alph[:len(valid_row_indexes)])
    go(alph, instr_table_int, sorter)
    for cat in attributes:
        sorter.add_group_maximize(set(map(lambda x: new_indexes[x], attributes[cat])))
    
    # Solve the problem
    print("Solving constraint-based sorting problem...")
    solution = sorter.solve(max_iterations=2000)
    
    if not solution:
        errors.append("No valid solution found!")
        return table
    elif type(solution) is string:
        errors.append(f"Error when sorting: {solution!r}")
        return table
    print(f"Solution found: {solution}")
    print(f"Is valid: {sorter.is_valid_placement(solution)}")
    print(f"Distance score: {sorter.calculate_distance_score(solution)}")
    
    # Show positions for clarity
    print("\nPositions:")
    for i, elem in enumerate(solution):
        print(f"Position {i}: {elem}")

    res = [0] + [to_old_indexes[alph.index(elem)] for elem in solution]
    i = 0
    while i < len(res):
        d = 0
        while d < len(cat_rows):
            e = cat_rows[d]
            if e in point_to_all[res[i]]:
                res.insert(i, e)
                i += 1
                del cat_rows[d]
            else:
                d += 1
        i += 1
    res.extend(cat_rows)
    new_table = order_table(res, table, roles, dep_pattern)
    return new_table


if __name__ == "__main__":
    #take from clipboard
    import pyperclip
    clipboard_content = pyperclip.paste()
    table = [line.split('\t') for line in clipboard_content.split('\n')]
    for row in table:
        for j, cell in enumerate(row):
            cells = cell.split(';')
            for k, c in enumerate(cells):
                cells[k] = c.strip().lower()
            row[j] = ';'.join(cells)
    for row in table:
        row[-1] = row[-1].strip()
    roles = table[0]
    warnings = []
    errors = []
    result = sorter(table[1:], roles, errors, warnings)
    result.insert(0, roles)
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
    # input("Sorted table copied to clipboard. Press Enter to exit.")