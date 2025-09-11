def sorter(name, table, roles, errors, warnings, preload_thread, musics, music_col):
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
    saved = {"input": {"name": name, "table": table, "roles": roles}, "output": {"errors": errors}}
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
                if i != k:
                    instr_table[i].append(instr_struct(True, False, [new_indexes[k]], [(-float("inf"), -1)]))
    for k, v in lst_cat.items():
        if is_valid[k]:
            t = v
            while t in lst_cat:
                t = lst_cat[t]
            for i in attributes[t].keys():
                if i != k:
                    instr_table[i].append(instr_struct(True, False, [new_indexes[k]], [(1, float("inf"))]))
    if first_element is not None:
        for i in valid_row_indexes:
            if i != first_element:
                instr_table[i].append(instr_struct(True, False, [new_indexes[first_element]], [(-float("inf"), -1)]))
    if last_element is not None:
        for i in valid_row_indexes:
            if i != last_element:
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
                else:
                    idx = next((i for i, k in enumerate(stack) if k[0] == to_old_indexes[target]), None)
                    if idx:
                        # remove the idx first elements without creating a new list
                        stack[:] = stack[idx:] + [[to_old_indexes[target]]]
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
        saved["musics"] = musics
        saved["music_col"] = music_col
    solver(saved, errors, fst_row is not None, preload_thread)
