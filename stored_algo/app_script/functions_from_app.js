const PATTERN_DISTANCE = /^(?<prefix>as far as possible from )(?<any>any)?((?<number>\d+)|(?<name>.+))(?<suffix>)$/;
const PATTERN_AREAS = /^(?<prefix>.*\|)(?<any>any)?((?<number>\d+)|(?<name>.+))(?<suffix>\|.*)$/;

const Roles = Object.freeze({
    PATH: 'path',
    NAMES: 'names',
    ATTRIBUTES: 'attributes',
    DEPENDENCIES: 'dependencies',
    SPRAWL: 'sprawl',
    MUSICS: 'musics'
});

class InstrStruct {
    constructor(isConstraint, any, numbers, intervals, path = []) {
        this.isConstraint = isConstraint;
        this.any = any;
        this.numbers = numbers;
        this.intervals = intervals;
        this.path = path;
    }
    
    equals(other) {
        if (!(other instanceof InstrStruct)) {
            return false;
        }
        return (this.isConstraint === other.isConstraint &&
                this.any === other.any &&
                JSON.stringify([...this.numbers].sort()) === JSON.stringify([...other.numbers].sort()) &&
                JSON.stringify([...this.intervals].sort()) === JSON.stringify([...other.intervals].sort()));
    }
    
    // For use in Set operations and comparisons
    toString() {
        return JSON.stringify({
            isConstraint: this.isConstraint,
            any: this.any,
            numbers: [...this.numbers].sort(),
            intervals: [...this.intervals].sort()
        });
    }
}

function getIntervals(intervalStr) {
    // First, parse the positions of intervals
    const intervals = [[], []];
    const negPos = intervalStr.split('|');
    let positive = 0;
    
    for (const negPosPart of [negPos[0], negPos[2]]) {
        const parts = negPosPart.split('_');
        for (const part of parts) {
            if (!part) {
                intervals[positive].push([null, null]);
            } else if (part.includes(':')) {
                const [startStr, endStr] = part.split(':');
                let start, end;
                
                try {
                    start = parseInt(startStr);
                    if (isNaN(start)) throw new Error();
                } catch {
                    start = Infinity;
                }
                
                try {
                    end = parseInt(endStr);
                    if (isNaN(end)) throw new Error();
                } catch {
                    end = Infinity;
                }
                
                if (!positive) {
                    start = -start;
                    end = -end;
                }
                intervals[positive].push([start, end]);
            } else {
                const num = parseInt(part);
                intervals[positive].push([num, num]);
            }
        }
        positive = 1;
    }
    
    // Now calculate underscore intervals
    const result = [];
    positive = 0;
    
    for (const negPosPart of intervals) {
        for (let i = 0; i < negPosPart.length - 1; i++) {
            let endOfCurrent = negPosPart[i][1];
            let startOfNext = negPosPart[i + 1][0];
            
            if (endOfCurrent === null) {
                if (!positive) {
                    endOfCurrent = -Infinity;
                } else if (result.length > 0 && result[result.length - 1][1] === -1) {
                    endOfCurrent = result[result.length - 1][0] - 1;
                    result.pop();
                } else {
                    endOfCurrent = 0;
                }
            }
            
            if (startOfNext === null) {
                if (!positive) {
                    startOfNext = 0;
                } else {
                    startOfNext = Infinity;
                }
            }
            
            if (startOfNext - endOfCurrent <= 1) {
                throw new Error("Invalid interval: overlapping or adjacent intervals found.");
            }
            
            result.push([endOfCurrent + 1, startOfNext - 1]);
        }
        positive = 1;
    }
    
    return result;
}

function generateUniqueStrings(n) {
    const charset = 'abcdefghijklmnopqrstuvwxyz'; // lowercase letters
    const result = [];
    let length = 1;
    
    while (result.length < n) {
        // Generate all combinations of given length
        function generateCombinations(currentCombo, remainingLength) {
            if (remainingLength === 0) {
                result.push(currentCombo);
                return result.length >= n;
            }
            
            for (const char of charset) {
                if (generateCombinations(currentCombo + char, remainingLength - 1)) {
                    return true;
                }
            }
            return false;
        }
        
        generateCombinations('', length);
        length++;
    }
    
    return result.slice(0, n);
}

function accumulateDependencies(graph, warnings) {
    const result = {};
    
    function dfs(node, path, warnings) {
        if (path.includes(node)) {
            const cycleStart = path.indexOf(node);
            const cycle = path.slice(cycleStart).concat([node]);
            throw new Error(`Cycle detected: ${cycle.join(' -> ')}`);
        }
        
        if (node in result) {
            return result[node];
        }
        
        path.push(node);
        const accumulated = { ...graph[node] };
        
        for (const neighbor of Object.keys(graph[node])) {
            if (!(neighbor in graph)) {
                continue;
            }
            
            const res = dfs(neighbor, [...path], warnings);
            for (const [key, value] of Object.entries(res)) {
                accumulated[key] = [value[0], [...value[1], node]];
                if (key in graph[node]) {
                    warnings.push(`Warning: ${JSON.stringify(key)} has a redundant dependency ${JSON.stringify(node)} given by ${accumulated[key][1].join(' -> ')}`);
                }
            }
        }
        
        path.pop();
        result[node] = accumulated;
        return accumulated;
    }
    
    for (const node of Object.keys(graph)) {
        dfs(node, [], warnings);
    }
    
    return result;
}

function getCategories(table, result, musics, music_col, roles, name) {
    const pathIndex = roles.includes('path') ? roles.indexOf('path') : -1;
    const errors = result.errors;
    const warnings = result.warnings;
    if (pathIndex === -1) {
        errors.push("Error: 'path' role not found in roles");
        return result;
    }
    
    for (let i = 0; i < table.length; i++) {
        for (let j = 0; j < table[i].length; j++) {
            let cells = table[i][j].split(';');
            for (let k = 0; k < cells.length; k++) {
                cells[k] = cells[k].trim();
                if (j !== pathIndex) {
                    cells[k] = cells[k].toLowerCase();
                    if (roles[j] === 'attributes' || roles[j] === 'sprawl') {
                        if (cells[k].endsWith("-fst")) {
                            cells[k] = cells[k].slice(0, -4).trim() + " -fst";
                        } else if (cells[k].endsWith("-lst")) {
                            cells[k] = cells[k].slice(0, -4).trim() + " -lst";
                        }
                    }
                }
            }
            table[i][j] = cells.join('; ');
        }
    }
    
    const saved = {
        input: { name: name, table: table, roles: roles },
        output: { errors: errors }
    };
    
    const alph = generateUniqueStrings(Math.max(roles.length, table.length));
    
    for (let i = 0; i < table.length - 1; i++) {
        const row = table[i + 1];
        const cell = row[pathIndex];
        if (cell) {
            if (!/^(https?:\/\/|file:\/\/)/.test(cell)) {
                warnings.push(`Warning in row ${i + 1}, column ${alph[pathIndex]}: ${JSON.stringify(cell)} is not a valid URL or local path`);
            }
        }
    }
    
    const names = {};
    for (let i = 1; i < table.length; i++) {
        const row = table[i];
        for (let j = 0; j < row.length; j++) {
            if (roles[j] === 'names' && row[j]) {
                const cellList = row[j].split('; ');
                for (const name of cellList) {
                    if (!isNaN(parseInt(name))) {
                        errors.push(`Error in row ${i}, column ${alph[j]}: ${JSON.stringify(name)} is not a valid name`);
                        return result;
                    }
                    
                    const match = name.match(/ -(\w+)$/);
                    if (name.includes("_") || name.includes(":") || name.includes("|") || 
                        (match && !["fst", "lst"].includes(match[1]))) {
                        errors.push(`Error in row ${i}, column ${alph[j]}: ${JSON.stringify(name)} contains invalid characters (_ : | -)`);
                    }
                    
                    const parenMatch = name.match(/(\(\d+\))$/);
                    if (parenMatch) {
                        errors.push(`Error in row ${i}, column ${alph[j]}: ${JSON.stringify(name)} contains invalid parentheses`);
                    }
                    
                    if (["fst", "lst"].includes(name)) {
                        errors.push(`Error in row ${i}, column ${alph[j]}: ${JSON.stringify(name)} is a reserved name`);
                    }
                    
                    if (name in names) {
                        errors.push(`Error in row ${i}, column ${alph[j]}: name ${JSON.stringify(name)} already exists in row ${names[name]}`);
                    }
                    names[name] = i;
                }
            }
        }
    }
    
    if (errors.length > 0) {
        return result;
    }
    
    let attributes = {};
    let firstElement = null;
    let lastElement = null;
    const fstCat = {};
    const lstCat = {};
    
    for (let i = 1; i < table.length; i++) {
        const row = table[i];
        for (let j = 0; j < row.length; j++) {
            const isSprawl = roles[j] === 'sprawl';
            if (roles[j] === 'attributes' || isSprawl) {
                if (!row[j]) {
                    continue;
                }
                const newCellList = [];
                const cellList = row[j].split('; ');
                for (let instr of cellList) {
                    if (!instr) {
                        errors.push(`Error in row ${i}, column ${alph[j]}: empty attribute name`);
                        return result;
                    }
                    
                    let isFst = instr.endsWith("-fst");
                    let isLst = false;
                    
                    if (isFst) {
                        instr = instr.slice(0, -5);
                    } else if (instr === "fst") {
                        firstElement = i;
                    } else if (isLst = instr.endsWith("-lst")) {
                        instr = instr.slice(0, -5);
                    } else if (instr === "lst") {
                        lastElement = i;
                    } else if (instr.includes("-fst")) {
                        errors.push(`Error in row ${i}, column ${alph[j]}: '-fst' is not at the end of ${JSON.stringify(instr)}`);
                        return result;
                    }
                    
                    let k;
                    const numK = parseInt(instr);
                    if (!isNaN(numK)) {
                        if (numK < 1 || numK > table.length - 1) {
                            errors.push(`Error in row ${i}, column ${alph[j]}: ${JSON.stringify(instr)} points to an invalid row ${numK}`);
                            return result;
                        }
                        k = numK;
                    } else {
                        k = -1;
                        if (instr in names) {
                            k = names[instr];
                        }
                    }
                    
                    if (k === -1) {
                        k = instr;
                    }
                    
                    if (!(k in attributes)) {
                        attributes[k] = {};
                    }
                    
                    if (!(i in attributes[k])) {
                        attributes[k][i] = [isSprawl, [k]];
                    } else {
                        warnings.push(`Redundant attribute ${JSON.stringify(instr)} in row ${i}, column ${alph[j]}`);
                    }
                    
                    if (isFst) {
                        fstCat[i] = k;
                    } else if (isLst) {
                        lstCat[i] = k;
                    }
                    
                    newCellList.push(instr);
                }
                row[j] = newCellList.join('; ');
            }
        }
    }
    
    attributes = accumulateDependencies(attributes, warnings);
    
    let urls = table.map((_, i) => [table[i][pathIndex], []]);
    
    for (let i = 1; i < table.length; i++) {
        const row = table[i];
        if (row[pathIndex] && i in attributes) {
            for (const k of Object.keys(attributes[i])) {
                if (urls[k][0]) {
                    const path1 = urls[i][1].map(x => `${table[x][0].split('; ')[0]}(${x})`).join(' -> ');
                    const path2 = attributes[i][k][1].map(x => `${table[x][0].split('; ')[0]}(${x})`).join(' -> ');
                    errors.push(`Error in row ${i}, column ${alph[pathIndex]}: a URL given by ${path1} conflicts with another given by ${path2}`);
                    return result;
                }
                urls[k] = [row[pathIndex], attributes[i][k][1]];
            }
        }
    }
    
    const validRowIndexes = [];
    const isValid = new Array(table.length).fill(false);
    const newIndexes = Array.from({ length: table.length }, (_, i) => i);
    const toOldIndexes = [];
    const catRows = [];
    let newIndex = 0;
    
    for (let i = 1; i < table.length; i++) {
        const row = table[i];
        if (row[pathIndex]) {
            isValid[i] = true;
            validRowIndexes.push(i);
            newIndexes[i] = newIndex;
            newIndex++;
            toOldIndexes.push(i);
        } else {
            catRows.push(i);
        }
    }
    
    if (validRowIndexes.length === 0) {
        errors.push("Error: No valid rows found in the table!");
        return result;
    }
    
    for (const cat of Object.keys(attributes)) {
        if (typeof cat !== 'string' || isNaN(parseInt(cat))) {
            continue;
        }
        const catInt = parseInt(cat);
        const filtered = {};
        for (const [k, v] of Object.entries(attributes[catInt])) {
            if (table[k][pathIndex]) {
                filtered[k] = v;
            }
        }
        attributes[catInt] = filtered;
        if (Object.keys(attributes[catInt]).length === 0) {
            delete attributes[catInt];
        }
    }
    
    const attributesTable = Array.from({ length: table.length }, () => new Set());
    for (const [attr, deps] of Object.entries(attributes)) {
        for (const dep of Object.keys(deps)) {
            attributesTable[dep].add(attr);
        }
    }
    
    const instrTable = Array.from({ length: table.length }, () => []);
    
    for (const [k, v] of Object.entries(fstCat)) {
        const kInt = parseInt(k);
        if (isValid[kInt]) {
            let t = v;
            while (t in fstCat) {
                t = fstCat[t];
            }
            for (const i of Object.keys(attributes[t])) {
                if (parseInt(i) !== kInt) {
                    instrTable[i].push(instrStruct(true, false, [newIndexes[kInt]], [[-Infinity, -1]]));
                }
            }
        }
    }
    
    for (const [k, v] of Object.entries(lstCat)) {
        const kInt = parseInt(k);
        if (isValid[kInt]) {
            let t = v;
            while (t in lstCat) {
                t = lstCat[t];
            }
            for (const i of Object.keys(attributes[t])) {
                if (parseInt(i) !== kInt) {
                    instrTable[i].push(instrStruct(true, false, [newIndexes[kInt]], [[1, Infinity]]));
                }
            }
        }
    }
    
    if (firstElement !== null) {
        for (const i of validRowIndexes) {
            if (i !== firstElement) {
                instrTable[i].push(instrStruct(true, false, [newIndexes[firstElement]], [[-Infinity, -1]]));
            }
        }
    }
    
    if (lastElement !== null) {
        for (const i of validRowIndexes) {
            if (i !== lastElement) {
                instrTable[i].push(instrStruct(true, false, [newIndexes[lastElement]], [[1, Infinity]]));
            }
        }
    }
    
    const depPattern = table[0].map(cell => cell.split('.'));
    
    for (let i = 1; i < table.length; i++) {
        if (!table[i][pathIndex] && !(i in attributes)) {
            continue;
        }
        
        const row = table[i];
        for (let j = 0; j < row.length; j++) {
            if (roles[j] === 'dependencies' && row[j]) {
                const cellList = row[j].split('; ');
                for (let instr of cellList) {
                    if (instr) {
                        const instrSplit = instr.split('.');
                        if (instrSplit.length !== depPattern[j].length - 1 && depPattern[j].length > 1) {
                            errors.push(`Error in row ${i}, column ${alph[j]}: ${JSON.stringify(instr)} does not match dependencies pattern ${JSON.stringify(depPattern[j])}`);
                            return result;
                        }
                        
                        if (depPattern[j].length > 1) {
                            instr = depPattern[j][0] + instrSplit.map((split, idx) => 
                                split + depPattern[j][idx + 1]).join('');
                        }
                        
                        let match = instr.match(PATTERN_DISTANCE);
                        let intervals = [];
                        let isConstraint = !match;
                        
                        if (isConstraint) {
                            match = instr.match(PATTERN_AREAS);
                            if (!match) {
                                errors.push(`Error in row ${i}, column ${alph[j]}: ${JSON.stringify(instr)} does not match expected format`);
                                return result;
                            }
                            intervals = getIntervals(instr);
                        }
                        
                        const numbers = [];
                        let name;
                        
                        if (match.groups && match.groups.number) {
                            const number = parseInt(match.groups.number);
                            if (number === 0 || number > table.length) {
                                errors.push(`Error in row ${i}, column ${alph[j]}: invalid number.`);
                                return result;
                            }
                            if (table[number][pathIndex]) {
                                numbers.push(number);
                            }
                            name = number;
                        } else if (!(name = match.groups && match.groups.name)) {
                            errors.push(`Error in row ${i}, column ${alph[j]}: ${JSON.stringify(instr)} does not match expected format`);
                            return result;
                        }
                        
                        if (name in attributes) {
                            for (const r of Object.keys(attributes[name])) {
                                numbers.push(parseInt(r));
                            }
                        } else if (match.groups && match.groups.name) {
                            if (!(name in names)) {
                                errors.push(`Error in row ${i}, column ${alph[j]}: attribute ${JSON.stringify(name)} does not exist`);
                                return result;
                            }
                            if (table[names[name]][pathIndex]) {
                                numbers.push(names[name]);
                            }
                            if (names[name] in attributes) {
                                for (const r of Object.keys(attributes[names[name]])) {
                                    numbers.push(parseInt(r));
                                }
                            }
                        }
                        
                        const mappedNumbers = numbers.map(x => newIndexes[x]);
                        instrTable[i].push(instrStruct(isConstraint, match.groups && match.groups.any, mappedNumbers, intervals));
                    }
                }
            }
        }
    }
    
    const instrTableExt = instrTable.map(x => [...x]);
    
    for (const i of validRowIndexes) {
        for (const j of attributesTable[i]) {
            if (typeof j === 'string' && !isNaN(parseInt(j))) {
                const jInt = parseInt(j);
                for (const x2 of instrTable[jInt]) {
                    const x = JSON.parse(JSON.stringify(x2)); // deep copy
                    if (!instrTable[i].some(instr => JSON.stringify(instr) === JSON.stringify(x))) {
                        x.path = [...attributes[jInt][i][1], ...x.path];
                        instrTableExt[i].push(x);
                    } else {
                        const existingIndex = instrTable[i].findIndex(instr => JSON.stringify(instr) === JSON.stringify(x));
                        if (instrTable[i][existingIndex].path.length === 1) {
                            const pathStr = [...attributes[jInt][i][1], ...x.path].join(' -> ');
                            warnings.push(`Redundant instruction ${JSON.stringify(x)} in row ${i}, column ${alph[j]} given by ${pathStr}`);
                        }
                    }
                }
            }
        }
    }
    
    const instrTableInt = [];
    for (const i of validRowIndexes) {
        // Remove duplicates by converting to Set (approximate)
        const unique = [];
        const seen = new Set();
        for (const item of instrTableExt[i]) {
            const key = JSON.stringify(item);
            if (!seen.has(key)) {
                seen.add(key);
                unique.push(item);
            }
        }
        instrTableInt.push(unique);
    }
    
    // Detect cycles in instrTable
    function hasCycle(instrTable, visited, stack, node, after = true) {
        stack.push([toOldIndexes[node]]);
        visited.add(node);
        
        for (const neighbor of instrTable[node]) {
            if (neighbor.any || !neighbor.isConstraint || 
                (after ? neighbor.intervals[0][0] !== -Infinity || neighbor.intervals[0][1] !== -1 : 
                 neighbor.intervals[neighbor.intervals.length - 1][0] !== 1 || neighbor.intervals[neighbor.intervals.length - 1][1] !== Infinity)) {
                continue;
            }
            
            for (const target of neighbor.numbers) {
                stack[stack.length - 1].splice(1, Infinity, ...neighbor.path);
                if (!visited.has(target)) {
                    if (hasCycle(instrTable, visited, stack, target, after)) {
                        return true;
                    }
                } else {
                    const idx = stack.findIndex(k => k[0] === toOldIndexes[target]);
                    if (idx !== -1) {
                        stack.splice(0, idx);
                        stack.push([toOldIndexes[target]]);
                        return true;
                    }
                }
            }
        }
        stack.pop();
        return false;
    }
    
    for (let p = 0; p <= 1; p++) {
        const visited = new Set();
        const stack = [];
        for (let i = 0; i < instrTableInt.length; i++) {
            if (hasCycle(instrTableInt, visited, stack, i, p === 1)) {
                const cycleStr = stack.map(k => 
                    k.map(x => `${table[x][0].split('; ')[0]}(${x})`).join('->')
                ).join(p === 1 ? ' after ' : ' before ');
                errors.push(`Cycle detected: ${cycleStr}`);
                return result;
            }
        }
    }
    
    urls = validRowIndexes.map(i => urls[i][0]);
    
    saved.data = {
        elements: alph.slice(0, validRowIndexes.length),
        instructions: instrTableInt,
        urls: urls,
        toOldIndexes: toOldIndexes,
        catRows: catRows,
        attributesTable: attributesTable,
        roles: roles,
        depPattern: depPattern,
        pathIndex: pathIndex,
        attributes: attributes,
        newIndexes: newIndexes
    };
    
    // Note: fstRow is not defined in the original code, assuming it should be firstElement
    if (firstElement !== null) {
        saved.musics = musics;
        saved.musicCol = music_col;
    }
    
    // TODO: solve sorting pb

    return result;
}

function normalize(table) {
  const result = {
    categories: [],
    sprawlPairs: [],
    errors: [],
    warnings: []
  };
  const crop_line = table.length;
  const crop_column = table[0].length;
  for (let r = crop_line - 1; r >= 0; r--) {
    if (table[r].some(cell => cell !== "")) {
      break;
    }
    crop_line--;
  }
  for (let c = crop_column - 1; c >= 0; c--) {
    if (table[1][c] !== "") {
      break;
    }
    crop_column--;
  }
  table = table.slice(0, crop_line).map(row => row.slice(0, crop_column));
  const roles = table[1];
  table.splice(1, 1);
  let music_col = -1;
  const name = sheet.getName();
    if (name === "s") {
        errors.push(`Error: Invalid role '${role}' - 's' is a reserved name`);
        foundError = true;
        return result;
    }
  let foundError = false;
  for (let i = 0; i < roles.length; i++) {
      let role = roles[i];
      
      // Check if role is valid (assuming Roles is available as an object with values)
      const validRoles = Object.values(Roles);
      if (!validRoles.includes(role)) {
          errors.push(`Error: Invalid role '${role}' found in roles`);
          foundError = true;
          break;
      }
      
      if (role === Roles.NAMES) {
          name = table[0][i];
      } else if (role === Roles.MUSICS) {
          music_col = i;
      }
      
      roles[i] = role;
  }

  if (!foundError) {
    if (music_col !== -1) {
      roles = [...roles.slice(0, music_col), ...roles.slice(music_col + 1)];
      table = table.map(row => [...row.slice(0, music_col), ...row.slice(music_col + 1)]);
    }
    getCategories(table, result, roles, name, music_col);
    return result;
  }
  return result;
}