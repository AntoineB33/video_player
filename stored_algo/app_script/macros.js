function onOpen() {
  SpreadsheetApp.getUi()
  .createMenu('Custom Tools')
  .addItem('Find & Replace Enhanced', 'showSidebar')
  .addToUi();
}

function toA1Notation(row, col) {
  let colStr = '';
  while (col > 0) {
  let remainder = (col - 1) % 26;
  colStr = String.fromCharCode(65 + remainder) + colStr; // 65 = 'A'
  col = Math.floor((col - 1) / 26);
  }
  return colStr + row;
}

function clearAllProperties() {
  PropertiesService.getScriptProperties().deleteAllProperties();
}

function showSidebar() {
  const html = HtmlService.createHtmlOutputFromFile('Sidebar')
  .setTitle('Find & Replace Enhanced');
  SpreadsheetApp.getUi().showSidebar(html);
}

function findMatches(findText) {
  const sheet = SpreadsheetApp.getActiveSheet();
  const range = sheet.getDataRange();
  const values = range.getValues();
  const matches = [];

  for (let row = 0; row < values.length; row++) {
  for (let col = 0; col < values[row].length; col++) {
    const cellValue = values[row][col];
    if (typeof cellValue === 'string') {
    const parts = cellValue.split(';').map(s => s.trim());
    if (parts.includes(findText)) {
      // Get A1 notation for the found cell
      const cellNotation = sheet.getRange(row + 1, col + 1).getA1Notation();
      matches.push({
      row: row + 1,
      col: col + 1,
      value: cellValue,
      a1: cellNotation
      });
    }
    }
  }
  }
  return matches;
}

function replaceMatches(findText, replaceText) {
  const sheet = SpreadsheetApp.getActiveSheet();
  const range = sheet.getDataRange();
  const values = range.getValues();

  for (let row = 0; row < values.length; row++) {
  for (let col = 0; col < values[row].length; col++) {
    const cell = values[row][col];
    if (typeof cell === 'string') {
    let parts = cell.split(';').map(s => s.trim());
    let modified = false;
    parts = parts.map(p => {
      if (p === findText) {
      modified = true;
      return replaceText;
      }
      return p;
    });
    if (modified) {
      values[row][col] = parts.join('; ');
    }
    }
  }
  }
  range.setValues(values);
}

/**
 * Activates a specific cell in the spreadsheet.
 * @param {number} row The row number to activate.
 * @param {number} col The column number to activate.
 */
function activateCell(row, col) {
  SpreadsheetApp.getActiveSheet().getRange(row, col).activate();
}

/**
 * Given elements from the active cell, check if they match values
 * in the "names" column (the column whose header is "names" in row 2).
 * If an element ends with " -fst", we search using the version without that suffix.
 * Returns an array of objects with { text, row, col, isLink }.
 */
function getCellElementsWithLinks(sheet, data) {
  const cell = sheet.getActiveRange();
  if (!cell) return [];

  const value = cell.getValue();
  if (typeof value !== 'string') return [];

  const elements = value.split(';').map(s => s.trim()).filter(s => s);

  // 🔹 Find "names" column in the 2nd row
  let namesCol = null;
  for (let c = 0; c < data[1].length; c++) {
  if (data[1][c] === Roles.NAMES) {
    namesCol = c + 1; // 1-based column index
    break;
  }
  }

  if (!namesCol) {
  // No "names" column found → return plain list
  return elements.map(el => ({ text: el, row: null, col: null, isLink: false }));
  }

  const result = [];
  elements.forEach(el => {
  // 🔹 If element ends with " -fst", strip it for lookup
  const lookupValue = el.endsWith(" -fst")
    ? el.slice(0, -5).trim()
    : el;

  let linkRow = null;

  // search in "names" column starting from row 3 (below headers)
  for (let r = 2; r < data.length; r++) {
    if (data[r][namesCol - 1].split(';').map(s => s.trim()).filter(s => s).includes(lookupValue)) {
    linkRow = r + 1; // 1-based row
    break;
    }
  }

  if (linkRow) {
    result.push({ text: el, row: linkRow, col: namesCol, isLink: true });
  } else {
    result.push({ text: el, row: null, col: null, isLink: false });
  }
  });

  return result;
}

// Track history of focused cells in script properties
function onSelectionChange(e) {
  if (!e || !e.range) return;

  const props = PropertiesService.getScriptProperties();
  let history = JSON.parse(props.getProperty('cellHistory') || "[]");

  const newCell = {
  row: e.range.getRow(),
  col: e.range.getColumn(),
  timestamp: new Date().toISOString()
  };
  history.push(newCell);
  if (history.length > 5) history.shift();

  props.setProperty('cellHistory', JSON.stringify(history));
}

/**
 * Returns the full history of selected cells as [{row, col, timestamp}...]
 */
function getCellHistory() {
  const props = PropertiesService.getScriptProperties();
  const history = props.getProperty('cellHistory');
  return history ? JSON.parse(history) : [];
}

function generateUniqueStrings(n) {
  const charset = "abcdefghijklmnopqrstuvwxyz"; // can expand with digits/uppercase
  const result = [];
  let length = 1;

  function generate(prefix, depth) {
  if (result.length >= n) return;
  if (depth === 0) {
    result.push(prefix);
    return;
  }
  for (let i = 0; i < charset.length; i++) {
    if (result.length >= n) break;
    generate(prefix + charset[i], depth - 1);
  }
  }

  while (result.length < n) {
  generate("", length);
  length++;
  }

  return result;
}

/*
Returns: { categories: [ [[colIndex, colName], [[category, listLength], [[rowIndex, [rowName1, ...]], ... ] ], ... ],
      sprawlPairs: [ [[colIndex, colName], [ { category, row1, row2, distance }, ... ] ], ... ],
      errors: [ "error message", ... ]
      }
*/
function getCategories(data) {
  const result = {
  categories: [],
  sprawlPairs: []
  };
  if (data.length < 3) return result;

  
  let namesCol = null;
  let pathCol = null;
  data[1].forEach((h, idx) => {
  if (h === "path") pathCol = idx;
  else if (h === "names") namesCol = idx;
  });
  
  data[1].forEach((h, idx) => {
  if (h === "sprawl" || h === "attributes") {
    const colCat = {};
    for (let r = 2; r < data.length; r++) {
    if (!data[r][pathCol]) continue;
    const categories = String(data[r][idx]).trim();
    for (let cat of categories.split(';').map(s => s.trim())) {
      if (cat) {
      if (!colCat[cat]) colCat[cat] = [];
      colCat[cat].push(r);
      }
    }
    }
    
    const colList = [];
    result.categories.push([[idx, h], colList]);
    // get the list of categories sorted by name
    Object.entries(colCat).sort((a, b) => a[0].localeCompare(b[0])).forEach(([cat, rows]) => {
    colList.push([[cat, rows.length], rows.map(r => [r, data[r][0]])]);
    });

    if (h === "sprawl") {
    const sprawlPairs = [];
    result.sprawlPairs.push([[idx, h], sprawlPairs]);

    // For each category, compute smallest distance
    Object.entries(colCat).forEach(([cat, rows]) => {
      if (rows.length < 2) return;
      let minDist = Infinity;
      let bestPair = null;

      for (let i = 0; i < rows.length; i++) {
      for (let j = i + 1; j < rows.length; j++) {
        let d = Math.abs(rows[i] - rows[j]);
        for (let k = rows[i] + 1; k < rows[j]; k++) {
        if (!String(data[k - 1][pathCol]).trim()) {
          d--; // reduce distance for each empty path in between
        }
        }
        if (d < minDist) {
        minDist = d;
        bestPair = [rows[i], rows[j]];
        }
      }
      }

      if (bestPair) {
      sprawlPairs.push({
        category: cat,
        row1: bestPair[0]+3,
        row2: bestPair[1]+3,
        distance: minDist
      });
      }
    });

    // sort sprawlPairs by distance ascending
    sprawlPairs.sort((a, b) => a.distance - b.distance);
    }
  }
  });
  if (attributesCol.length === 0) {
  return result;
  }

  const results = [];

  return results;
  const uniqueCategories = Object.entries(categoryCount)
  .filter(([cat, count]) => count === 1)
  .map(([cat, count]) => cat);
  const sharedCategories = Object.entries(categoryCount)
  .filter(([cat, count]) => count > 1)
  .map(([cat, count]) => cat);

  return { uniqueCategories, sharedCategories };
}

/**
 * Finds closest row pairs per sprawl column based on shared category and path filter.
 * Returns: [{ colName, category, row1, row2, distance }]
 */
function getClosestSprawlPairs(data) {

  if (data.length < 5) return [];

  // Find "sprawl" and "path" columns from row 2
  const headers = data[1].map(h => String(h).toLowerCase().trim());
  const sprawlCols = [];
  let pathCol = null;

  headers.forEach((h, idx) => {
  if (h.includes("sprawl")) sprawlCols.push(idx);
  if (h.includes("path")) pathCol = idx;
  });

  if (sprawlCols.length === 0 || pathCol === null) return [];

  const results = [];

  sprawlCols.forEach(sc => {
  const categoryRows = {};

  // Collect rows by category, but only keep rows with non-empty "path"
  for (let r = 3; r < data.length; r++) {
    const category = String(data[r][sc]).trim();
    const pathVal = String(data[r][pathCol]).trim();
    if (category && pathVal) {
    if (!categoryRows[category]) categoryRows[category] = [];
    categoryRows[category].push(r + 1); // 1-based row index
    }
  }

  // For each category, compute smallest distance
  Object.entries(categoryRows).forEach(([cat, rows]) => {
    if (rows.length < 3) return;
    let minDist = Infinity;
    let bestPair = null;

    for (let i = 0; i < rows.length; i++) {
    for (let j = i + 1; j < rows.length; j++) {
      let d = Math.abs(rows[i] - rows[j]);
      for (let k = rows[i] + 1; k < rows[j]; k++) {
      if (!String(data[k - 1][pathCol]).trim()) {
        d--; // reduce distance for each empty path in between
      }
      }
      if (d < minDist) {
      minDist = d;
      bestPair = [rows[i], rows[j]];
      }
    }
    }

    if (bestPair) {
    results.push({
      category: cat,
      row1: bestPair[0]+3,
      row2: bestPair[1]+3,
      distance: minDist
    });
    }
  });
  });

  return results;
}

function getEverything() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const table = sheet.getDataRange().getValues();
  table.forEach(row => row.forEach((cell, idx) => {
  if (typeof cell !== 'string') {
    row[idx] = String(cell);
  }
  row[idx] = row[idx].trim().toLowerCase();
  }));
  const cellElements = getCellElementsWithLinks(sheet, table);
  return {cellElements, ...normalize(table)};
}