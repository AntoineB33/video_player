function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("Custom Tools")
    .addItem("Find & Replace Enhanced", "showSidebar")
    .addToUi();
}

function onEdit(e) {
  PropertiesService.getScriptProperties().setProperty("wasEdited", "true");
}

function toA1Notation(row, col) {
  let colStr = "";
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
  const html = HtmlService.createHtmlOutputFromFile("Sidebar").setTitle(
    "Find & Replace Enhanced",
  );
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
      if (typeof cellValue === "string") {
        const parts = cellValue.split(";").map((s) => s.trim());
        if (parts.includes(findText)) {
          // Get A1 notation for the found cell
          const cellNotation = sheet.getRange(row + 1, col + 1).getA1Notation();
          matches.push({
            row: row + 1,
            col: col + 1,
            value: cellValue,
            a1: cellNotation,
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
      if (typeof cell === "string") {
        let parts = cell.split(";").map((s) => s.trim());
        let modified = false;
        parts = parts.map((p) => {
          if (p === findText) {
            modified = true;
            return replaceText;
          }
          return p;
        });
        if (modified) {
          values[row][col] = parts.join("; ");
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
function getCellElementsWithLinks(sheet, data, roles) {
  const cell = sheet.getActiveRange();
  if (!cell) return [];

  const value = cell.getValue().split(";").map((s) => s.trim().toLowerCase()).filter((s) => s).join("; ");

  const elements = value
    .split(";")
    .map((s) => s.trim())
    .filter((s) => s);
  const nameIndex = roles.includes(Roles.NAMES) ? roles.indexOf(Roles.NAMES) : -1;
  const result = [];
  elements.forEach((el) => {
    // 🔹 If element ends with " -fst", strip it for lookup
    const lookupValue = el.endsWith(" -fst") ? el.slice(0, -5).trim() : el;

    let linkRow = null;

    // search in "names" column starting from row 3 (below headers)
    for (let r = 2; r < data.length; r++) {
      if (
        data[r][nameIndex - 1]
          .split(";")
          .map((s) => s.trim())
          .filter((s) => s)
          .includes(lookupValue)
      ) {
        linkRow = r + 1; // 1-based row
        break;
      }
    }

    if (linkRow) {
      result.push({ text: el, row: linkRow, col: nameIndex, isLink: true });
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
  let history = JSON.parse(props.getProperty("cellHistory") || "[]");

  const newCell = {
    row: e.range.getRow(),
    col: e.range.getColumn(),
    timestamp: new Date().toISOString(),
  };
  history.push(newCell);
  if (history.length > 5) history.shift();

  props.setProperty("cellHistory", JSON.stringify(history));
}

/**
 * Returns the full history of selected cells as [{row, col, timestamp}...]
 */
function getCellHistory() {
  const props = PropertiesService.getScriptProperties();
  const history = props.getProperty("cellHistory");
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

function getRoles() {
  return JSON.parse(PropertiesService.getScriptProperties().getProperty("roles"));
}

function getEverything(roles) {
  let result = {
    categories: [],
    sprawlPairs: [],
    cellElementsWithLinks: getCellElementsWithLinks(sheet, table, roles),
    row_to_names: [],
    errors: [],
    warnings: [],
    nameIndex: -1,
    wasEdited: PropertiesService.getScriptProperties().getProperty("wasEdited") === "true",
  };
  if (result.wasEdited !== "true") {
    return result;
  }
  const sheet = SpreadsheetApp.getActiveSheet();
  const table = sheet.getDataRange().getValues();
  table.forEach((row) =>
    row.forEach((cell, idx) => {
      if (typeof cell !== "string") {
        row[idx] = String(cell);
      }
      row[idx] = row[idx].trim().toLowerCase();
    }),
  );
  normalize(sheet, table, result, roles);
  return result;
}