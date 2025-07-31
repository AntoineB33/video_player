function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Custom Tools')
    .addItem('Find & Replace Enhanced', 'showSidebar')
    .addToUi();
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
      const cell = values[row][col];
      if (typeof cell === 'string') {
        const parts = cell.split(';').map(s => s.trim());
        if (parts.includes(findText)) {
          matches.push({ row: row + 1, col: col + 1, value: cell });
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
