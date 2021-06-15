/* global document fetch d3 screen window $*/

let pageName;

const themeDataDict = [];

let themes = [];
let cmColNames = [];

const sentenceStopWordsRegex = new RegExp(/\b(iv|p|a)\d+\s+|p\d+_*\w*\s+|\biv\b|\d{2}:\d{2}:\d{2}|speaker key:|interviewer \d*|participant \w*/, 'gi');

let firstLoading = true;
let counts;

let data;

let dragging = false;

let maxZIndex = 1;

const changedData = [];


const getData = function (page) {
  pageName = page;

  d3.select('#loading-gif')
    .style('display', 'block');

  fetch(`/get_${page}_data`)
    .then((res) => res.json().then((tableData) => {
      data = tableData;

      console.log(data);

      generateTable();

      d3.select('#loading-gif')
        .style('display', 'none');
    }));
};


const generateTable = function () {
  if (!/.*_matrix$/.test(pageName)) {
    themes = [
      'practices',
      'social',
      'study vs product',
      'system perception',
      'system use',
      'value judgements'
    ];
  } else {
    cmColNames = Object.keys(data[0]);
  }

  if (firstLoading) {
    if (/.*keywords$/.test(pageName)) {
      counts = [];

      // get counts from first entry
      for (let i = 0; i < themes.length; i++) {
        const theme = themes[i];
        const count = parseInt(data[0][theme][0], 10);
        counts.push(count);
      }

      // remove counts entry
      delete data[0];
      data.splice(0, 1);
    } else if (pageName === 'train_codes') {
      counts = data[data.length - 1].counts;

      // remove counts entry
      delete data[data.length - 1];
      data.splice(-1, 1);
    }

    /// create dict ///
    for (let i = 0; i < themes.length; i++) {
      const themeName = themes[i];
      const themeData = [];

      Object.keys(data).forEach((dataKey) => {
        const dataRow = data[dataKey];

        if (typeof dataRow[themeName][0] !== 'undefined') {
          const themeDataRow = [];
          themeDataRow.push(data[dataKey][themeName][0]);
          themeDataRow.push(data[dataKey][themeName][1]);
          if (pageName === 'keywords') {
            themeDataRow.push(data[dataKey][themeName][2]);
          }

          themeData.push(themeDataRow);
        }
      });
      themeDataDict.push({ [themeName]: themeData });
    }

    console.log(themeDataDict);
  }
  // else {
  //   if (pageName === 'predict_keywords') {
  //     counts = [];
  //
  //     for (let i = 0; i < themes.length; i++) {
  //       const theme = themes[i];
  //       const themeData = themeDataDict[i];
  //
  //       const sentences = new Set();
  //
  //       let socialCounter = 0;
  //
  //       for (let j = 0; j < themeData[theme].length; j++) {
  //         const sentence = themeData[theme][j][1][0];
  //
  //         if (theme === 'social') {
  //           console.log(`${socialCounter}: adding to ${theme}`);
  //           socialCounter++;
  //         }
  //
  //         sentences.add(sentence);
  //       }
  //
  //       if (theme === 'social') {
  //         console.log(`social has now count = ${sentences.size}`);
  //         console.log(sentences);
  //       }
  //
  //       counts.push(sentences.size);
  //     }
  //   }
  // }

  const table = d3.select('body')
    .append('table')
    .attr('class', () => {
      if (pageName === 'predict_keywords' || pageName === 'keywords'
      || /.*_matrix$/.test(pageName)) {
        return 'blue';
      } else if (pageName === 'train_keywords' || pageName === 'train_codes') {
        return 'grey';
      }
    })
    .classed('center', true);

  const thead = table.append('thead');
  const	tbody = table.append('tbody');

  if (!/.*_matrix$/.test(pageName)) {
    thead.append('tr')
      .selectAll('th')
      .data(themes)
      .enter()
      .append('th')
      .text((theme, i) => `${theme} (${counts[i].toString()})`)
      .style('width', () => (100 / themes.length).toString());
  } else {
    thead.append('tr')
      .selectAll('th')
      .data(cmColNames)
      .enter()
      .append('th')
      .text((text) => text)
      .style('width', () => (100 / data[0].length).toString());
  }

  const rows = tbody.selectAll('tr')
    .data(data)
    .enter()
    .append('tr');

  // cells
  rows.selectAll('td')
    .data((row) => {
      if (!/.*_matrix$/.test(pageName)) {
        return themes.map((theme) => ({ theme, value: row[theme] }));
      }
      return cmColNames.map((colName) => ({ colName, value: row[colName] }));
    })
    .enter()
    .append('td')
    .style('position', 'relative')
    .attr('column', (d) => {
      if (/.*_matrix$/.test(pageName)) {
        return d.colName;
      }
      return d.theme;
    })
    .append('div')
    .classed('td-div', true)
    .style('top', '0px')
    .style('left', '0px')
    .style('width', () => {
      if (!/.*_matrix$/.test(pageName)) {
        return `${screen.width / themes.length}px`;
      }
      return `${screen.width / cmColNames.length}px`;
    })
    .attr('column', (d) => {
      if (/.*_matrix$/.test(pageName)) {
        return d.colName;
      }
      return d.theme;
    })
    .append('span')
    .classed('td-text', true)
    .text((d) => {
      // keywords/codes
      if (d.value.length > 1) {
        if (pageName === 'train_codes') {
          return `${d.value[0]} (${d.value[1].length})`;
        }
        return d.value[0];
      }
    })
    .classed('td-with-sentences', (d) => {
      if (d.value.length > 1) {
        if (d.value[1].length > 0 || d.value[2].length > 0) {
          return true;
        }
        return false;
      }
    })
    .attr('column', (d) => {
      if (/.*_matrix$/.test(pageName)) {
        return d.colName;
      }
      return d.theme;
    })
    .attr('data-sentences', (d) => {
      if (d.value.length === 2) {
        if (d.value[1].length > 0) {
          const sentences = d.value[1];

          let dataString = '{"sentences": [';

          for (let i = 0; i < sentences.length; i++) {
            // JSON cleaning
            const sentence = sentences[i]
              .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
              // to-do: encoding
              .replace(/�/g, ' ')
              .replace(/\t/g, '    ')
              .replace(/\n/g, ' ')
              .trim();
            dataString += (`"${sentence}"`);

            if (i === (sentences.length - 1)) {
              dataString += ']}';
            } else {
              dataString += ', ';
            }
          }
          return dataString;
        }
      } else if (d.value.length === 3) {
        const predictSentences = d.value[1];
        const trainSentences = d.value[2];

        if (predictSentences.length > 0 || trainSentences.length > 0) {
          let dataString = '{"predictSentences": [';

          if (predictSentences.length > 0) {
            for (let i = 0; i < predictSentences.length; i++) {
              // JSON cleaning
              const sentence = predictSentences[i]
                .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                // to-do: encoding
                .replace(/�/g, ' ')
                .replace(/\t/g, '    ')
                .replace(/\n/g, ' ')
                .trim();
              dataString += (`"${sentence}"`);

              if (i === (predictSentences.length - 1)) {
                dataString += '], "trainSentences": [';
              } else {
                dataString += ', ';
              }
            }
          } else {
            dataString += '], "trainSentences": [';
          }

          if (trainSentences.length > 0) {
            for (let i = 0; i < trainSentences.length; i++) {
              // JSON cleaning
              const sentence = trainSentences[i]
                .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                // to-do: encoding
                .replace(/�/g, ' ')
                .replace(/\t/g, '    ')
                .replace(/\n/g, ' ')
                .trim();
              dataString += (`"${sentence}"`);

              if (i === (trainSentences.length - 1)) {
                dataString += ']}';
              } else {
                dataString += ', ';
              }
            }
          } else {
            dataString += ']}';
          }

          return dataString;
        }
      }
    });

    if (!/.*_matrix$/.test(pageName)) {
      // highlight cell that has been moved after reloading data
      let colored = 0;

      d3.selectAll('.td-text')
        .each(function () {
          const tdText = d3.select(this);

          for (let i = 0; i < changedData.length && colored < changedData.length; i++) {
            if (tdText.text() === changedData[i].movedText
            && tdText.attr('column') === changedData[i].targetColumn) {
              d3.select(this.parentNode.parentNode)
                .style('background-color', '#9fe5fc');
              colored++;
            }
          }
        });
    }

    d3.selectAll('td') // append tooltip to each td
      .each(function () {
        const tooltip = d3.select(this)
          .append('div')
          .classed('td-tooltip', true)
          .style('position', 'absolute')
          .style('visibility', 'hidden');

        tooltip
          .append('div')
          .classed('close-icon-wrapper', true)
          .append('img')
          .attr('src', '../static/close.svg')
          .classed('close-icon', true)
          .on('click', function () {
            tooltip
              .style('visibility', 'hidden');
            d3.select(this.parentNode.parentNode.parentNode)
              .select('.td-text')
              .classed('td-clicked', false);
          });

        tooltip
          .append('div')
          .classed('td-tooltip-inner', true)
          .append('div')
          .classed('td-tooltip-sentences', true);
      });

  const titleRect = d3.select('#table-title').node().getBoundingClientRect();

  // TO-DO: FIX
  //jQuery to add shadow to th on scroll
  $(window).scroll(() => {
    const scroll = $(window).scrollTop();
    // title margin = 26.75px
    if (scroll > titleRect.height + 26.75) {
      $('th').addClass('active');
    } else {
      $('th').removeClass('active');
    }
  });

  generateClickEvents();

  if (pageName === 'predict_keywords' || pageName === 'keywords') {
    generateDragAndDropEvents();
  }

  d3.select('#loading-gif')
      .style('display', 'none');

  const scrollBarWidth = window.innerWidth - document.documentElement.clientWidth;

  d3.select('#table-title')
    .style('padding-left', `${scrollBarWidth}px`);

  if (firstLoading) {
    addReclassifyListener();
    firstLoading = false;
  } else {
    d3.select('#re-classify-button')
      .attr('disabled', null);
  }
};


const generateClickEvents = function () {
  d3.selectAll('.td-with-sentences')
    .each(function () {
      const word = d3.select(this).text().replace(/ \(\d+\)/, '');

      d3.select(this)
        .on('click', function () {
          // hide other tooltips and change font to normal
          d3.selectAll('.td-tooltip')
            .style('visibility', 'hidden');
          d3.selectAll('.td-text')
            .classed('td-clicked', false);

          d3.select(this)
            .classed('td-clicked', true);

          const tooltip = d3.select(this.parentNode.parentNode)
            .select('.td-tooltip');

          const tooltipRect = tooltip.node().getBoundingClientRect();
          const tableRect = d3.select('table').node().getBoundingClientRect();
          const tdRect = this.parentNode.parentNode.getBoundingClientRect();

          tooltip
            .style('top', `${tdRect.height}px`)
            .style('left', () => {
              if (tdRect.x === 0) {
                return '10px';
              } else if (tdRect.x + tooltipRect.width < tableRect.width) {
                return '0px';
              }
              const cutOff = (tdRect.x + tooltipRect.width) - tableRect.width;
              return `${-(cutOff) - 10}px`;
            })
            .style('visibility', 'visible')
            .style('z-index', maxZIndex++);

          d3.selectAll('th')
            .style('z-index', maxZIndex++);

          d3.select('#table-title')
            .style('z-index', maxZIndex++);

          const tooltipSentences = tooltip
            .select('.td-tooltip-sentences');

          tooltipSentences
            .html(() => {
              if (tooltipSentences.html().length === 0) {
                if (pageName === 'keywords') {
                  // both types of sentences (predict + train)
                  const predictSentences =
                    JSON.parse(d3.select(this).attr('data-sentences')).predictSentences;
                  const trainSentences =
                    JSON.parse(d3.select(this).attr('data-sentences')).trainSentences;

                  let predictText = predictSentences.join('</br></br>');
                  let trainText = trainSentences.join('</br></br>');

                  const regex = new RegExp(`\\b${word}\\b`, 'gi');

                  predictText = predictText.replace(regex, '<span style="color: #0081eb;' +
                    ` font-weight: bold">${word}</span>`);
                  trainText = trainText.replace(regex, '<span style="font-weight: bold">' +
                    `${word}</span>`);

                  let html;

                  if (predictText.length > 0 && trainText.length > 0) {
                    html = `<div style="background-color: #ebfaff">${predictText}</div></br>` +
                      `<div style="background-color: #f2f2f2">${trainText}</div>`;
                  } else if (predictText.length > 0) {
                    html = `<div style="background-color: #ebfaff">${predictText}</div>`;
                  } else {
                    html = `<div style="background-color: #f2f2f2">${trainText}</div>`;
                  }

                  return html;
                }
                // only one type of sentences (predict/train)
                const sentences = JSON.parse(d3.select(this).attr('data-sentences')).sentences;
                let text = sentences.join('</br></br>');

                if (pageName === 'train_keywords') {
                  const regex = new RegExp(`\\b${word}\\b`, 'gi');
                  text = text.replace(regex, `<span style="font-weight: bold">${word}</span>`);
                } else if (pageName === 'train_codes') {
                  return text;
                } else {
                  const regex = new RegExp(`\\b${word}\\b`, 'gi');
                  text = text.replace(regex, '<span style="color: #0081eb; font-weight: bold">' +
                    `${word}</span>`);
                }
                return text;
              }
              return tooltipSentences.html();
            });
      });
    });
};


const generateDragAndDropEvents = function () {
  const dragStarted = function () {
    d3.selectAll('.td-tooltip')
      .style('visibility', 'hidden');
    d3.selectAll('.td-text')
      .classed('td-clicked', false);

    d3.select(this)
      .style('z-index', maxZIndex++);

    d3.selectAll('th')
      .style('z-index', maxZIndex++);

    d3.select('#table-title')
      .style('z-index', maxZIndex++);

    dragging = true;
  };

  const dragged = function (event) {
    d3.select(this)
      .style('left', `${event.x}px`)
      .style('top', `${event.y}px`);
  // d3.select(this)
  //   .style('transform', `translate(${event.x}px,${event.y}px)`);
  };

  const dragEnded = function (event) {
    if (dragging) {
      d3.select(this)
        .style('left', '0px')
        .style('top', '0px');

      const tdDiv = d3.select(this);
      const movingText = tdDiv.text();
      const movingSentences = JSON.parse(tdDiv.select('span').attr('data-sentences'));
      const movingColumn = tdDiv.attr('column');
      const targetColumn =
      d3.select(document.elementFromPoint(d3.pointer(event)[0] - window.pageXOffset,
        d3.pointer(event)[1] - window.pageYOffset)).attr('column');

      if (targetColumn !== movingColumn) {
        d3.select('table').remove();

        d3.select('#table-title')
          .style('padding-left', '0px');

        d3.select('#loading-gif')
            .style('display', 'block');

        // setTimeout to avoid freezing
        setTimeout(() => {
          updateData(movingText, movingSentences, movingColumn, targetColumn);
        }, 1);
      }

      dragging = false;
    }
  };

  d3.selectAll('.td-div')
    .call(d3.drag()
      .on('start', dragStarted)
      .on('drag', dragged)
      .on('end', dragEnded));
};

const updateData = function (movingText, movingSentences, movingColumn, targetColumn) {
  console.log(`moving ${movingText} from ${movingColumn} to ${targetColumn}`);

  d3.select('#re-classify-button')
    .attr('disabled', 'disabled');

  let movedText = movingText;

  if (!/.*_matrix$/.test(pageName)) {
    const movingColumnData = themeDataDict[themes.indexOf(movingColumn)][movingColumn];

    for (let i = 0; i < movingColumnData.length; i++) {
      const movingColumnDataRow = movingColumnData[i];

      if (movingColumnDataRow[0] === movingText) {
        const movingWord = movingText.match(/(\w+?)(?: \(\d+\))?$/)[1];
        const movingCount = parseInt(movingText.match(/\((\d+?)\)$/)[1], 10);

        const targetColumnData = themeDataDict[themes.indexOf(targetColumn)][targetColumn];

        let keywordJoined = false;

        // check if target column already contains keyword
        for (let j = 0; j < targetColumnData.length; j++) {
          const targetWord = targetColumnData[j][0].match(/(\w+?)(?: \(\d+\))?$/)[1];
          if (targetWord === movingWord) {
            // update keyword count
            const targetCount = parseInt(targetColumnData[j][0].match(/\((\d+?)\)/)[1], 10);
            movedText = `${movingWord} (${targetCount + movingCount})`;
            targetColumnData[j][0] = movedText;
            // add keyword sentences
            targetColumnData[j][1] = targetColumnData[j][1].concat(movingColumnDataRow[1]);
            targetColumnData[j][2] = targetColumnData[j][2].concat(movingColumnDataRow[2]);

            keywordJoined = true;
          }
        }

        if (!keywordJoined) {
          const themeDataRow = [];
          themeDataRow.push(movingColumnDataRow[0]);
          themeDataRow.push(movingColumnDataRow[1]);
          themeDataRow.push(movingColumnDataRow[2]);

          targetColumnData.push(themeDataRow);
        }

        delete movingColumnData[i];
        movingColumnData.splice(i, 1);

        // //update counters//
        // let targetCounter = parseInt(
        //   themeDataDict[themes.indexOf(targetColumn)][targetColumn][0][0], 10);
        // const numberOfMovingSentences = movingColumnDataRow[1].length;
        // console.log(`targetCounter before increasing = ${targetCounter}`);
        // themeDataDict[themes.indexOf(targetColumn)][targetColumn][0][0] =
        //   (targetCounter += numberOfMovingSentences).toString();
        // console.log(`targetCounter (string) after increasing =
        //   ${themeDataDict[themes.indexOf(targetColumn)][targetColumn][0][0]}`);
        // let movingCounter = parseInt(
        //   themeDataDict[themes.indexOf(movingColumn)][movingColumn][0][0], 10);
        // themeDataDict[themes.indexOf(movingColumn)][movingColumn][0][0] =
        //   (movingCounter += numberOfMovingSentences).toString();
      }
    }

    let longestThemeLength = 0;

    for (let i = 0; i < themes.length; i++) {
      // find longest theme length
      const themeLength = themeDataDict[i][themes[i]].length;
      if (themeLength > longestThemeLength) {
        longestThemeLength = themeLength;
      }

      // sort themeDataDict
      themeDataDict[i][themes[i]].sort((x, y) => {
        const xCount = parseInt(x[0].match(/(\d+)(?!.*\d)/), 10);
        const yCount = parseInt(y[0].match(/(\d+)(?!.*\d)/), 10);
        return xCount === yCount ? 0 : xCount > yCount ? -1 : 1;
      });
    }

    data = [];

    for (let i = 0; i < longestThemeLength; i++) {
      const dataRow = {};

      for (let j = 0; j < themes.length; j++) {
        const themeData = themeDataDict[j][themes[j]][i];

        if (typeof themeData !== 'undefined') {
          dataRow[themes[j]] = themeData;
        } else {
          dataRow[themes[j]] = '';
        }
      }

      data.push(dataRow);
    }

    console.log('---------themeDataDict AFTER MOVING---------');
    console.log(themeDataDict);
    console.log('---------data AFTER MOVING---------');
    console.log(data);

    changedData.push({ movedText, movingSentences, movingColumn, targetColumn });

    console.log('changedData vvvv');
    console.log(changedData);

    generateTable();
  }
};


const addReclassifyListener = function () {
  d3.select('#re-classify-button')
    .on('click', () => {
      d3.select('table').remove();

      d3.select('#table-title')
        .style('padding-left', '0px');

      d3.select('#re-classify-button')
        .attr('disabled', 'disabled');

      d3.select('#loading-gif')
          .style('display', 'block');

      fetch(`/re-classify_${pageName}`, {
        method: 'POST',
        mode: 'cors',
        cache: 'no-cache',
        credentials: 'same-origin',
        headers: {
          'Content-Type': 'application/json',
        },
        redirect: 'follow',
        referrerPolicy: 'no-referrer',
        body: JSON.stringify(changedData)
      }).then((res) => res.json().then((tableData) => {
        data = tableData;
        console.log(data);

        firstLoading = true;

        generateTable();

        d3.select('#loading-gif')
          .style('display', 'none');
      }));
    });
};

export { getData };
