/* global document fetch d3 screen window reclassifyBackend $*/
const tableLib = (function () {
  let page;

  let themeDataDict = [];
  let oldThemeDataDict = [];
  let oldThemeDataDictSaved = false;

  let themes = [];
  let cmColNames = [];

  const sentenceStopWordsRegex = new RegExp(/\b(iv|p|a)\d+\s+|p\d+_*\w*\s+|\biv\b|\d{2}:\d{2}:\d{2}|speaker key:|interviewer \d*|participant \w*/, 'gi');

  let firstLoading = true;
  let reclassifyCount = 0;

  let counts;

  let data;

  let dragging = false;

  let maxZIndex = 1;

  let changedData = [];
  let reclassifyChangesDict = []; // only updated to latest classification
                                  // (for comparison/visualisation purpose)

  const loadTable = function (pageName, tableData) {
    const startTime = new Date().getTime();

    page = pageName;
    data = tableData;

    d3.select('#loading-gif')
      .style('display', 'none');

    generateTable();

    d3.select('#loading-gif')
      .style('display', 'none');

    const endTime = new Date().getTime();
    console.log(`Table (JavaScript) => ${((endTime - startTime) / 1000).toFixed(2)} seconds`);
  };


  const loadReclassifiedTable = function (reclassifiedData) {
    data = reclassifiedData;
    // console.log(data);

    firstLoading = true;

    oldThemeDataDictSaved = false;

    reclassifyCount++;

    changedData = [];
    
    d3.select('#loading-text')
      .text('Updating table...');

    // timeout needed to change loading text
    setTimeout(() => {
      generateTable();

      d3.select('#loading-gif')
        .style('display', 'none');

      d3.select('#loading-text')
        .style('display', 'none');
    }, 1);
  };


  const generateTable = function () {
    if (!/.*_matrix$/.test(page)) {
      themes = [
        'practices',
        'social',
        'study vs product',
        'system perception',
        'system use',
        'value judgements'
      ];
    } else {
      // TO-DO: not working anymore
      cmColNames = Object.keys(data[0]);
    }

    if (firstLoading) {
      if (/.*keywords$/.test(page)) {
        counts = [];

        // get counts from first entry
        for (let i = 0; i < themes.length; i++) {
          const count = parseInt(data[0][i], 10);
          counts.push(count);
        }

        // remove counts entry
        delete data[0];
        data.splice(0, 1);
      } else if (page === 'train_codes') {
        counts = data[data.length - 1].counts;

        // remove counts entry
        delete data[data.length - 1];
        data.splice(-1, 1);
      }

      // reset reclassify changes
      if (reclassifyChangesDict.length > 0) {
        reclassifyChangesDict = [];
      }

      themeDataDict = [];

      for (let i = 0; i < themes.length; i++) {
        const theme = themes[i];
        const themeData = [];

        for (let j = 0; j < data.length; j++) {
          const dataRow = data[j];

          if (typeof dataRow[theme][0] !== 'undefined') {
            const themeDataRow = [];
            themeDataRow.push(dataRow[theme][0]);
            themeDataRow.push(dataRow[theme][1]);
            if (page === 'keywords') {
              themeDataRow.push(dataRow[theme][2]);
            }
            themeData.push(themeDataRow);
          }
        }
        themeDataDict.push({ [theme]: themeData });
      }

      // create classification dict if it does not exist yet (empty)
      if (oldThemeDataDict.length > 0) {
      // find reclassify differences (to highlight later)
        // console.log('CALCULATING DIFFERENCES......');
        for (let i = 0; i < themeDataDict.length; i++) {
          const themeDataRow = themeDataDict[i][themes[i]];

          const themeWords = [];

          for (let j = 0; j < themeDataRow.length; j++) {
            const themeDataEntry = themeDataRow[j];

            const reclassifiedWord = themeDataEntry[0].match(/(\w+?)(?: \(\d+\))?$/)[1];
            const reclassifiedCount =
              parseInt(themeDataEntry[0].match(/\((\d+?)\)/)[1], 10);

            let matched = false;

            for (let l = 0; l < oldThemeDataDict[i][themes[i]].length; l++) {
              const oldWord =
                oldThemeDataDict[i][themes[i]][l][0].match(/(\w+?)(?: \(\d+\))?$/)[1];
              const oldCount =
                parseInt(oldThemeDataDict[i][themes[i]][l][0].match(/\((\d+?)\)/)[1], 10);

              if (oldWord === reclassifiedWord) {
                matched = true;
                if (Math.abs(reclassifiedCount - oldCount) > (0.5 * oldCount)) {
                  themeWords.push(reclassifiedWord);
                  break;
                }
              }
            }
            if (!matched) {
              // if not matched, word is new
              themeWords.push(reclassifiedWord);
            }
          }
          reclassifyChangesDict.push({ [themes[i]]: themeWords });
        }
        // console.log('DONE!');
      }

      // console.log('oldThemeDataDict vvvvv');
      // console.log(oldThemeDataDict);
      // console.log('themeDataDict vvvvvvv');
      // console.log(themeDataDict);
      // console.log('reclassifyChangesDict vvvvvvvvvv');
      // console.log(reclassifyChangesDict);
    }
    // else {
    //   if (page === 'predict_keywords') {
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

    const table = d3.select('#table-container')
      .append('table')
      .attr('class', () => {
        if (page === 'predict_keywords' || /.*_matrix$/.test(page)) {
          return 'blue';
        } else if (page === 'keywords') {
          return 'green';
        } else if (page === 'train_keywords' || page === 'train_codes') {
          return 'grey';
        }
      })
      .classed('center', true);

    const thead = table.append('thead');
    const	tbody = table.append('tbody');

    if (!/.*_matrix$/.test(page)) {
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
        if (!/.*_matrix$/.test(page)) {
          return themes.map((theme) => ({ theme, value: row[theme] }));
        }
        return cmColNames.map((colName) => ({ colName, value: row[colName] }));
      })
      .enter()
      .append('td')
      .style('position', 'relative')
      .attr('column', (d) => {
        if (/.*_matrix$/.test(page)) {
          return d.colName;
        }
        return d.theme;
      })
      .append('div')
      .classed('td-div', true)
      .style('top', '0px')
      .style('left', '0px')
      .style('width', () => {
        if (!/.*_matrix$/.test(page)) {
          return `${screen.width / themes.length}px`;
        }
        return `${screen.width / cmColNames.length}px`;
      })
      .attr('column', (d) => {
        if (/.*_matrix$/.test(page)) {
          return d.colName;
        }
        return d.theme;
      })
      .append('span')
      .classed('td-text', true)
      .text((d) => {
        // keywords/codes
        if (d.value.length > 1) {
          if (page === 'train_codes') {
            return `${d.value[0]} (${d.value[1].length})`;
          }
          return d.value[0];
        }
      })
      .classed('td-with-sentences', (d) => {
        if (d.value.length > 1) {
          if (d.value[1].length > 0 
            || (d.value[2] !== undefined && d.value[2].length > 0)) {
            return true;
          }
          return false;
        }
      })
      .attr('column', (d) => {
        if (/.*_matrix$/.test(page)) {
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
                .replace(/\t/g, '\\\\t').replace(/\n/g, '\\\\n');
                // .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                // // to-do: encoding
                // .replace(/�/g, ' ')
                // .replace(/\t/g, '    ')
                // .replace(/\n/g, ' ')
                // .trim();
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
                  .replace(/\t/g, '\\\\t').replace(/\n/g, '\\\\n');
                  // .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                  // // to-do: encoding
                  // .replace(/�/g, ' ')
                  // .replace(/\t/g, '    ')
                  // .replace(/\n/g, ' ')
                  // .trim();
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
                  .replace(/\t/g, '\\\\t').replace(/\n/g, '\\\\n');
                  // .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                  // // to-do: encoding
                  // .replace(/�/g, ' ')
                  // .replace(/\t/g, '    ')
                  // .replace(/\n/g, ' ')
                  // .trim();
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

      if (!/.*_matrix$/.test(page)) {
        // highlight cells that have been modified after reloading data
        d3.selectAll('.td-text')
          .each(function () {
            const tdText = d3.select(this);
            if (tdText.text().length > 0) {
              const tdTheme = tdText.attr('column');
              const tdWord = tdText.text().match(/(\w+?)(?: \(\d+\))?$/)[1];
              const tdSentences = JSON.parse(tdText.attr('data-sentences'));

              let moved = false;
              let reclassified = false;

              for (let i = 0; i < changedData.length; i++) {
                const changedDataRow = changedData[i];
                let movedSentence;

                if (changedDataRow.movedText === null) { // check moved single sentences
                  if (changedDataRow.movingSentences.predictSentences.length > 0) {
                    movedSentence = changedDataRow.movingSentences.predictSentences[0];
                  } else {
                    movedSentence = changedDataRow.movingSentences.trainSentences[0];
                  }
                }

                if (tdTheme === changedDataRow.targetColumn                 //
                  && (tdText.text() === changedDataRow.movedText            // check if keyword was moved
                    || tdSentences.predictSentences.includes(movedSentence)    //
                    || tdSentences.trainSentences.includes(movedSentence))) {  // check if sentence was moved
                  moved = true;
                  break;
                }
              }

              if (reclassifyChangesDict.length > 0) {
                // check if word was reclassified
                if (reclassifyChangesDict[themes.indexOf(tdTheme)][tdTheme].indexOf(tdWord) > -1) {
                  reclassified = true;
                }
              }

              if (moved || reclassified) { // color background if moved or reclassified
                d3.select(this.parentNode.parentNode)
                  .style('border', () => {
                    if (page === 'predict_keywords') {
                      return '4px solid #69b7d1';
                    } else if (page === 'keywords') {
                      return '4px solid #6bcf83';
                    }
                  });
              }
            }
          });
      }

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

    if (page === 'predict_keywords' || page === 'keywords') {
      d3.selectAll('.td-div')
        .call(d3.drag()
          .subject(function () {
            const t = d3.select(this);
            return { x: t.attr('x'), y: t.attr('y') };
          })
          .on('start', keywordDragStarted)
          .on('drag', keywordDragged)
          .on('end', keywordDragEnded));
    }

    d3.select('#loading-gif')
      .style('display', 'none');

    d3.select('#loading-text')
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

    d3.select('#bin-div')
      .style('visibility', 'visible')
      .style('z-index', maxZIndex);
  };


  const keywordDragStarted = function () {
    d3.selectAll('.td-tooltip')
      .remove();

    d3.selectAll('.td-text')
      .classed('td-clicked', false);

    d3.selectAll('th')
      .style('z-index', maxZIndex++);

    d3.select('#table-title')
      .style('z-index', maxZIndex++);

    d3.select(this)
      .style('z-index', maxZIndex++);

    dragging = true;
  };


  const sentenceDragStarted = function (event) {
    d3.select(this)
      .style('transform', `translate(${event.x}px, ${event.y}px)`);

    d3.selectAll('.td-tooltip-sentences')
      .style('overflow-x', 'visible')
      .style('overflow-y', 'visible');

    d3.selectAll('.td-tooltip')
      .style('visibility', 'hidden');

    d3.selectAll('.td-text')
      .classed('td-clicked', false);

    d3.selectAll('th')
      .style('z-index', maxZIndex++);

    d3.select('#table-title')
      .style('z-index', maxZIndex++);

    d3.select(this)
      .style('visibility', 'visible')
      .style('display', 'inline-block')
      .style('padding', '7px')
      .style('z-index', maxZIndex++);

    dragging = true;
  };


  const keywordDragged = function (event) {
    if (!d3.select('#bin-div').classed('expanded')) {
      d3.select('#bin-div')
        .classed('expanded', true)
        .style('z-index', maxZIndex - 1);
    }

    d3.select(this)
      .style('left', `${event.x}px`)
      .style('top', `${event.y}px`);
  };


  const sentenceDragged = function (event) {
    if (!d3.select('#bin-div').classed('expanded')) {
      d3.select('#bin-div')
        .classed('expanded', true)
        .style('z-index', maxZIndex - 1);

      d3.selectAll('.td-tooltip')
        .style('z-index', maxZIndex++);
    }

    d3.select(this)
      .style('transform', `translate(${event.x}px, ${event.y}px)`);
  };


  const keywordDragEnded = function (event) {
    if (dragging) {
      d3.select('#bin-div')
        .classed('expanded', false);

      d3.select(this)
        .style('left', '0px')
        .style('top', '0px');

      const tdDiv = d3.select(this);
      const movingText = tdDiv.text();
      const movingSentences = JSON.parse(tdDiv.select('span').attr('data-sentences'));
      const movingColumn = tdDiv.attr('column');
      let targetColumn = null;

      // if not in bin, set targetColumn
      if (d3.pointer(event)[0] < ((window.innerWidth + window.pageXOffset) - 400) ||
      d3.pointer(event)[1] < ((window.innerHeight + window.pageYOffset) - 250)) {
        targetColumn =
        d3.select(document.elementFromPoint(d3.pointer(event)[0] - window.pageXOffset,
          d3.pointer(event)[1] - window.pageYOffset)).attr('column');
      }

      if (targetColumn !== movingColumn) {
        d3.select('table').remove();

        d3.select('#bin-div')
          .style('visibility', 'hidden');

        d3.select('#re-classify-button')
          .attr('disabled', 'disabled');

        d3.select('#table-title')
          .style('padding-left', '0px');

        d3.select('#loading-gif')
          .style('display', 'block');

        d3.select('#loading-text')
          .text('Updating table...')
          .style('display', 'block');

        // setTimeout to avoid freezing
        setTimeout(() => {
          updateData(movingText, movingSentences, movingColumn, targetColumn);
        }, 1);
      }

      dragging = false;
    }
  };


  const sentenceDragEnded = function (event) {
    if (dragging) {
      d3.select('#bin-div')
        .classed('expanded', false);

      const sentenceDiv = d3.select(this);
      const movingSentence = sentenceDiv.text();
      const sentenceType = sentenceDiv.attr('type');
      const movingSentences = {};
      const movingColumn = d3.select(this.parentNode.parentNode.parentNode.parentNode).attr('column');
      let targetColumn = null;

      d3.selectAll('.td-tooltip')
        .remove();

      // if not in bin, set targetColumn
      if (d3.pointer(event)[0] < ((window.innerWidth + window.pageXOffset) - 400) ||
      d3.pointer(event)[1] < ((window.innerHeight + window.pageYOffset) - 250)) {
        targetColumn =
        d3.select(document.elementFromPoint(d3.pointer(event)[0] - window.pageXOffset,
          d3.pointer(event)[1] - window.pageYOffset)).attr('column');
      }

      if (targetColumn !== movingColumn) {
        d3.select('table').remove();

        d3.select('#bin-div')
          .style('visibility', 'hidden');

        d3.select('#re-classify-button')
          .attr('disabled', 'disabled');

        d3.select('#table-title')
          .style('padding-left', '0px');

        d3.select('#loading-gif')
          .style('display', 'block');

        d3.select('#loading-text')
          .text('Updating table...')
          .style('display', 'block');

        if (sentenceType === 'trainSentence') {
          movingSentences.trainSentences = [movingSentence];
          movingSentences.predictSentences = [];
        } else if (sentenceType === 'predictSentence') {
          movingSentences.trainSentences = [];
          movingSentences.predictSentences = [movingSentence];
        }

        // setTimeout to avoid freezing
        setTimeout(() => {
          updateData(null, movingSentences, movingColumn, targetColumn);
        }, 1);
      } else {
        sentenceDiv.remove();
      }

      dragging = false;

      console.log(`moving "${sentenceType}" sentence "${movingSentence}" from "${movingColumn}" to "${targetColumn}"`);
    }
  };


  const generateClickEvents = function () {
    d3.selectAll('.td-with-sentences')
      .each(function () {
        const word = d3.select(this).text().replace(/ \(\d+\)/, '');

        d3.select(this)
          .on('click', function () {
            // remove other tooltips and change font to normal
            d3.selectAll('.td-tooltip')
              .remove();

            d3.selectAll('.td-text')
              .classed('td-clicked', false);

            d3.select(this)
              .classed('td-clicked', true);

            const tooltip = d3.select(this.parentNode.parentNode) // append tooltip to td
              .append('div')
              .classed('td-tooltip', true)
              .style('position', 'absolute'); //inherited?

            tooltip
              .append('div')
              .classed('close-icon-wrapper', true)
              .append('img')
              .attr('src', '../static/res/close.svg')
              .classed('close-icon', true)
              .on('click', function () {
                d3.select(this.parentNode.parentNode.parentNode)
                  .select('.td-text')
                  .classed('td-clicked', false);
                tooltip
                  .remove();
              });

            tooltip
              .append('div')
              .classed('td-tooltip-inner', true)
              .append('div')
              .classed('td-tooltip-sentences', true);

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
                  const regex = new RegExp(`\\b${word}\\b`, 'gi');

                  if (page === 'keywords') {
                    const predictSentences =
                      JSON.parse(d3.select(this).attr('data-sentences')).predictSentences;
                    const trainSentences =
                      JSON.parse(d3.select(this).attr('data-sentences')).trainSentences;

                    for (let i = 0; i < predictSentences.length; i++) {
                      predictSentences[i] = predictSentences[i]
                        .replace(/\\t/g, '\t')
                        .replace(/\\n/g, '\n')
                        .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                        // to-do: encoding
                        .replace(/�/g, ' ')
                        .trim();
                      predictSentences[i] = '<div class="sentence" type="predictSentence" style="background-color: #ebfaff">' +
                        `${predictSentences[i].replace(regex, '<span style="color: #0081eb;' +
                        ` font-weight: bold">${word}</span>`)}</div>`;
                    }

                    for (let i = 0; i < trainSentences.length; i++) {
                      trainSentences[i] = trainSentences[i]
                        .replace(/\\t/g, '\t')
                        .replace(/\\n/g, '\n')
                        .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                        // to-do: encoding
                        .replace(/�/g, ' ')
                        .trim();
                      trainSentences[i] = '<div class="sentence" type="trainSentence" style="background-color: #f2f2f2">' +
                        `${trainSentences[i].replace(regex, '<span style="font-weight: bold">' +
                        `${word}</span>`)}</div>`;
                    }

                    let html;

                    if (predictSentences.length > 0 && trainSentences.length > 0) {
                      html = `${predictSentences.join('</br>')}</br>` +
                        `${trainSentences.join('</br>')}`;
                      } else if (predictSentences.length > 0) {
                        html = predictSentences.join('</br>');
                      } else {
                        html = trainSentences.join('</br>');
                      }

                    return html;
                  } else if (page === 'train_keywords' || 'predict_keywords') {
                    const sentences =
                      JSON.parse(d3.select(this).attr('data-sentences')).sentences;

                    for (let i = 0; i < sentences.length; i++) {
                      sentences[i] = sentences[i]
                        .replace(/\\t/g, '\t')
                        .replace(/\\n/g, '\n')
                        .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                        // to-do: encoding
                        .replace(/�/g, ' ')
                        .trim();
                      if (page === 'predict_keywords') {
                        sentences[i] = '<div class="sentence" type="predict" style="background-color: #ebfaff">' +
                          `${sentences[i].replace(regex, '<span style="color: #0081eb;' +
                          ` font-weight: bold">${word}</span>`)}</div>`;
                      } else {
                        sentences[i] = '<div class="sentence" type="train" style="background-color: #f2f2f2">' +
                          `${sentences[i].replace(regex, '<span style="font-weight: bold">' +
                          `${word}</span>`)}</div>`;
                      }
                    }

                    const html = `${sentences.join('</br>')}</br>` +
                      `${sentences.join('</br>')}`;

                    return html;
                  }
                }
                return tooltipSentences.html();
              });

            d3.selectAll('.sentence')
              .call(d3.drag()
                .subject(function () {
                  // to-do: less hacky?
                  return { x: 0, y: -(this.parentNode.scrollTop * 0.98) };
                })
                .on('start', sentenceDragStarted)
                .on('drag', sentenceDragged)
                .on('end', sentenceDragEnded));
        });
      });
  };


  const updateData = function (movingText, movingSentences, movingColumn, targetColumn) {
    console.log(`moving ${movingText} from ${movingColumn} to ${targetColumn}`);

    // TODO: move to txt?
    const stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
      'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
      'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
      'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
      'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
      'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
      'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
      'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
      'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
      'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
      'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', '\'s',
      '\'t', 'p', '\'ll', 'n', '\'t', 'nt', '\'nt', 'n\'t', '\'m', '\'ve', '\'d',
      'd', '\'re', 're', 'okay', 'like', 'yes', 'actually', 'something', 'going',
      'could', 'would', 'oh', 'ah', 'things', 'think', 'know', 'really', 'well',
      'kind', 'always', 'mean', 'maybe', 'get', 'guess', 'bit', 'much', 'go', 'one',
      'thing', 'probably', 'iv', 'i', 'so', 'dont', 'but', 'and', 'how', 'why',
      'wouldnt', 'wasnt', 'didnt', 'thats', 'thatll', 'im', 'you', 'no', 'isnt',
      'what', 'do', 'did', 'got', 'ill', 'id', 'or', 'do', 'is', 'ive', 'youd',
      'cant', 'wont', 'youve', 'dooesnt', 'is', 'it', 'its', 'the', 'thenokay',
      'theres', 'ofyes', 'reasonsbecause', 'hadnt', 'youre', 'okay', 'if',
      'andyes', 'a'];

    if (!oldThemeDataDictSaved) {
      // CLONE array of objects (save dict state before moving keywords/sentences)
      oldThemeDataDict = JSON.parse(JSON.stringify(themeDataDict));
      oldThemeDataDictSaved = true;
    }

    let movedText = movingText;

    if (!/.*_matrix$/.test(page)) {
      const movingColumnData = themeDataDict[themes.indexOf(movingColumn)][movingColumn];

      if (movingText === null) { // moving 1 sentence
        let movingSentence;
        let sentenceType;

        if (movingSentences.predictSentences.length === 1) {
          sentenceType = 'predictSentence';
          movingSentence = movingSentences.predictSentences[0];
        } else {
          sentenceType = 'trainSentence';
          movingSentence = movingSentences.trainSentences[0];
        }

        const vocab = new Set(movingSentence.split(/\W+/).filter((token) => {
          token = token.toLowerCase();
          return token.length >= 2 && stopWords.indexOf(token) === -1;
        }));

        // console.log(vocab);

        vocab.forEach((word) => {
          word = word.toLowerCase();

          let originalSentence = movingSentence;
          // remove sentence from movingcolumn
          for (let i = 0; i < movingColumnData.length; i++) {
            const movingColumnWord = movingColumnData[i][0].match(/(\w+?)(?: \(\d+\))?$/)[1];
            if (movingColumnWord === word) {
              const movingCount = parseInt(movingColumnData[i][0].match(/\((\d+?)\)/)[1], 10);
              if (movingCount === 1) {
                movingColumnData.splice(i, 1);
              } else {
                // remove sentence
                let index = 1;
                if (sentenceType === 'trainSentence') {
                  index = 2;
                }
                // console.log(`trying to find and remove "${movingSentence}" in "${word}"`);

                for (let j = 0; j < movingColumnData[i][index].length; j++) {
                  const sentence = movingColumnData[i][index][j]
                    .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                    // to-do: encoding
                    .replace(/�/g, ' ')
                    .replace(/\t/g, '    ')
                    .replace(/\n/g, ' ')
                    .trim();

                  // to-do: check why sometimes this is never true
                  // (e.g. move 1st sentence from "ten (2)" from vj -> social)
                  if (sentence === movingSentence) {
                    // update text
                    movedText = `${word} (${movingCount - 1})`;
                    console.log(`${movingColumn}: "${movingColumnData[i][0]}" ==> "${movedText}"`);
                    movingColumnData[i][0] = movedText;
                    // find original sentence (to move later if needed)
                    originalSentence = movingColumnData[i][index][j];
                    // remove sentence
                    movingColumnData[i][index].splice(j, 1);
                  }
                }
              }
            }
          }
          if (targetColumn !== null) { // move sentence into another theme
            const targetColumnData = themeDataDict[themes.indexOf(targetColumn)][targetColumn];

            let keywordJoined = false;

            for (let i = 0; i < targetColumnData.length; i++) {
              const targetWord = targetColumnData[i][0].match(/(\w+?)(?: \(\d+\))?$/)[1];

              if (targetWord === word) {
                const targetCount = parseInt(targetColumnData[i][0].match(/\((\d+?)\)/)[1], 10);
                movedText = `${word} (${targetCount + 1})`;
                console.log(`${targetColumn}: "${targetColumnData[i][0]}" ==> "${movedText}"`);
                targetColumnData[i][0] = movedText;
                // add sentence
                let index = 1;
                if (sentenceType === 'trainSentence') {
                  index = 2;
                }
                targetColumnData[i][index] = targetColumnData[i][index].concat(originalSentence);

                keywordJoined = true;
              }
            }

            if (!keywordJoined) { // keyword not found already in target column
              const themeDataRow = [];
              themeDataRow.push(`${word} (1)`);
              if (sentenceType === 'predictSentence') {
                themeDataRow.push([originalSentence]);
                themeDataRow.push([]);
              } else {
                themeDataRow.push([]);
                themeDataRow.push([originalSentence]);
              }
              console.log(`${targetColumn}: ==> "${word} (1)"`);

              targetColumnData.push(themeDataRow);
            }
          }
        });
      } else { // moving keyword
        for (let i = 0; i < movingColumnData.length; i++) {
          const movingColumnDataRow = movingColumnData[i];

          if (movingColumnDataRow[0] === movingText) {
            if (targetColumn !== null) {
              const targetColumnData = themeDataDict[themes.indexOf(targetColumn)][targetColumn];
              // console.log(JSON.stringify(targetColumnData));

              const movingWord = movingText.match(/(\w+?)(?: \(\d+\))?$/)[1];
              const movingCount = parseInt(movingText.match(/\((\d+?)\)$/)[1], 10);

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

              if (!keywordJoined) { // keyword not found already in target column
                const themeDataRow = [];
                themeDataRow.push(movingColumnDataRow[0]);
                themeDataRow.push(movingColumnDataRow[1]);
                themeDataRow.push(movingColumnDataRow[2]);

                // console.log('themeDataRow (keyword) vvvvvvvvvvvvvvvvvv');
                // console.log(themeDataRow);

                targetColumnData.push(themeDataRow);
              }
            }
            // remove moving cell
            movingColumnData.splice(i, 1);
            break;
          }
        }

        // propagate changes to other cells in movingColumn
        for (let i = 0; i < movingColumnData.length; i++) {
          const movingColumnDataRow = movingColumnData[i];
          const movingWord = movingText.match(/(\w+?)(?: \(\d+\))?$/)[1];
          const regExp = new RegExp(`\\b${movingWord}\\b`, 'i');

          const movingColumnWord = movingColumnDataRow[0].match(/(\w+?)(?: \(\d+\))?$/)[1];
          let movingColumnCount = parseInt(movingColumnDataRow[0].match(/\((\d+?)\)$/)[1], 10);

          for (let j = 0; j < movingColumnDataRow[1].length; j++) {
            const movingColumnPredictSentence = movingColumnDataRow[1][j];

            if (regExp.test(movingColumnPredictSentence)) {
              if (movingColumnCount === 1) {
                movingColumnData.splice(i, 1);
              } else {
                movingColumnDataRow[0] = `${movingColumnWord} (${movingColumnCount - 1})`;
                movingColumnCount--;
                movingColumnDataRow[1].splice(j, 1);
              }
            }
          }

          for (let j = 0; j < movingColumnDataRow[2].length; j++) {
            const movingColumnTrainSentence = movingColumnDataRow[2][j];

            if (regExp.test(movingColumnTrainSentence)) {
              if (movingColumnCount === 1) {
                movingColumnData.splice(i, 1);
              } else {
                movingColumnDataRow[0] = `${movingColumnWord} (${movingColumnCount - 1})`;
                movingColumnCount--;
                movingColumnDataRow[2].splice(j, 1);
              }
            }
          }
        }

        // propagate changes to other cells in targetColumn
        if (targetColumn !== null) {
          const movingWord = movingText.match(/(\w+?)(?: \(\d+\))?$/)[1];
          const targetColumnData = themeDataDict[themes.indexOf(targetColumn)][targetColumn];

          for (let i = 0; i < movingSentences.predictSentences.length; i++) {
            const movingPredictSentence = movingSentences.predictSentences[i];

            const predictVocab = new Set(movingPredictSentence.split(/\W+/).filter((token) => {
              token = token.toLowerCase();
              return token.length >= 2 && stopWords.indexOf(token) === -1;
            }));

            predictVocab.forEach((word) => {
              word = word.toLowerCase();

              let joined = false;

              for (let j = 0; j < targetColumnData.length; j++) {
                const targetColumnDataRow = targetColumnData[j];
                // do not append to moved word
                if (targetColumnDataRow[0].match(/(\w+?)(?: \(\d+\))?$/)[1] !== movingWord) {
                  const targetColumnWord = targetColumnDataRow[0].match(/(\w+?)(?: \(\d+\))?$/)[1];

                  if (targetColumnWord === word) {
                    const targetColumnCount =
                      parseInt(targetColumnDataRow[0].match(/\((\d+?)\)$/)[1], 10);

                    targetColumnDataRow[1] = targetColumnDataRow[1].concat(movingPredictSentence);
                    targetColumnDataRow[0] = `${targetColumnWord} (${targetColumnCount + 1})`;

                    joined = true;
                  }
                }
              }
              if (!joined && word !== movingWord) { // create new cell if not found existing
                const newDataRow = [];
                newDataRow.push(`${word} (1)`);
                newDataRow.push([movingPredictSentence]);
                newDataRow.push([]);

                targetColumnData.push(newDataRow);
              }
            });
          }

          for (let i = 0; i < movingSentences.trainSentences.length; i++) {
            const movingTrainSentence = movingSentences.trainSentences[i];

            const trainVocab = new Set(movingTrainSentence.split(/\W+/).filter((token) => {
              token = token.toLowerCase();
              return token.length >= 2 && stopWords.indexOf(token) === -1;
            }));

            trainVocab.forEach((word) => {
              const lowerWord = word.toLowerCase();

              let joined = false;

              for (let j = 0; j < targetColumnData.length; j++) {
                const targetColumnDataRow = targetColumnData[j];
                // do not append to moved word
                if (targetColumnDataRow[0].match(/(\w+?)(?: \(\d+\))?$/)[1] !== movingWord) {
                  const targetColumnWord = targetColumnDataRow[0].match(/(\w+?)(?: \(\d+\))?$/)[1];

                  if (targetColumnWord === lowerWord) {
                    const targetColumnCount =
                      parseInt(targetColumnDataRow[0].match(/\((\d+?)\)$/)[1], 10);

                    targetColumnDataRow[2] = targetColumnDataRow[2].concat(movingTrainSentence);
                    targetColumnDataRow[0] = `${targetColumnWord} (${targetColumnCount + 1})`;

                    joined = true;
                  }
                }
              }
              if (!joined && lowerWord !== movingWord) { // create new cell if not found existing
                const newDataRow = [];
                newDataRow.push(`${lowerWord} (1)`);
                newDataRow.push([]);
                newDataRow.push([movingTrainSentence]);

                targetColumnData.push(newDataRow);
              }
            });
          }
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

      // console.log('---------themeDataDict AFTER MOVING---------');
      // console.log(themeDataDict);
      // console.log('---------(oldThemeDataDict after moving)-------');
      // console.log(oldThemeDataDict);
      // console.log('---------data AFTER MOVING---------');
      // console.log(data);

      if (movingText !== null) { // moved keyword
        changedData.push({ movedText, movingSentences, movingColumn, targetColumn });
      } else { // moved sentence
        changedData.push({ movedText: null, movingSentences, movingColumn, targetColumn });
      }

      // console.log('changedData vvvv');
      // console.log(changedData);

      generateTable();
    }
  };


  const addReclassifyListener = function () {
    d3.select('#re-classify-button')
      .on('click', () => {
        d3.select('table').remove();

        d3.select('#bin-div')
          .style('visibility', 'hidden');

        d3.select('#table-title')
          .style('padding-left', '0px');

        d3.select('#re-classify-button')
          .attr('disabled', 'disabled');

        d3.select('#loading-gif')
          .style('display', 'block');

        d3.select('#loading-text')
          .text('Re-classifying sentences...')
          .style('display', 'block');
        
        if (reclassifyCount === 0) {
          reclassifyBackend.get_data(page, changedData, true);
        } else {
          reclassifyBackend.get_data(page, changedData, false);
        }
      });
  };

  return { loadTable, loadReclassifiedTable };
}());

// export { loadTable };
