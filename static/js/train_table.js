/* global document d3 screen window regexp reclassifyBackend threadStartId log $*/

const trainTableLib = (function () {
  let themeDataDict = [];
  let oldThemeDataDict = [];
  let oldThemeDataDictSaved = false;

  let firstLoading = true;
  let reclassifyCount = 0;

  let counts;

  let data;

  let dragging = false;

  let maxZIndex = 1;

  let keywordZIndex;

  let changedData = [];
  let reclassifyChangesDict = []; // only updated to latest classification
                                  // (for comparison/visualisation purpose)

  const loadTable = function (tableData, callback) {
    const startTime = new Date().getTime();

    data = tableData;

    generateTable();

    d3.select('#train-table-container')
      .select('#loading-gif')
      .style('display', 'none');

    const endTime = new Date().getTime();
    console.log(`TrainTable (JavaScript) => ${((endTime - startTime) / 1000).toFixed(2)} seconds`);

    callback();
  };


  const loadReclassifiedTable = function (reclassifiedData, callback) {
    console.log('===========================> lOADING RECLASSIFIED TRAIN_TABLE');
    const startTime = new Date().getTime();
    data = reclassifiedData;

    firstLoading = true;

    oldThemeDataDictSaved = false;

    reclassifyCount++;

    changedData = [];
    
    // d3.select('#train-table-container')
    //   .select('#loading-text')
    //   .text('Updating table...');

    // timeout needed to change loading text
    setTimeout(() => {
      // if (!d3.select('#train-table-container').select('table').empty()) {
      //   d3.select('#train-table-container')
      //     .select('table')
      //     .remove();

      //   d3.select('#train-table-container')
      //     .select('#loading-gif')
      //     .style('display', 'block');
      // }

      generateTable();

      const endTime = new Date().getTime();
      console.log(`TrainTable (JavaScript) => ${((endTime - startTime) / 1000).toFixed(2)} seconds`);

      console.log('calling callback (other threads)...');
      callback();
    }, 1);
  };


  const generateTable = function () {
    if (firstLoading) {
      counts = [];

      // get counts from first entry
      for (let i = 0; i < themes.length; i++) {
        const count = parseInt(data[0][i], 10);
        counts.push(count);
      }

      // remove counts entry
      delete data[0];
      data.splice(0, 1);

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

          if (dataRow[theme] !== null) {
            const themeDataRow = [];
            themeDataRow.push(dataRow[theme][0]); // text and count
            themeDataRow.push(dataRow[theme][1]); // train sentences
            themeData.push(themeDataRow);
          }
        }
        themeDataDict.push({ [theme]: themeData });
      }

      // create classification dict if it does not exist yet (empty)
      if (oldThemeDataDict.length > 0) {
      // find reclassify differences (to highlight later)
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
      }

      // console.log('oldThemeDataDict vvvvv');
      // console.log(oldThemeDataDict);
      // console.log('themeDataDict vvvvvvv');
      // console.log(themeDataDict);
      // console.log('reclassifyChangesDict vvvvvvvvvv');
      // console.log(reclassifyChangesDict);
    }

    const table = d3.select('#train-table-container')
      .append('table')
      .attr('class', 'grey')
      .classed('center', true);

    const thead = table.append('thead');
    const	tbody = table.append('tbody');

    thead.append('tr')
      .selectAll('th')
      .data(themes)
      .enter()
      .append('th')
      .text((theme, k) => `${theme} (${counts[k].toString()})`)
      .style('width', () => (100 / themes.length).toString());

    const rows = tbody.selectAll('tr')
      .data(data)
      .enter()
      .append('tr')
      .attr('position', (row) => data.indexOf(row));

    // cells
    rows.selectAll('td')
      .data((row) => themes.map((theme) => ({ theme, value: row[theme] })))
      .enter()
      .append('td')
      .style('position', 'relative')
      .attr('column', (d) => d.theme)
      .append('div')
      .classed('td-div', true)
      .style('top', '0px')
      .style('left', '0px')
      .style('width', `${screen.width / themes.length}px`)
      .attr('column', (d) => d.theme)
      .append('span')
      .classed('td-text', true)
      .text((d) => {
        if (d.value !== null && typeof d.value !== 'undefined' && d.value.length > 0) {
          return d.value[0];
        }
      })
      .classed('td-with-sentences', (d) => {
        if (d.value !== null && typeof d.value !== 'undefined' && d.value.length > 1) {
          if (d.value[1].length > 0) {
            return true;
          }
          return false;
        }
      })
      .attr('column', (d) => d.theme)
      .attr('data-sentences', (d) => {
        if (d.value !== null && typeof d.value !== 'undefined' && d.value.length > 0) {
          const trainSentences = d.value[1];

          if (trainSentences.length > 0) {
            let dataString = '{"trainSentences": [';

            for (let j = 0; j < trainSentences.length; j++) {
              // JSON cleaning
              const sentence = trainSentences[j]
                .replace(/\t/g, '\\\\t')
                .replace(/\n/g, '\\\\n')
                .replace(/"/g, "'");

              dataString += (`"${sentence}"`);

              if (j === (trainSentences.length - 1)) {
                dataString += ']}';
              } else {
                dataString += ', ';
              }
            }
            return dataString;
          }
        }
      });

      // highlight cells that have been modified after reloading data
      d3.select('#train-table-container')
      .selectAll('.td-text')
        .each(function () {
          const tdText = d3.select(this);
          if (tdText.text().length > 0) {
            const tdTheme = tdText.attr('column');
            const tdWord = tdText.text().match(/(\w+?)(?: \(\d+\))?$/)[1];
            const tdSentences = JSON.parse(tdText.attr('data-sentences')); /////// <===

            let moved = false;
            let reclassified = false;

            for (let j = 0; j < changedData.length; j++) {
              const changedDataRow = changedData[j];
              let movedSentence;

              if (changedDataRow.movedText === null) { // check moved single sentences
                if (changedDataRow.movingSentences.trainSentences.length > 0) {
                  movedSentence = changedDataRow.movingSentences.trainSentences[0];
                }
              }

              if (tdTheme === changedDataRow.targetColumn                 //
                && (tdText.text() === changedDataRow.movedText            // check if keyword was moved
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
                .style('border', '4px solid #8f8d8d');
            }
          }
        });
      // }

    const titleRect = d3.select('#train-table-container')
      .select('#table-title').node().getBoundingClientRect();

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

    d3.select('#train-table-container')
      .selectAll('.td-div')
      .call(d3.drag()
        .subject(function () {
          const t = d3.select(this);
          return { x: t.attr('x'), y: t.attr('y') };
        })
        .on('start', keywordDragStarted)
        .on('drag', keywordDragged)
        .on('end', keywordDragEnded));

    d3.select('#train-table-container')
      .select('#loading-gif')
      .style('display', 'none');

    // d3.select('#train-table-container')
    //   .select('#loading-text')
    //   .style('display', 'none');

    const scrollBarWidth = window.innerWidth - document.documentElement.clientWidth;

    d3.select('#table-title')
      .style('padding-left', `${scrollBarWidth}px`);

    if (firstLoading) {
      addReclassifyListener();
      firstLoading = false;
    } else {
      d3.select('#train-table-container')
      .select('#re-classify-button')
        .attr('disabled', null);
    }

    d3.select('#train-table-container')
      .select('#bin-div')
      .style('visibility', 'visible')
      .style('z-index', maxZIndex);
  };


  const keywordDragStarted = function () {
    d3.select('#train-table-container')
      .selectAll('.td-tooltip')
      .remove();

    d3.select('#train-table-container')
      .selectAll('.td-text')
      .classed('td-clicked', false);

    d3.select('#train-table-container')
      .selectAll('th')
      .style('z-index', maxZIndex++);

    d3.select('#train-table-container')
      .select('#table-title')
      .style('z-index', maxZIndex++);

    keywordZIndex = d3.select(this).style('z-index'); // reassign after dragging (fixes bug)

    d3.select(this)
      .classed('dragging', true)
      .style('z-index', maxZIndex++);

    dragging = true;
  };


  const sentenceDragStarted = function (event) {
    d3.select(this)
      .style('transform', `translate(${event.x}px, ${event.y}px)`);

    d3.select('#train-table-container')
      .selectAll('.td-tooltip-sentences')
      .style('overflow-x', 'visible')
      .style('overflow-y', 'visible');

    d3.select('#train-table-container')
      .selectAll('.td-tooltip')
      .style('visibility', 'hidden');

    d3.select('#train-table-container')
      .selectAll('.td-text')
      .classed('td-clicked', false);

    d3.select('#train-table-container')
      .selectAll('th')
      .style('z-index', maxZIndex++);

    d3.select('#train-table-container')
      .select('#table-title')
      .style('z-index', maxZIndex++);

    d3.select(this) 
      .classed('dragging', true)
      .style('visibility', 'visible')
      .style('display', 'inline-block')
      .style('padding', '7px')
      .style('z-index', maxZIndex++);

    dragging = true;
  };


  const keywordDragged = function (event) {
    if (!d3.select('#train-table-container').select('#bin-div').classed('expanded')) {
      d3.select('#train-table-container')
        .select('#bin-div')
        .classed('expanded', true)
        .style('z-index', maxZIndex - 1);
    }

    d3.select(this)
      .style('left', `${event.x}px`)
      .style('top', `${event.y}px`);
  };


  const sentenceDragged = function (event) {
    if (!d3.select('#train-table-container').select('#bin-div').classed('expanded')) {
      d3.select('#train-table-container')
        .select('#bin-div')
        .classed('expanded', true)
        .style('z-index', maxZIndex - 1);

      d3.select('#train-table-container')
        .selectAll('.td-tooltip')
        .style('z-index', maxZIndex++);
    }

    d3.select(this)
      .style('transform', `translate(${event.x}px, ${event.y}px)`);
  };


  const keywordDragEnded = function (event) {
    if (dragging) {
      d3.select('#train-table-container')
        .select('#bin-div')
        .classed('expanded', false);

      d3.select(this)
        .classed('dragging', false)
        .style('left', '0px')
        .style('top', '0px')
        .style ('z-index', keywordZIndex);

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
        d3.select('#train-table-container')
          .select('table').remove();

        d3.select('#train-table-container')
          .select('#bin-div')
          .style('visibility', 'hidden');

        d3.select('#train-table-container')
          .select('#re-classify-button')
          .attr('disabled', 'disabled');

        d3.select('#train-table-container')
          .select('#table-title')
          .style('padding-left', '0px');

        d3.select('#train-table-container')
          .select('#loading-gif')
          .style('display', 'block');

        // d3.select('#train-table-container')
        //   .select('#loading-text')
        //   .text('Updating table...')
        //   .style('display', 'block');
        log(`train keyword "${movingText}" (${movingColumn}) at position ${d3.select(this.parentNode.parentNode).attr('position')} moved to "${targetColumn}"`);
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
      d3.select(this)
        .classed('dragging', false);

      d3.select('#train-table-container')
        .select('#bin-div')
        .classed('expanded', false);

      const sentenceDiv = d3.select(this);
      const movingSentence = sentenceDiv.text();
      const movingSentences = {};
      const movingColumn = d3.select(this.parentNode.parentNode.parentNode.parentNode).attr('column');
      let targetColumn = null;

      d3.select('#train-table-container')
        .selectAll('.td-tooltip')
        .remove();

      // if not in bin, set targetColumn
      if (d3.pointer(event)[0] < ((window.innerWidth + window.pageXOffset) - 400) ||
      d3.pointer(event)[1] < ((window.innerHeight + window.pageYOffset) - 250)) {
        targetColumn =
        d3.select(document.elementFromPoint(d3.pointer(event)[0] - window.pageXOffset,
          d3.pointer(event)[1] - window.pageYOffset)).attr('column');
      }

      if (targetColumn !== movingColumn) {
        d3.select('#train-table-container')
          .select('table').remove();

        d3.select('#train-table-container')
          .select('#bin-div')
          .style('visibility', 'hidden');

        d3.select('#train-table-container')
          .select('#re-classify-button')
          .attr('disabled', 'disabled');

        d3.select('#train-table-container')
          .select('#table-title')
          .style('padding-left', '0px');

        d3.select('#train-table-container')
          .select('#loading-gif')
          .style('display', 'block');

        // d3.select('#train-table-container')
        //   .select('#loading-text')
        //   .text('Updating table...')
        //   .style('display', 'block');
     
        movingSentences.trainSentences = [movingSentence];

        log(`train sentence moved to "${targetColumn}"`);

        // setTimeout to avoid freezing
        setTimeout(() => {
          updateData(null, movingSentences, movingColumn, targetColumn);
        }, 1);
      } else {
        sentenceDiv.remove();
      }

      dragging = false;

      // console.log(`moving trainSentence "${movingSentence}" from "${movingColumn}" to "${targetColumn}"`);
    }
  };


  const generateClickEvents = function () {
    d3.select('#train-table-container')
      .selectAll('.td-with-sentences')
      .each(function () {
        const word = d3.select(this).text().replace(/ \(\d+\)/, '');

        d3.select(this)
          .on('click', function () {
            log(`train keyword "${d3.select(this).text()}" (${d3.select(this).attr('column')}) at position ${d3.select(this.parentNode.parentNode.parentNode).attr('position')} clicked`);
            // remove other tooltips and change font to normal
            d3.select('#train-table-container')
              .selectAll('.td-tooltip')
              .remove();

            d3.select('#train-table-container')
              .selectAll('.td-text')
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
                log('tooltip closed');
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

            d3.select('#train-table-container')
              .selectAll('th')
              .style('z-index', maxZIndex++);

            d3.select('#train-table-container')
              .select('#table-title')
              .style('z-index', maxZIndex++);

            const tooltipSentences = tooltip
              .select('.td-tooltip-sentences');

            tooltipSentences
              .html(() => {
                if (tooltipSentences.html().length === 0) {
                  const regex = new RegExp(`\\b${word}\\b`, 'gi');

                  const trainSentences =
                    JSON.parse(d3.select(this).attr('data-sentences')).trainSentences;

                  for (let i = 0; i < trainSentences.length; i++) {
                    trainSentences[i] = trainSentences[i]
                      .replace(/\\t/g, '\t')
                      .replace(/\\n/g, '\n')
                      .replace(/"/g, "'")
                      .replace(regexp, '') // remove interviewer keywords
                      // to-do: encoding
                      .replace(/�/g, ' ')
                      .trim();
                    trainSentences[i] = '<div class="sentence" type="trainSentence" style="background-color: #f2f2f2">' +
                      `${trainSentences[i].replace(regex, '<span style="font-weight: bold">' +
                      `${word}</span>`)}</div>`;
                  }

                  let html;

                  if (trainSentences.length > 0) {
                    html = trainSentences.join('</br>');
                  }

                  return html;
                }
                return tooltipSentences.html();
              });

            const tooltipRect = tooltip.node().getBoundingClientRect();
            const tableRect = d3.select('#train-table-container').select('table').node().getBoundingClientRect();
            const tdRect = this.parentNode.parentNode.getBoundingClientRect();

            tooltip
              .style('top', () => {
                if (tdRect.y + tooltipRect.height + d3.select('#train-table-container').node().scrollTop < tableRect.height) {             // fix this
                  return `${tdRect.height}px`;
                }
                const cutOff = ((tableRect.height - tdRect.y - tooltipRect.height
                  - d3.select('#train-table-container').node().scrollTop) + 105);
                return `${cutOff}px`;
              })
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

            d3.select('#train-table-container')
              .selectAll('.sentence')
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
    // console.log(`moving ${movingText} from ${movingColumn} to ${targetColumn}`);

    // TODO: move to txt?
    const stopWords = ['yeah', 'yeah,', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
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

    const movingColumnData = themeDataDict[themes.indexOf(movingColumn)][movingColumn];

    if (movingText === null) { // moving 1 sentence
      let movingSentence;

      if (movingSentences.trainSentences.length === 1) {
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
              // console.log(`trying to find and remove "${movingSentence}" in "${word}"`);

              for (let j = 0; j < movingColumnData[i][1].length; j++) {
                const sentence = movingColumnData[i][1][j]
                  .replace(regexp, '') // remove interviewer keywords
                  // to-do: encoding
                  .replace(/�/g, ' ')
                  .replace(/\t/g, '    ')
                  .replace(/\n/g, ' ')
                  .replace(/"/g, "'")
                  .trim();

                // to-do: check why sometimes this is never true
                // (e.g. move 1st sentence from "ten (2)" from vj -> social)
                if (sentence === movingSentence) {
                  // update text
                  movedText = `${word} (${movingCount - 1})`;
                  // console.log(`${movingColumn}: "${movingColumnData[i][0]}" ==> "${movedText}"`);
                  movingColumnData[i][0] = movedText;
                  // find original sentence (to move later if needed)
                  originalSentence = movingColumnData[i][1][j];
                  // remove sentence
                  movingColumnData[i][1].splice(j, 1);
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
              // console.log(`${targetColumn}: "${targetColumnData[i][0]}" ==> "${movedText}"`);
              targetColumnData[i][0] = movedText;
              // add sentence
              targetColumnData[i][1] = targetColumnData[i][1].concat(originalSentence);

              keywordJoined = true;
            }
          }

          if (!keywordJoined) { // keyword not found already in target column
            const themeDataRow = [];
            themeDataRow.push(`${word} (1)`);
            themeDataRow.push([originalSentence]);
            // console.log(`${targetColumn}: ==> "${word} (1)"`);

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

                keywordJoined = true;
              }
            }

            if (!keywordJoined) { // keyword not found already in target column
              const themeDataRow = [];
              themeDataRow.push(movingColumnDataRow[0]);
              themeDataRow.push(movingColumnDataRow[1]);

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
          const movingColumntrainSentence = movingColumnDataRow[1][j];

          if (regExp.test(movingColumntrainSentence)) {
            if (movingColumnCount === 1) {
              movingColumnData.splice(i, 1);
            } else {
              movingColumnDataRow[0] = `${movingColumnWord} (${movingColumnCount - 1})`;
              movingColumnCount--;
              movingColumnDataRow[1].splice(j, 1);
            }
          }
        }
      }

      // propagate changes to other cells in targetColumn
      if (targetColumn !== null) {
        const movingWord = movingText.match(/(\w+?)(?: \(\d+\))?$/)[1];
        const targetColumnData = themeDataDict[themes.indexOf(targetColumn)][targetColumn];

        for (let i = 0; i < movingSentences.trainSentences.length; i++) {
          const movingtrainSentence = movingSentences.trainSentences[i];

          const trainVocab = new Set(movingtrainSentence.split(/\W+/).filter((token) => {
            token = token.toLowerCase();
            return token.length >= 2 && stopWords.indexOf(token) === -1;
          }));

          trainVocab.forEach((word) => {
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

                  targetColumnDataRow[1] = targetColumnDataRow[1].concat(movingtrainSentence);
                  targetColumnDataRow[0] = `${targetColumnWord} (${targetColumnCount + 1})`;

                  joined = true;
                }
              }
            }
            if (!joined && word !== movingWord) { // create new cell if not found existing
              const newDataRow = [];
              newDataRow.push(`${word} (1)`);
              newDataRow.push([movingtrainSentence]);

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
  };


  const addReclassifyListener = function () {
    d3.select('#train-table-container')
      .select('#re-classify-button')
      .on('click', () => {
        log('reclassify');

        if (!d3.select('#text-container').select('.row').empty()) {
          d3.select('#text-container').select('.row').remove();
          d3.select('#text-container').select('#loading-gif').style('display', 'block');
        } 
        if (!d3.select('#all-table-container').select('table').empty()) {
          d3.select('#all-table-container').select('table').remove();
          d3.select('#all-table-container').select('#loading-gif').style('display', 'block');
        }
        if (!d3.select('#predict-table-container').select('table').empty()) {
          d3.select('#predict-table-container').select('table').remove();
          d3.select('#predict-table-container').select('#loading-gif').style('display', 'block');
        }
        if (!d3.selectAll('.cm-container').empty()) {
          d3.selectAll('.cm-container').remove();
          d3.select('#confusion-tables-container').select('#loading-gif').style('display', 'block');
        }

        d3.select('#train-table-container')
          .select('table').remove();

        d3.select('#train-table-container')
          .select('#bin-div')
          .style('visibility', 'hidden');

        d3.select('#train-table-container')
          .select('#table-title')
          .style('padding-left', '0px');

        d3.select('#train-table-container')
          .select('#re-classify-button')
          .attr('disabled', 'disabled');

        d3.select('#train-table-container')
          .select('#loading-gif')
          .style('display', 'block');

        // d3.select('#train-table-container')
        //   .select('#loading-text')
        //   .text('Re-classifying sentences...')
        //   .style('display', 'block');

        threadStartId = 'train-keywords-button';

        if (reclassifyCount === 0) {
          reclassifyBackend.get_data(changedData, true);
        } else {
          reclassifyBackend.get_data(changedData, false);
        }
      });
  };

  return { loadTable, loadReclassifiedTable };
})();
