/* global document fetch d3 screen window regexp log themes $*/

const confusionTablesLib = (function () {
  let data;

  let maxZIndex = 1;

  const loadTables = function (tablesData) {
    const startTime = new Date().getTime();

    d3.select('#confusion-tables-container')
      .select('#loading-gif')
      .style('display', 'block');

    data = tablesData;

    generateTables();

    d3.select('#confusion-tables-container')
      .select('#loading-gif')
      .style('display', 'none');

    const endTime = new Date().getTime();
    console.log(`ConfusionTables (JavaScript) => ${((endTime - startTime) / 1000).toFixed(2)} seconds`);
  };


  const generateTables = function () {
    for (let i = 0; i < themes.length; i++) {
      const theme = themes[i];
      const escapedTheme = theme.replace(/([^a-zA-Z\d\s])/g, '').replace(/\s/g, '-');
      const tableData = data[i];
      const titles = data[i][0];
      // remove counts entry
      delete tableData[0];
      tableData.splice(0, 1);

      const tableContainer = d3.select('#confusion-tables-container')
        .append('div')
        .attr('id', `${escapedTheme}-table-container`)
        .classed('cm-container', true)
        .classed('table-container', true);

      const capitalize = function (string) {
        return string.toLowerCase().split(' ')
          .map(word => word.charAt(0).toUpperCase() + word.substring(1)).join(' ');
      };

      tableContainer
        .append('h1')
        .attr('id', 'table-title')
        .text(`"${capitalize(theme)}" Confusion Table`);

      const table = tableContainer
        .append('table')
        .attr('class', 'blue')
        .classed('center', true);

      const thead = table.append('thead');
      const	tbody = table.append('tbody');

      thead.append('tr')
        .selectAll('th')
        .data(titles)
        .enter()
        .append('th')
        .text((title) => title)
        .style('width', (100 / data[0].length).toString());

      const rows = tbody.selectAll('tr')
        .data(tableData)
        .enter()
        .append('tr')
        .attr('position', (row) => tableData.indexOf(row));

      // cells
      rows.selectAll('td')
        .data((row) => titles.map((title) => ({ title, value: row[title] })))
        .enter()
        .append('td')
        .style('position', 'relative')
        .attr('column', (d) => d.title)
        .append('div')
        .classed('td-div', true)
        .style('top', '0px')
        .style('left', '0px')
        .style('width', `${screen.width / titles.length}px`)
        .append('span')
        .classed('td-text', true)
        .text((d) => {
          if (typeof d.value !== 'undefined') {
            return d.value[0];
          }
        })
        .classed('td-with-sentences', (d) => {
          if (typeof d.value !== 'undefined' && d.value[1].length > 0) {
            return true;
          }
          return false;
        })
        .attr('data-sentences', (d) => {
          if (typeof d.value !== 'undefined' && d.value[1].length > 0) {
            const sentences = d.value[1];

            let dataString = '{"sentences": [';

            for (let j = 0; j < sentences.length; j++) {
              const sentence = sentences[j]
                .replace(/\t/g, '\\\\t')
                .replace(/\n/g, '\\\\n')
                .replace(/"/g, "'");
              dataString += (`"${sentence}"`);

              if (j === (sentences.length - 1)) {
                dataString += ']}';
              } else {
                dataString += ', ';
              }
            }
            return dataString;
          }
          return '';
        });

        const titleRect = tableContainer
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

        generateClickEvents(escapedTheme);

        const scrollBarWidth = window.innerWidth - document.documentElement.clientWidth;

        d3.select(`#${escapedTheme}-table-container`)
          .select('#table-title')
          .style('padding-left', `${scrollBarWidth}px`);
    }

    d3.select('#confusion-tables-container')
      .select('#loading-gif')
      .style('display', 'none');
  };


  const generateClickEvents = function (escapedTheme) {
    const tableContainer = d3.select(`#${escapedTheme}-table-container`);

    tableContainer
      .selectAll('.td-with-sentences')
      .each(function () {
        const word = d3.select(this).text().replace(/ \(\d+\)/, '');

        d3.select(this)
          .on('click', function () {
            const columnName = d3.select(this.parentNode.parentNode).attr('column').replace(/ \(\d+\)$/, '');

            log(`keyword at position ${d3.select(this.parentNode.parentNode.parentNode).attr('position')} clicked`);
            // remove other tooltips and change font to normal
            tableContainer
              .selectAll('.td-tooltip')
              .remove();

            tableContainer
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

            tableContainer.selectAll('th')
              .style('z-index', maxZIndex++);

            tableContainer.select('#table-title')
              .style('z-index', maxZIndex++);

            const tooltipSentences = tooltip
              .select('.td-tooltip-sentences');

            tooltipSentences
              .html(() => {
                const regex = new RegExp(`\\b${word}\\b`, 'gi');

                const sentences = JSON.parse(d3.select(this).attr('data-sentences')).sentences;

                for (let i = 0; i < sentences.length; i++) {
                  sentences[i] = sentences[i]
                    .replace(/\\t/g, '\t')
                    .replace(/\\n/g, '\n')
                    .replace(regexp, '') // remove interviewer keywords
                    // to-do: encoding
                    .replace(/�/g, ' ')
                    .trim();
                  sentences[i] = '<div class="sentence" type="predictSentence" style="background-color: #ebfaff">' +
                    `${sentences[i].replace(regex, '<span style="color: #0081eb;' +
                    ` font-weight: bold">${word}</span>`)}</div>`;
                }

                let html;

                if (sentences.length > 0) {
                  html = sentences.join('</br>');
                }

                return html;
              });

              const tooltipRect = tooltip.node().getBoundingClientRect();
              const tableRect = tableContainer.select('table').node().getBoundingClientRect();
              const tdRect = this.parentNode.parentNode.getBoundingClientRect();
  
              tooltip
                .style('top', () => {
                  if (tdRect.y + tooltipRect.height + d3.select('#confusion-tables-container').node().scrollTop < tableRect.height) {             // fix this
                    return `${tdRect.height}px`;
                  }
                  const cutOff = ((tableRect.height - tdRect.y - tooltipRect.height
                     - d3.select('#confusion-tables-container').node().scrollTop) + 105);
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
        });
      });
  };

  return { loadTables };
})();
