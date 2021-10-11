/* global document fetch d3 screen window regexp log $*/

const codesTableLib = (function () {
  let data;
  let maxZIndex = 1;


  const loadTable = function (tableData, callback) {
    const startTime = new Date().getTime();

    data = tableData;

    generateTable();

    d3.select('#codes-table-container')
      .select('#loading-gif')
      .style('display', 'none');

    const endTime = new Date().getTime();
    console.log(`CodesTable (JavaScript) => ${((endTime - startTime) / 1000).toFixed(2)} seconds`);

    callback();
  };


  const generateTable = function () {
    const titles = data[0];
    const themes = [];
    delete data[0];
    data.splice(0, 1);

    for (let i = 0; i < titles.length; i++) {
      const theme = titles[i].match(/(.+?)(?: \(\d+\))?$/)[1];
      themes.push(theme);
    }

    const table = d3.select('#codes-table-container')
      .append('table')
      .attr('class', 'grey')
      .classed('center', true);

    const thead = table.append('thead');
    const	tbody = table.append('tbody');

    thead.append('tr')
      .selectAll('th')
      .data(titles)
      .enter()
      .append('th')
      .text((title) => title)
      .style('width', () => (100 / titles.length).toString());

    const rows = tbody.selectAll('tr')
      .data(data)
      .enter()
      .append('tr')
      .attr('position', (row) => data.indexOf(row));

    // cells
    rows.selectAll('td')
      .data((row) => { 
        // const rowData = themes.map((theme) => ({ theme, value: row[theme] }));
        // for (let i = 0; i < rowData.length; i++) {
        //   console.log(JSON.stringify(rowData[i]));
        // }

        return themes.map((theme) => ({ theme, value: row[theme] }));
      })
      .enter()
      .append('td')
      .style('position', 'relative')
      .attr('column', (d) => d.theme)
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

          for (let i = 0; i < sentences.length; i++) {
            // JSON cleaning
            const sentence = sentences[i]
              .replace(/\t/g, '\\\\t').replace(/\n/g, '\\\\n');
            dataString += (`"${sentence}"`);

            if (i === (sentences.length - 1)) {
              dataString += ']}';
            } else {
              dataString += ', ';
            }
          }
          return dataString;
        }
        return '';
      });

    const titleRect = d3.select('#codes-table-container').select('#table-title').node().getBoundingClientRect();

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

    d3.select('#codes-table-container')
      .select('#loading-gif')
      .style('display', 'none');

    const scrollBarWidth = window.innerWidth - document.documentElement.clientWidth;

    d3.select('#codes-table-container')
      .select('#table-title')
      .style('padding-left', `${scrollBarWidth}px`);
  };


  const generateClickEvents = function () {
    d3.select('#codes-table-container')
      .selectAll('.td-with-sentences')
      .each(function () {
        d3.select(this)
          .on('click', function () {
            log(`code "${d3.select(this).text()}" (${d3.select(this.parentNode.parentNode).attr('column')}) at position ${d3.select(this.parentNode.parentNode.parentNode).attr('position')} clicked`);

            // remove other tooltips and change font to normal
            d3.select('#codes-table-container')
              .selectAll('.td-tooltip')
              .remove();

            d3.select('#codes-table-container')
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

            const tooltipRect = tooltip.node().getBoundingClientRect();
            const tableRect = d3.select('#codes-table-container')
              .select('table').node().getBoundingClientRect();
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

            d3.select('#codes-table-container')
              .selectAll('th')
              .style('z-index', maxZIndex++);

            d3.select('#codes-table-container')
              .select('#table-title')
              .style('z-index', maxZIndex++);

            const tooltipSentences = tooltip
              .select('.td-tooltip-sentences');

            tooltipSentences
              .html(() => {
                if (tooltipSentences.html().length === 0) {
                  const sentences = JSON.parse(d3.select(this).attr('data-sentences')).sentences;

                  for (let i = 0; i < sentences.length; i++) {
                    const filteredSentence = sentences[i]
                      .replace(/\\t/g, '\t')
                      .replace(/\\n/g, '\n')
                      .replace(regexp, '')
                      .replace(/�/g, ' ')
                      .trim();
                    sentences[i] = `<div style="background-color: #f2f2f2">${filteredSentence}</div>`;
                  }
                  return sentences.join('</br>');
                }
                return tooltipSentences.html();
              });
        });
      });
  };

  return { loadTable };
})();
