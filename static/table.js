/* global fetch d3 window $*/

const getData = function (page) {
  const sentenceStopWordsRegex = new RegExp(/\b(iv|p|a)\d+\s+|p\d+_*\w*\s+|\biv\b|\d{2}:\d{2}:\d{2}|speaker key:|interviewer \d*|participant \w*/, 'gi');

  d3.select('#loading-gif')
    .style('display', 'block');

  fetch(`/get_${page}_data`)
    .then((res) => res.json().then((tableData) => {
        const data = tableData;

        console.log(data);

        let themes = [];
        let cmColNames = [];

        if (page === 'train_keywords' || page === 'train_codes'
        || page === 'predict_keywords') {
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

        let counts;

        if (page === 'predict_keywords' || page === 'train_keywords') {
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
        } else if (page === 'train_codes') {
          counts = data[data.length - 1].counts;

          // remove counts entry
          delete data[data.length - 1];
          data.splice(-1, 1);
        }

        d3.select('#loading-gif')
            .style('display', 'none');

        d3.select('#table-title')
          .style('display', 'block');

        const table = d3.select('body')
          .append('table')
          .attr('class', () => {
            if (page === 'predict_keywords' || page.match(/.*_matrix$/)) {
              return 'blue';
            } else if (page === 'train_keywords' || page === 'train_codes') {
              return 'grey';
            }
          })
          .classed('center', true);
        const thead = table.append('thead');
        const	tbody = table.append('tbody');

        if (page === 'train_keywords' || page === 'train_codes' || page === 'predict_keywords') {
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

        const titleRect = d3.select('#table-title').node().getBoundingClientRect();

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

        const rows = tbody.selectAll('tr')
          .data(data)
          .enter()
          .append('tr');

        // cells
        rows.selectAll('td')
          .data((row) => {
            if (page === 'train_keywords' || page === 'train_codes'
            || page === 'predict_keywords') {
              return themes.map((column) => ({ column, value: row[column] }));
            }
            return cmColNames.map((column) => ({ column, value: row[column] }));
          })
          .enter()
          .append('td')
          .style('width', () => {
            if (page === 'train_keywords' || page === 'train_codes'
            || page === 'predict_keywords') {
              return (100 / themes.length).toString();
            }
            return (100 / cmColNames.length).toString();
          })
          .append('span')
          .classed('td-text', true)
          .text((d) => {
            // keywords/codes
            if (d.value.length === 2) {
              if (page === 'train_codes') {
                return `${d.value[0]} (${d.value[1].length})`;
              }
              return d.value[0];
            }
          })
          .classed('td-with-sentences', (d) => {
            if (d.value.length === 2) {
              if (d.value[1].length > 0) {
                return true;
              }
              return false;
            }
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
            }
          });

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
                .classed('td-tooltip-sentences', true);
            });

        generateClickEvents(page);
      }));
};


const generateClickEvents = function (page) {
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

          const tooltip = d3.select(this.parentNode)
            .select('.td-tooltip');

          const titleRect = d3.select('#table-title').node().getBoundingClientRect();
          const tooltipRect = tooltip.node().getBoundingClientRect();
          const tableRect = d3.select('table').node().getBoundingClientRect();
          const tdRect = this.parentNode.getBoundingClientRect();

          tooltip
            .style('top', () => `${((tdRect.y - titleRect.height)
              + window.scrollY + 13).toString()}px`)
            .style('left', () => {
              if (tdRect.x === 0) {
                return '10px';
              } else if (tdRect.x + tooltipRect.width < tableRect.width) {
                return `${tdRect.x.toString()}px`;
              }
              return `${(tableRect.width - tooltipRect.width - 10).toString()}px`;
            })
            .style('visibility', 'visible');

          const tooltipSentences = tooltip
            .select('.td-tooltip-sentences');
          const sentences = JSON.parse(d3.select(this).attr('data-sentences')).sentences;

          tooltipSentences
            .html(() => {
              if (tooltipSentences.html().length === 0) {
                let text = sentences.join('</br></br>');

                if (page === 'train_keywords') {
                  const regex = new RegExp(`\\b${word}\\b`, 'gi');
                  text = text.replace(regex, `<span style="font-weight: bold">${word}</span>`);
                } else if (page === 'train_codes') {
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


export { getData };
