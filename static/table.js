/* global fetch d3 window $*/

const getData = function (page) {
  const sentenceStopWordsRegex = new RegExp('\b(iv|p|a)d+w*|\biv\b|' +
    'd{2}:d{2}:d{2}|speaker key:|interviewer d*|participant w*', 'gi');

  d3.select('#loading-gif')
    .style('display', 'block');

  let url;

  switch (page) {
    case 'predict_keywords':
      url = '/get_predict_keywords_data';
      break;
    case 'train_keywords':
      url = '/get_train_keywords_data';
      break;
    case 'train_codes':
      url = '/get_train_codes_data';
      break;
    default:
      url = '/get_predict_keywords_data';
  }

  fetch(url)
    .then((res) => res.json().then((tableData) => {
        const data = tableData;

        const themes = [
          'practices',
          'social',
          'study vs product',
          'system perception',
          'system use',
          'value judgements',
        ];

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

        const table = d3.select('body')
          .append('table')
          .attr('class', () => {
            if (page === 'predict_keywords') {
              return 'predict';
            } else if (page === 'train_keywords' || page === 'train_codes') {
              return 'train';
            }
          })
          .classed('center', true);
        const thead = table.append('thead');
        const	tbody = table.append('tbody');

        thead.append('tr')
          .selectAll('th')
          .data(themes)
          .enter()
          .append('th')
          .text((theme, i) => `${theme} (${counts[i].toString()})`)
          .style('width', () => (100 / themes.length).toString());

          //jQuery to add shadow to th on scroll
          $(window).scroll(() => {
            const scroll = $(window).scrollTop();
            if (scroll > 0) {
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
          .data((row) => themes.map((column) => ({ column, value: row[column] })))
          .enter()
          .append('td')
          .style('width', () => (100 / themes.length).toString())
          .append('span')
          .classed('td-text', true)
          .text((d) => {
            if (d.value.length === 2) {
              if (page === 'train_keywords' || page === 'predict_keywords') {
                return d.value[0];
              } else if (page === 'train_codes') {
                return `${d.value[0]} (${d.value[1].length})`;
              }
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
                    .replace(/"/g, '"')
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

          const tooltipRect = tooltip.node().getBoundingClientRect();
          const tableRect = d3.select('table').node().getBoundingClientRect();
          const tdRect = this.parentNode.getBoundingClientRect();

          tooltip
            .style('top', () => `${(tdRect.y + tdRect.height + window.scrollY).toString()}px`)
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
                } else if (page === 'predict_keywords') {
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
