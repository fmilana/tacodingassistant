var getData = function(page) {
  var sentenceStopWordsRegex = /\b(iv|p|a)\d+\w*|\biv\b|\d{2}:\d{2}:\d{2}|speaker key:|interviewer \d*|participant \w*/gi;

  d3.select('#loading-gif')
    .style('display', 'block');

  var url;

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
    .then(function(res) {
      return res.json().then(function(data) {
        var themes = [
          'practices',
          'social',
          'study vs product',
          'system perception',
          'system use',
          'value judgements',
        ];

        var counts;

        if (page == 'predict_keywords' || page == 'train_keywords') {
          counts = [];

          // get counts from first entry
          for (var i = 0; i < themes.length; i++) {
            var theme = themes[i];
            var count = parseInt(data[0][theme][0]);
            counts.push(count);
          }

          // remove counts entry
          delete data[0];
          data.splice(0, 1);
        } else if (page == 'train_codes') {
          counts = data[data.length - 1]['counts'];

          // remove counts entry
          delete data[data.length - 1];
          data.splice(-1, 1);
        }

        d3.select('#loading-gif')
            .style('display', 'none');

    		var table = d3.select('body')
          .append('table')
          .attr('class', function() {
            if (page == 'predict_keywords') {
              return 'predict';
            } else if (page == 'train_keywords' || page == 'train_codes') {
              return 'train';
            }
          })
          .classed('center', true);
    		var thead = table.append('thead');
    		var	tbody = table.append('tbody');

    		thead.append('tr')
    		  .selectAll('th')
    		  .data(themes)
          .enter()
    		  .append('th')
  		    .text(function(theme, i) {
            return theme + ' (' + counts[i].toString() + ')';
          })
          .style('width', function() {
            return (100/themes.length).toString();
          });

          //jQuery to add shadow to th on scroll
          $(window).scroll(function() {
            var scroll = $(window).scrollTop();
            if (scroll > 0) {
              $('th').addClass('active');
            }
            else {
              $('th').removeClass('active');
            }
          });

    		var rows = tbody.selectAll('tr')
    		  .data(data)
    		  .enter()
    		  .append('tr');

    		var cells = rows.selectAll('td')
    		  .data(function(row) {
    		    return themes.map(function(column) {
    		      return {column: column, value: row[column]};
    		    });
    		  })
    		  .enter()
    		  .append('td')
          .style('width', function() {
            return (100/themes.length).toString();
          })
          .append('span')
          .classed('td-text', true)
  		    .text(function(d) {
            if (d.value.length == 2) {
              if (page == 'train_keywords' || page == 'predict_keywords') {
                return d.value[0];
              } else if (page == 'train_codes') {
                return d.value[0] + ' (' + d.value[1].length + ')';
              }
            }
          })
          .classed('td-with-sentences', function(d) {
            if (d.value.length == 2) {
              if (d.value[1].length > 0) {
                return true;
              } else {
                return false;
              }
            }
          })
          .attr('data-sentences', function(d) {
            if (d.value.length == 2) {
              if (d.value[1].length > 0) {
                var sentences = d.value[1];

                var dataString = '{"sentences": [';

                for (var i = 0; i < sentences.length; i++) {
                  // JSON cleaning
                  var sentence = sentences[i]
                    .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                    .replace(/"/g, '\"')
                    // to-do: encoding
                    .replace(/�/g, ' ')
                    .replace(/\t/g, '    ')
                    .replace(/\n/g, ' ')
                    .trim();
                  dataString += ('"' + sentence + '"');

                  if (i == (sentences.length - 1)) {
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
            .each(function(d) {
              var tooltip = d3.select(this)
                .append('div')
                .classed('td-tooltip', true)
                .style('position', 'absolute')
                .style('visibility', 'hidden')

              tooltip
                .append('div')
                .classed('close-icon-wrapper', true)
                .append('img')
                .attr('src', '../static/close.svg')
                .classed('close-icon', true)
                .on('click', function() {
                  tooltip
                    .style('visibility', 'hidden');
                  d3.select(this.parentNode.parentNode.parentNode)
                    .select('.td-text')
                    .classed('td-clicked', false);
                });

              tooltip
                .append('div')
                .classed('td-tooltip-sentences', true)
            });

        generateClickEvents(page);
      });
    });
}


var generateClickEvents = function(page) {
  d3.selectAll('.td-with-sentences')
    .each(function(d) {
      var word = d3.select(this).text();
      word = word.replace(/ \(\d+\)/, '');

      d3.select(this)
        .on('click', function() {
          // hide other tooltips and change font to normal
          d3.selectAll('.td-tooltip')
            .style('visibility', 'hidden');
          d3.selectAll('.td-text')
            .classed('td-clicked', false);

          d3.select(this)
            .classed('td-clicked', true);

          var tooltip = d3.select(this.parentNode)
            .select('.td-tooltip')

          var tooltipRect = tooltip.node().getBoundingClientRect();
          var tableRect = d3.select('table').node().getBoundingClientRect();
          var tdRect = this.parentNode.getBoundingClientRect();

          tooltip
            .style('top', function() {
              return (tdRect.y + tdRect.height + window.scrollY).toString() + 'px';
            })
            .style('left', function() {
              if (tdRect.x == 0) {
                return '10px';
              } else if (tdRect.x + tooltipRect.width < tableRect.width) {
                return tdRect.x.toString() + 'px';
              } else {
                return (tableRect.width - tooltipRect.width - 10).toString() + 'px';
              }
            })
            .style('visibility', 'visible');

          var tooltipSentences = tooltip
            .select('.td-tooltip-sentences');
          var sentences = JSON.parse(d3.select(this).attr('data-sentences'))['sentences'];

          tooltipSentences
            .html(function() {
              if (tooltipSentences.html().length == 0) {
                var text = sentences.join('</br></br>');

                if (page == 'train_keywords') {
                  var regex = new RegExp('\\b' + word + '\\b', 'gi');
                  text = text.replace(regex, '<span style="font-weight: bold">' + word + '</span>');
                } else if (page == 'predict_keywords') {
                  var regex = new RegExp('\\b' + word + '\\b', 'gi');
                  text = text.replace(regex, '<span style="color: #0081eb; font-weight: bold">' + word + '</span>');
                }
                return text;
              } else {
                return tooltipSentences.html();
              }
            });
      });
    });
}

export { getData };
