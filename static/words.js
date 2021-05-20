var sentenceStopWordsRegex = /\b(iv|p|a)\d+\w*|\biv\b|\d{2}:\d{2}:\d{2}|speaker key:|interviewer \d*|participant \w*/gi;
var themes = [
  'practices',
  'social',
  'study vs product',
  'system perception',
  'system use',
  'value judgements',
];



var generateTables = function(callback) {
  var tableNames = [
    'train_keywords',
    'predict_keywords'
  ];

  var urls = [];

  tableNames.forEach(function(tableName) {
    urls.push('/get_' + tableName + '_data');
  });

  Promise.all(
    urls.map(url =>
      fetch(url)
        .then(function(res) {

          console.log('inside fetch then with url:', url);

          return res.json().then(function(data) {
            var counts = [];

            // get counts from first entry
            for (i = 0; i < themes.length; i++) {
              var theme = themes[i];
              var count = parseInt(data[0][theme][0]);
              counts.push(count);
            }

            // remove counts entry
            delete data[0];
            data.splice(0, 1);

        		var table = d3.select('body')
              .append('table')
              .attr('class', function() {
                if (url == '/get_train_keywords_data') {
                  return 'train-keywords';
                } else if (url == '/get_predict_keywords_data') {
                  return 'predict-keywords';
                }
              })
              .classed('center', true)
              .classed('invisible', true);
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
                //to-do: why are there empty d.value?
                if (d.value.length == 2) {
                  return d.value[0];
                }
              })
              .attr('data-sentences', function(d) {
                //to-do: why are there empty d.value?
                if (d.value.length == 2) {
                  var sentences = d.value[1];

                  var dataString = '{"sentences": [';

                  for (i = 0; i < sentences.length; i++) {
                    // JSON cleaning
                    var sentence = sentences[i]
                      .replace(sentenceStopWordsRegex, '') // remove interviewer keywords
                      .replace(/"/g, '\"')
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
          });
        })
    )
  ).then((value) => {
    console.log('value:', value);
    generateClickEvents();
  });
}


var generateClickEvents = function() {
  console.log('inside generateClickEvents');
  d3.selectAll('.td-text')
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

          var tabRect = d3.select('.tab').node().getBoundingClientRect();
          var tooltipRect = tooltip.node().getBoundingClientRect();
          var tableRect = d3.select('table').node().getBoundingClientRect();
          var tdRect = this.parentNode.getBoundingClientRect();

          tooltip
            .style('top', function() {
              return (tdRect.y + tdRect.height + window.scrollY - tabRect.height).toString() + 'px';
            })
            .style('left', function() {
              if (tdRect.x + tooltipRect.width < tableRect.width) {
                return tdRect.x.toString() + 'px';
              } else {
                // avoid tooltip cut off screen
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
                var regex = new RegExp('\\b' + word + '\\b', 'gi');
                text = text.replace(regex, '<span style="color: #0081eb; font-weight: bold">' + word + '</span>');
                return text;
              } else {
                return tooltipSentences.html();
              }
            });
      });
    });

  console.log('calling updateView...');
  updateView();
}


var generateTabEvents = function() {
  var tabButtons = d3.selectAll('.tablinks');

  tabButtons
    .each(function(d) {
      var tabButton = d3.select(this);

      tabButton
        .on('click', function() {
          tabButtons
            .each(function() {
              d3.select(this)
                .classed('active', false);

              d3.selectAll('table')
                .each(function(d) {

                });
            });

          tabButton
            .classed('active', true);

            d3.selectAll('table')
              .each(function(d) {

              });
        });
    });
}


var updateView = function() {
  console.log('update!');

  d3.select('#loading-gif')
    .style('display', 'none');

  d3.select('.tab')
    .style('display', 'block');

  d3.select('.train-keywords')
    .classed('invisible', false);

  var tabRect = d3.select('.tab').node().getBoundingClientRect();

  //jQuery to add shadow to th on scroll
  $(window).scroll(function() {
    var scroll = $(window).scrollTop();
    if (scroll > tabRect.height) {
      $('th').addClass('active');
    }
    else {
      $('th').removeClass("active");
    }
  });

  generateTabEvents();
}

generateTables();
