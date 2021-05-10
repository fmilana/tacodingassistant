var textParagraph = d3.select('#text-paragraph');
var text = '';
var fileName = '';


var getHtml = function() {
  d3.select('#loading-gif')
    .style('display', 'block');

  fetch('/get_html')
    .then(function(res) {
      console.log('request back to client!');
      return res.json().then(function(jsonObj) {
        var text = ''
        var trainObjects = [];
        var predictObjects = [];

        for (i = 0; i < jsonObj.length; i++) {
          if (jsonObj[i].hasOwnProperty('trainSentence')) {
            trainObjects.push(jsonObj[i]);
          } else if (jsonObj[i].hasOwnProperty('predictSentence')) {
            predictObjects.push(jsonObj[i]);
          } else if (jsonObj[i].hasOwnProperty('wholeText')) {
            text = jsonObj[i].wholeText;
          }
        }
        console.log('read all train, predict sentences + whole text');
        console.log('trainObjects length:', trainObjects.length);
        console.log('predictObjects length:', predictObjects.length);

        text = highlightSentences(text, trainObjects, predictObjects);

        d3.select('#loading-gif')
            .style('display', 'none');

        console.log('updating html...');

        d3.select('#text-box')
          .selectAll('p')
          .data([text])
          .enter()
          .append('p')
          .attr('id', 'text-paragraph')
          .html(data => data);

        console.log('updated html!');

        console.log('generating comments...');

        generateComments();
      });
    });
}


var highlightSentences = function(text, trainObjects, predictObjects) {
  var mapObj = {};

  for (i = 0; i < trainObjects.length; i++) {
    var obj = trainObjects[i];
    // var position = obj.position;
    var trainSentence = obj.trainSentence;
    var themes = obj.themes;
    //better way?
    if (trainSentence.length > 6) {
      mapObj[trainSentence] = '<span data-tooltip="' + themes + '" style="background-color: #dbdbdb">' + trainSentence + '</span>';
    }
  }

  for (i = 0; i < predictObjects.length; i++) {
    var obj = predictObjects[i];
    var position = obj.position
    var predictSentence = obj.predictSentence;

    var themes = obj.themes;
    // better way?
    if (predictSentence.length > 6) {
      mapObj[predictSentence] = '<span data-tooltip="' + themes + '" style="background-color: #a8e5ff">' + predictSentence + '</span>';
    }
  }

  // 2 methods: https://stackoverflow.com/questions/15604140/replace-multiple-strings-with-multiple-other-strings

  // Regexp method (bugged: 366 undefined's) ------------------------------
  // var escapedMapObjKeys = Object.keys(mapObj).map(key => key.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&'));
  // var re = new RegExp(escapedMapObjKeys.join('|'), 'gim');
  //
  // var j = 0;
  // text = text.replace(re, function(matched) {
  //   return mapObj[matched];
  // });
  //
  // return text

  // split-join method -------------------------------------------

  ////////////// -------> to-do: use position somehow

  var entries = Object.entries(mapObj);

  var highlightedText = entries.reduce(
    // replace all the occurrences of the keys in the text into an index placholder using split-join
    (_text, [key], i) => _text.split(key).join(`{${i}}`),
    // manipulate all exisitng index placeholder-like formats, in order to prevent confusion
    text.replace(/\{(?=\d+\})/g, '{-')
  )
  // replace all index placeholders to the desired replacement values
  .replace(/\{(\d+)\}/g, (_, i) => entries[i][1])
  // undo the manipulation of index placeholder -like formats
  .replace(/\{-(?=\d+\})/g, '{')

  return highlightedText;
}


var generateComments = function() {

  commentsObj = []

  d3.selectAll('span')
    .each(function() {
      obj = {
        'themes': d3.select(this).attr('data-tooltip'),
        'y': this.getBoundingClientRect().y,
        'color': d3.select(this).style('background-color')
      };
      commentsObj.push(obj);
    });

  var lastThemes = '';
  var lastY = 0;
  var lastCanvas;

  var lastColor;

  var stacked = 0;

  var i = 0;

  Object.keys(commentsObj).forEach(function(key) {
    var obj = commentsObj[key];

    var y = obj.y.toString();
    var color = obj.color;
    var themes = obj.themes;

    var skipText = (themes == lastThemes) && ((lastY == (y - 28) || (lastY == (y - 33))));
    var concatCanvas = (y == lastY);

    if (!concatCanvas) {
      var canvas = d3.select('#comments-box')
        .append('svg')
        .style('position', 'absolute')
        .style('top', function() {
          if (skipText) {
            y = parseFloat(y) - 5;
            return y.toString().concat('px');
          } else {
            return y.concat('px');
          }
        })
        .attr('width', '400')
        .append('g')

      var rect = canvas.append('rect')
        .attr('height', function() {
          if (skipText) {
            return '30';
          } else {
            return '25';
          }
        })
        .attr('width', '6')
        .attr('x', '0')
        .style('fill', color)

      lastColor = color;

      if (!skipText && (y != lastY)) {
        var text = canvas.append('text')
          .attr('x', '20')
          .attr('y', '17')
          .text(themes);
      } else {
        var text = canvas.append('text')
          .attr('x', '20')
          .attr('y', '17')
          .text('');
      }

      lastCanvas = canvas;
      stacked = 1;
    } else {
      var existingThemes = lastCanvas.select('text').text().split(', ');

      if (color != lastColor) {
        lastCanvas.append('rect')
          .attr('height', '25')
          .attr('width', '6')
          .attr('x', 10 * stacked)
          .style('fill', color);

        lastColor = color;

        stacked += 1;
      }

      var newThemes = themes.split(', ');

      for (i = 0; i < newThemes.length; i++) {
        if (!existingThemes.includes(newThemes[i])) {
          existingThemes.push(newThemes[i]);
        }
      }

      var concatThemes = existingThemes.join(', ');

      lastCanvas.select('text')
        .attr('x', 10 + 10 * stacked)
        .text(concatThemes);
    }

    lastThemes = themes;
    lastY = y;
  });
}



getHtml();
