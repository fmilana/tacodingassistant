var textParagraph = d3.select('#text-paragraph');
var text = '';
var fileName = '';


var get_html = function() {
  d3.select('#loading-gif')
    .style('display', 'block');

  fetch('/get_html')
    .then(function(res) {
      console.log('request back to client!');
      return res.json().then(function(jsonObj) {
        var text = ''
        var trainSentences = [];
        var predictSentences = [];

        for (i = 0; i < jsonObj.length; i++) {
          if (jsonObj[i].hasOwnProperty('trainSentence')) {
            trainSentences.push(jsonObj[i].trainSentence);
          } else if (jsonObj[i].hasOwnProperty('predictSentence')) {
            predictSentences.push(jsonObj[i].predictSentence);
          } else if (jsonObj[i].hasOwnProperty('wholeText')) {
            text = jsonObj[i].wholeText;
          }
        }
        console.log('read all train, predict sentences + whole text');

        text = highlightSentences(text, trainSentences, predictSentences);

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
      });
    });
}

get_html();


var highlightSentences = function(text, trainSentences, predictSentences) {
  var mapObj = {};

  for (i = 0; i < trainSentences.length; i++) {
    var trainSentence = trainSentences[i];
    //better way?
    if (trainSentence.length > 6) {
      mapObj[trainSentence] = '<span style="background-color: #dbdbdb">' + trainSentence + '</span>';
    }
  }

  for (i = 0; i < predictSentences.length; i++) {
    var predictSentence = predictSentences[i];
      // better way?
      if (predictSentence.length > 6) {
        mapObj[predictSentence] = '<span style="background-color: #a8e5ff">' + predictSentence + '</span>';
      }
  }

  // 2 methods: https://stackoverflow.com/questions/15604140/replace-multiple-strings-with-multiple-other-strings

  // Regexp method -------------------------------------------------
  var escapedMapObjKeys = Object.keys(mapObj).map(key => key.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&'));
  var re = new RegExp(escapedMapObjKeys.join('|'), 'gim');

  var j = 0;
  text = text.replace(re, function(matched) {
    return mapObj[matched];
  });

  return text

  // split-join method -------------------------------------------
  // var entries = Object.entries(mapObj);
  //
  // var highlightedText = entries.reduce(
  //   // replace all the occurrences of the keys in the text into an index placholder using split-join
  //   (_text, [key], i) => _text.split(key).join(`{${i}}`),
  //   // manipulate all exisitng index placeholder-like formats, in order to prevent confusion
  //   text.replace(/\{(?=\d+\})/g, '{-')
  // )
  // // replace all index placeholders to the desired replacement values
  // .replace(/\{(\d+)\}/g, (_, i) => entries[i][1])
  // // undo the manipulation of index placeholder -like formats
  // .replace(/\{-(?=\d+\})/g, '{')

  return highlightedText;
}
