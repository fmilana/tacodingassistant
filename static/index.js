var textParagraph = d3.select('#text-paragraph');
var text = '';
var codeButton = d3.select('#code-button');
codeButton.attr('disabled', true);
codeButton.on('click', function() {
  textParagraph
    .style('display', 'none');

  d3.select('#loading-gif')
    .style('display', 'block');

  codeText();
});

var fileName = '';


var fileSelector = d3.select('#file-selector');
fileSelector.on('change', function() {
  d3.select('#text-paragraph')
    .remove();
  d3.select('#loading-gif')
    .style('display', 'block');

  fileName = event.target.files[0].name;

  loadFile();
});


var loadFile = function() {
  var file = event.target.files[0];

  console.log('file:', file);

  var formData = new FormData();
  formData.append('doc', file);

  fetch('/load_docx', {method: 'POST', body: formData})
    .then(function(response) {
      return response.json().then(function(jsonObj) {
        text = jsonObj[jsonObj.length - 1].whole_text;

        var sentences = [];

        for (i = 0; i < jsonObj.length - 1; i++) {
          sentences[i] = jsonObj[i].text;
        }

        text = text.replaceAll("’", "'");

        text = highlightSentences(text, sentences, false);

        console.log(text);

        addFileText(text);
      });
    });

  codeButton.attr('disabled', null);
}


var addFileText = function(text) {
  if (!textParagraph.empty()) {
    textParagraph.remove();
  }
  d3.select('#loading-gif')
    .style('display', 'none');

  d3.select('#text-box')
    .selectAll('p')
    .data([text])
    .enter()
    .append('p')
    .attr('id', 'text-paragraph')
    .html(data => data);
}


var highlightSentences = function(text, sentences, predicted) {
  var mapObj = {}

  var color = '';
  if (predicted) {
    color = '#6cd3ff';
  } else {
    color = '#b3b3b3';
  }

  for (i = 0; i < sentences.length; i++) {
    var sentence = sentences[i];
    mapObj[sentence] = '<span style="background-color: ' + color +'">' + sentence + '</span>';
  }

  var re = '';

  if (!predicted) {
    re = new RegExp(Object.keys(mapObj).join('|'), 'gi');
  } else {
    re = new RegExp(Object.keys(mapObj).join('') + '(?![^<]*>|[^<>]*<\/)');
  }
  text = text.replace(re, function(matched) {
    return mapObj[matched];
  });

  return text;
}


var codeText = function() {
  // var formData = new FormData();
  // formData.append('fileName', fileName);

  fetch('/code_docx', {
    method: 'POST',
    body: JSON.stringify({'fileName': fileName}),
    headers:{
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    }})
    .then(function(response) {
      console.log('got here!');
      return response.json().then(function(jsonObj) {
        var sentences = [];

        for (i = 0; i < jsonObj.length; i++) {
          //
          //
          // hard-coded, change
          //
          //
          if (jsonObj[i]['practices'] == 1 || jsonObj[i]['social'] == 1 || jsonObj[i]['study vs product'] == 1 || jsonObj[i]['system perception'] == 1 || jsonObj[i]['system use'] == 1 || jsonObj[i]['value judgements'] == 1) {
            sentences.push(jsonObj[i]['original sentence']);
          }
        }

        text = text.replaceAll("’", "'");

        console.log('number of sentences coded by ai:', sentences.length);

        text = highlightSentences(text, sentences, true);

        addFileText(text);

        textParagraph
          .style('display', 'block');
        d3.select('#loading-gif')
          .style('display', 'none');
      });
    });
}
