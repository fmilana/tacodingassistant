var codeButton = d3.select('#code-button');
codeButton.attr('disabled', true);
codeButton.on('click', code);

var fileSelector = d3.select('#file-selector');
fileSelector.on('change', function() {
  readFile();
});

function readFile() {
  var file = event.target.files[0];
  var fileReader = new FileReader();
  fileReader.readAsText(file);
  fileReader.onload = function() {
    addFileText(fileReader.result);
    codeButton.attr('disabled', null);
  }
}

function addFileText(text) {
  textParagraph = d3.select('#text-paragraph');
  if (!textParagraph.empty()) {
    textParagraph.remove();
  }
  d3.select('#text-box')
    .select('p')
    .data([text])
    .enter()
    .append('p')
    .attr('id', 'text-paragraph')
    .text(data => data);
}

function code() {
  $.ajax({
    type: 'POST',
    url: '/code',
    data: {'text': d3.select('#text-paragraph').text()},
    dataType: 'text',
    beforeSend: function() {
      codeButton.attr('disabled', true);
      // loading gif
    },
    success: function(response) {
      var output_dict = $.parseJSON(response);
      highlightText(output_dict);
      codeButton.attr('disabled', null);
    }
  });
}

function highlightText(dict) {
  var text = d3.select('#text-paragraph').text();
  for (var sentence in dict) {
    var color;
    // console.log(dict[sentence][0]);
    switch(dict[sentence][0]) {
      case 0:
        color = '#e6194B';
        // console.log(sentence + ' ---- red');
        break;
      case 1:
        color = '#3cb44b';
        // console.log(sentence + ' ---- green');
        break;
      case 2:
        color = '#ffe119';
        // console.log(sentence + ' ---- yellow');
        break;
      case 3:
        color = '#4363d8';
        // console.log(sentence + ' ---- blue');
        break;
      case 4:
        color = '#f58231';
        // console.log(sentence + ' ---- orange');
        break;
      case 5:
        color = '#42d4f4';
        // console.log(sentence + ' ---- cyan');
        break;
      case 6:
        color = '#f032e6';
        // console.log(sentence + ' ---- pink');
        break;
      case 7:
        color = '#9A6324';
        // console.log(sentence + ' ---- brown');
        break;
      }

    //// todo: ignore sentences in IV

    // var regExp = new RegExp('<span.*?<\/span>|' + sentence, 'i');
    // var replaceWith = '<span style="background-color: ' + color + ';">' + sentence + '</span>';
    //
    // if (text.includes(sentence)) {
    //   console.log('sentence IS IN TEXT!');
    // } else {
    //   console.log('sentence IS NOT IN TEXT');
    // }
    //
    // text = text.replace(regExp, function(m, group1) {
    //   if (group1 == '') return m;
    //   else return replaceWith;
    // });
  }
  // console.log(text);
  d3.select('#text-paragraph').text(text);
}
