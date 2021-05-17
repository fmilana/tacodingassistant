var express = require('express');
var bodyParser = require('body-parser');
const { extractText } = require('doxtract');
var csvtojson = require('csvtojson');

var minimumProba = 0.95;
var themesNames = [
  'practices',
  'social',
  'study vs product',
  'system perception',
  'system use',
  'value judgements'
];

var app = express();

app.use(bodyParser.json());

app.use('/static', express.static('static'));

app.get('/index.html', function(req, res) {
   res.sendFile(__dirname + '/templates/index.html');
});

app.get('/words.html', function(req, res) {
  res.sendFile(__dirname + '/templates/words.html');
})

app.get('/get_html', function(req, res) {
  extractText('text/reorder_exit.docx').then((text) => {
    csvtojson().fromFile('text/reorder_exit_train.csv')
      .then((trainObj) => {
        csvtojson().fromFile('text/reorder_exit_predict.csv')
          .then((predictObj) => {
            var jsonObj = [];
            text = text.replace('/"/g', "'");
            text = text.replace('/’/g', "'");
            jsonObj.push({'wholeText': text});

            for (i = 0; i < trainObj.length; i++) {
              var obj = trainObj[i];
              // var position = obj['position'];
              var trainSentence = obj['original sentence'];
              var themes = obj['themes'];
              jsonObj.push({
                // 'position': position,
                'trainSentence': trainSentence,
                'themes': themes.replace(/;/g, ',')
              });
            }

            for (i = 0; i < predictObj.length; i++) {
              var obj = predictObj[i];
              var position = obj['position'];
              var predictSentence = obj['original sentence'];
              var themes = '';

              for (j = 0; j < themesNames.length; j++) {
                var themeName = themesNames[j];
                if (obj[themeName] == 1 && obj[themeName.concat(' probability')] > minimumProba) {
                  if (themes.length > 0) {
                    themes = themes.concat(', ');
                  }
                  themes = themes.concat(themeName);
                }
              }

              // only push sentence if tagged
              if (themes.length > 0) {
                regExp = /iv[0-9]+|p[0-9]+|p[0-9]+_[0-9]+|/i
                predictSentence = predictSentence.replace(regExp, '').trim()

                jsonObj.push({
                  'position': position,
                  'predictSentence': predictSentence,
                  'themes': themes.replace(/;/, ',')
                });
              }
            }
            res.json(jsonObj);
          });
      });
  });
});

app.get('/get_data', function(req, res) {
  csvtojson().fromFile('text/reorder_exit_analyse.csv')
    .then((analyseObj) => {
      csvtojson().fromFile('text/reorder_exit_predict.csv')
        .then((predictObj) => {

          Object.keys(analyseObj).forEach(function(analyseKey) {
            var analyseRow = analyseObj[analyseKey];

            for (i = 0; i < themesNames.length; i++) {
              var theme = themesNames[i];
              var text = analyseRow[theme];

              if (text.length > 0) {
                var word = text.replace(/ \(\d+\)/, '').toLowerCase();
                var regex = new RegExp('\\b' + word + '\\b', 'i');
                var sentences = [];

                Object.keys(predictObj).forEach(function(predictKey) {
                  var predictRow = predictObj[predictKey];
                  var cleanedSentence = predictRow['cleaned sentence'].toLowerCase();
                  var predictProba = predictRow[theme.concat(' probability')];

                  if (regex.test(cleanedSentence) && predictProba > minimumProba) {
                    var originalSentence = predictRow['original sentence'];
                    sentences.push(originalSentence);
                  }
                });

                analyseRow[theme] = [text];
                analyseRow[theme].push(sentences);

              }
            }
          });

          res.json(analyseObj);
        });
    });
});



var server = app.listen(3000, function() {
   var host = server.address().address;
   var port = server.address().port;

   console.log('Example app listening at http://%s:%s', host, port);
});
