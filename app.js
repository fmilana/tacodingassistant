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

app.get('/predict_keywords.html', function(req, res) {
  res.sendFile(__dirname + '/templates/predict_keywords.html');
});

app.get('/train_keywords.html', function(req, res) {
  res.sendFile(__dirname + '/templates/train_keywords.html');
});

app.get('/train_codes.html', function(req, res) {
  res.sendFile(__dirname + '/templates/train_codes.html');
});

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

app.get('/get_predict_keywords_data', function(req, res) {
  csvtojson().fromFile('text/reorder_exit_predict_analyse.csv')
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

app.get('/get_train_keywords_data', function(req, res) {
  csvtojson().fromFile('text/reorder_exit_train_analyse.csv')
    .then((analyseObj) => {
      csvtojson().fromFile('text/reorder_exit_train.csv')
        .then((trainObj) => {

          Object.keys(analyseObj).forEach(function(analyseKey) {
            var analyseRow = analyseObj[analyseKey];

            for (i = 0; i < themesNames.length; i++) {
              var theme = themesNames[i];
              var text = analyseRow[theme];

              if (text.length > 0) {
                var word = text.replace(/ \(\d+\)/, '').toLowerCase();
                var regex = new RegExp('\\b' + word + '\\b', 'i');
                var sentences = [];

                Object.keys(trainObj).forEach(function(trainKey) {
                  var trainRow = trainObj[trainKey];
                  var cleanedSentence = trainRow['cleaned sentence'].toLowerCase();

                  if (regex.test(cleanedSentence)) {
                    var originalSentence = trainRow['original sentence'];
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

app.get('/get_train_codes_data', function(req, res) {
  csvtojson().fromFile('text/reorder_exit_themes.csv')
    .then((codesObj) => {
      csvtojson().fromFile('text/reorder_exit_train.csv')
        .then((trainObj) => {

          var counts = [0, 0, 0, 0, 0, 0];

          Object.keys(trainObj).forEach(function(trainKey) {
            var trainRow = trainObj[trainKey];
            for (i = 0; i < themesNames.length; i++) {
              counts[i] += parseInt(trainRow[themesNames[i]]);
            }
          });

          Object.keys(codesObj).forEach(function(codesKey) {
            var codesRow = codesObj[codesKey];

            for (i = 0; i < themesNames.length; i++) {
              var theme = themesNames[i];
              var code = codesRow[theme];

              if (code.length > 0) {
                var sentences = [];

                Object.keys(trainObj).forEach(function(trainKey) {
                  var trainRow = trainObj[trainKey];

                  if (trainRow['codes'].includes(code)) {
                    sentences.push(trainRow['original sentence']);
                  }
                });

                codesRow[theme] = [code];
                codesRow[theme].push(sentences);
              }
            }
          });

          codesObj.push({'counts': counts});

          res.json(codesObj);
        });
    });
});


var server = app.listen(3000, function() {
   var host = server.address().address;
   var port = server.address().port;

   console.log('App listening at http://%s:%s', host, port);
});
