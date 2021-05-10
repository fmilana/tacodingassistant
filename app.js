var express = require('express');
var multer = require('multer');
const bodyParser = require('body-parser');
// var mammoth = require('mammoth');
const { extractText } = require('doxtract');
var childProcess = require('child_process');
var csvtojson = require('csvtojson');

var app = express();

app.use(bodyParser.json());

var storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function(req, file, cb) {
    cb(null, file.originalname);
  }
});

var upload = multer({ storage: storage });

app.use('/static', express.static('static'));

app.get('/index.html', function(req, res) {
   res.sendFile(__dirname + '/templates/index.html');
});

app.get('/words.html', function(req, res) {
  res.sendFile(__dirname + '/templates/words.html');
})

app.get('/get_html', function(req, res) {
  extractText('text/reorder_exit.docx').then((text) => {
    // without python script --------------------------------
    csvtojson().fromFile('text/reorder_exit_train.csv')
      .then((trainObj) => {
        csvtojson().fromFile('text/reorder_exit_predict.csv')
          .then((predictObj) => {
            var jsonObj = [];
            // to-do: better way
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

            var minimumProba = 0.95;

            var themeNames = [
              'practices',
              'social',
              'study vs product',
              'system perception',
              'system use',
              'value judgements'
            ];

            for (i = 0; i < predictObj.length; i++) {
              var obj = predictObj[i];
              var position = obj['position'];
              var predictSentence = obj['original sentence'];
              var themes = '';

              for (j = 0; j < themeNames.length; j++) {
                var themeName = themeNames[j];
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


    // with python script -------------------------------------
    // console.log('running classify_docx.py...');
    // var spawn = childProcess.spawn;
    // var pythonProcess = spawn('python', ['classify_docx.py', 'text/reorder_exit.docx']);
    //
    // pythonProcess.on('exit', function() {
    //   console.log('extracting from train csv...');
    //   csvtojson().fromFile('text/reorder_exit_train.csv')
    //     .then((trainObj) => {
    //       console.log('extracting from predict csv...');
    //       csvtojson().fromFile('text/reorder_exit_predict.csv')
    //         .then((predictObj) => {
    //           var jsonObj = [];
    //           // to-do: better way
    //           text = text.replace('/"/g', "'");
    //           text = text.replace('/’/g', "'");
    //           jsonObj.push({'wholeText': text});
    //
    //           for (i = 0; i < trainObj.length; i++) {
    //             var trainSentence = trainObj[i]['original sentence'];
    //             jsonObj.push({'trainSentence': trainSentence});
    //           }
    //           console.log('pushed all train sentences!');
    //           for (i = 0; i < predictObj.length; i++) {
    //             if (predictObj[i]['practices'] == 1 || predictObj[i]['social'] == 1 || predictObj[i]['study vs product'] == 1 || predictObj[i]['system perception'] == 1 || predictObj[i]['system use'] == 1 || predictObj[i]['value judgements'] == 1) {
    //               var predictSentence = predictObj[i]['original sentence'];
    //               // console.log('pushing predictSentence:', predictSentence);
    //               jsonObj.push({'predictSentence': predictSentence});
    //             }
    //           }
    //           console.log('pushed all predict sentences!');
    //           console.log('returning object to client');
    //           res.json(jsonObj);
    //         });
    //     });
    // });
  });
});


app.get('/get_data', function(req, res) {
  csvtojson().fromFile('text/reorder_exit_analyse.csv')
    .then((dataObj) => {
      res.json(dataObj);
    });
});



var server = app.listen(3000, function() {
   var host = server.address().address;
   var port = server.address().port;

   console.log('Example app listening at http://%s:%s', host, port);
});
