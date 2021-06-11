const express = require('express');
const bodyParser = require('body-parser');
const { extractText } = require('doxtract');
const csvtojson = require('csvtojson');

const minimumProba = 0.95;
const themesNames = [
  'practices',
  'social',
  'study vs product',
  'system perception',
  'system use',
  'value judgements'
];

const app = express();

app.use(bodyParser.json());

app.use('/static', express.static('static'));

app.get('/index.html', (req, res) => {
   res.sendFile(`${__dirname}/templates/index.html`);
});

app.get('/keywords.html', (req, res) => {
  res.sendFile(`${__dirname}/templates/keywords.html`);
});

app.get('/predict_keywords.html', (req, res) => {
  res.sendFile(`${__dirname}/templates/predict_keywords.html`);
});

app.get('/train_keywords.html', (req, res) => {
  res.sendFile(`${__dirname}/templates/train_keywords.html`);
});

app.get('/train_codes.html', (req, res) => {
  res.sendFile(`${__dirname}/templates/train_codes.html`);
});

app.get(new RegExp(/^\/.*_matrix.html$/), (req, res) => {
  const themeName = req.url.match(/^\/(.*?)_matrix.html$/)[1];
  if (themesNames.map(name => name.replace(/ /g, '_')).includes(themeName)) {
    res.sendFile(`${__dirname}/templates/${themeName}_matrix.html`);
  } else {
    res.sendStatus(404);
  }
});

app.get('/get_html', (req, res) => {
  extractText('text/reorder_exit.docx').then((docText) => {
    csvtojson().fromFile('text/reorder_exit_train.csv')
      .then((trainObj) => {
        csvtojson().fromFile('text/reorder_exit_predict.csv')
          .then((predictObj) => {
            const jsonObj = [];
            const text = docText.replace('/"|’/g', "'");
            jsonObj.push({ wholeText: text });

            for (let i = 0; i < trainObj.length; i++) {
              const obj = trainObj[i];
              // var position = obj['position'];
              const trainSentence = obj.original_sentence;
              const themes = obj.themes;
              jsonObj.push({
                // 'position': position,
                trainSentence,
                themes: themes.replace(/;/g, ',')
              });
            }

            for (let i = 0; i < predictObj.length; i++) {
              const obj = predictObj[i];
              const position = obj.position;
              let predictSentence = obj.original_sentence;
              let themes = '';

              for (let j = 0; j < themesNames.length; j++) {
                const theme = themesNames[j];
                if (obj[theme] === '1' && obj[theme.concat(' probability')] > minimumProba) {
                  if (themes.length > 0) {
                    themes = themes.concat(', ');
                  }
                  themes = themes.concat(theme);
                }
              }

              // only push sentence if tagged
              if (themes.length > 0) {
                const regExp = /iv[0-9]+|p[0-9]+|p[0-9]+_[0-9]+|/i;
                predictSentence = predictSentence.replace(regExp, '').trim();

                jsonObj.push({
                  position,
                  predictSentence,
                  themes: themes.replace(/;/, ',')
                });
              }
            }
            res.json(jsonObj);
          });
      });
  });
});

app.get('/get_keywords_data', (req, res) => {
  csvtojson().fromFile('text/reorder_exit_analyse.csv')
  .then((analyseObj) => {
    csvtojson().fromFile('text/reorder_exit_predict.csv')
    .then((predictObj) => {
      csvtojson().fromFile('text/reorder_exit_train.csv')
      .then((trainObj) => {
        // get sentences from both train and predict files
        Object.keys(analyseObj).forEach((analyseKey) => {
          const analyseRow = analyseObj[analyseKey];

          for (let i = 0; i < themesNames.length; i++) {
            const theme = themesNames[i];
            const text = analyseRow[theme];

            if (text.length > 0) {
              const word = text.replace(/ \(\d+\)/, '').toLowerCase();
              const regex = new RegExp(`\\b${word}\\b`, 'i');
              const predictSentences = [];
              const trainSentences = [];

              Object.keys(predictObj).forEach((predictKey) => {
                const predictRow = predictObj[predictKey];
                const cleanedSentence = predictRow.cleaned_sentence.toLowerCase();
                const predictProba = predictRow[theme.concat(' probability')];

                if (regex.test(cleanedSentence) && predictProba > minimumProba) {
                  const originalSentence = predictRow.original_sentence;
                  predictSentences.push(originalSentence);
                }
              });

              Object.keys(trainObj).forEach((trainKey) => {
                const trainRow = trainObj[trainKey];

                if (trainRow[theme] === '1') {
                  const cleanedSentence = trainRow.cleaned_sentence.toLowerCase();

                  if (regex.test(cleanedSentence)) {
                    const originalSentence = trainRow.original_sentence;
                    trainSentences.push(originalSentence);
                  }
                }
              });

              analyseRow[theme] = [text];
              analyseRow[theme].push(predictSentences);
              analyseRow[theme].push(trainSentences);
            }
          }
        });

        res.json(analyseObj);
      });
    });
  });
});

app.get('/get_train_keywords_data', (req, res) => {
  csvtojson().fromFile('text/reorder_exit_train_analyse.csv')
  .then((analyseObj) => {
    csvtojson().fromFile('text/reorder_exit_train.csv')
    .then((trainObj) => {
      Object.keys(analyseObj).forEach((analyseKey) => {
        const analyseRow = analyseObj[analyseKey];

        for (let i = 0; i < themesNames.length; i++) {
          const theme = themesNames[i];
          const text = analyseRow[theme];

          if (text.length > 0) {
            const word = text.replace(/ \(\d+\)/, '').toLowerCase();
            const regex = new RegExp(`\\b${word}\\b`, 'i');
            const sentences = [];

            Object.keys(trainObj).forEach((trainKey) => {
              const trainRow = trainObj[trainKey];

              if (trainRow[theme] === '1') {
                const cleanedSentence = trainRow.cleaned_sentence.toLowerCase();

                if (regex.test(cleanedSentence)) {
                  const originalSentence = trainRow.original_sentence;
                  sentences.push(originalSentence);
                }
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

app.get('/get_predict_keywords_data', (req, res) => {
  csvtojson().fromFile('text/reorder_exit_predict_analyse.csv')
  .then((analyseObj) => {
    csvtojson().fromFile('text/reorder_exit_predict.csv')
    .then((predictObj) => {
      Object.keys(analyseObj).forEach((analyseKey) => {
        const analyseRow = analyseObj[analyseKey];

        for (let i = 0; i < themesNames.length; i++) {
          const theme = themesNames[i];
          const text = analyseRow[theme];

          if (text.length > 0) {
            const word = text.replace(/ \(\d+\)/, '').toLowerCase();
            const regex = new RegExp(`\\b${word}\\b`, 'i');
            const sentences = [];

            Object.keys(predictObj).forEach((predictKey) => {
              const predictRow = predictObj[predictKey];
              const cleanedSentence = predictRow.cleaned_sentence.toLowerCase();
              const predictProba = predictRow[theme.concat(' probability')];

              if (regex.test(cleanedSentence) && predictProba > minimumProba) {
                const originalSentence = predictRow.original_sentence;
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

app.get('/get_train_codes_data', (req, res) => {
  csvtojson().fromFile('text/reorder_exit_themes.csv')
  .then((codesObj) => {
    csvtojson().fromFile('text/reorder_exit_train.csv')
    .then((trainObj) => {
      const counts = [0, 0, 0, 0, 0, 0];

      Object.keys(trainObj).forEach((trainKey) => {
        const trainRow = trainObj[trainKey];
        for (let i = 0; i < themesNames.length; i++) {
          counts[i] += parseInt(trainRow[themesNames[i]], 10);
        }
      });

      Object.keys(codesObj).forEach((codesKey) => {
        const codesRow = codesObj[codesKey];

        for (let i = 0; i < themesNames.length; i++) {
          const theme = themesNames[i];
          const code = codesRow[theme];

          if (code.length > 0) {
            const sentences = [];

            Object.keys(trainObj).forEach((trainKey) => {
              const trainRow = trainObj[trainKey];

              if (trainRow.codes.includes(code)) {
                sentences.push(trainRow.original_sentence);
              }
            });

            codesRow[theme] = [code];
            codesRow[theme].push(sentences);
          }
        }
      });

      codesObj.push({ counts });

      res.json(codesObj);
    });
  });
});

app.get(new RegExp(/^\/get_.*_matrix_data$/), (req, res) => {
  const themeName = req.url.match(/\/get_(.*?)_matrix_data$/)[1];
  if (themesNames.map(name => name.replace(/ /g, '_')).includes(themeName)) {
    csvtojson().fromFile(`text/cm/reorder_exit_${themeName}_cm.csv`)
    .then((cmObj) => {
      csvtojson().fromFile(`text/cm/reorder_exit_${themeName}_cm_analyse.csv`)
      .then((analyseObj) => {
        const colNames = Object.keys(cmObj[0]);

        Object.keys(analyseObj).forEach((analyseKey) => {
          const analyseRow = analyseObj[analyseKey];

          for (let i = 0; i < colNames.length; i++) {
            const colName = colNames[i];
            const text = analyseRow[colName];
            const word = text.replace(/ \(\d+\)$/, '');
            const sentences = [];

            Object.keys(cmObj).forEach((cmKey) => {
              const sentence = cmObj[cmKey][colName];
              const regExp = new RegExp(`\\b${word}\\b`, 'i');

              if (typeof sentence !== 'undefined' && regExp.test(sentence)) {
                sentences.push(sentence);
              }
            });

            analyseRow[colName] = [text];
            analyseRow[colName].push(sentences);
          }
        });

        res.json(analyseObj);
      });
    });
  } else {
    res.sendStatus(404);
  }
});


const server = app.listen(3000, () => {
   const host = server.address().address;
   const port = server.address().port;

   console.log(`App listening at http://${host}:${port}`);
});
