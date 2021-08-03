const express = require('express');
const bodyParser = require('body-parser');
const { extractText } = require('doxtract');
const csvtojson = require('csvtojson');
const ObjectsToCsv = require('objects-to-csv');
const { spawn } = require('child_process');
const { performance } = require('perf_hooks');

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

app.use(bodyParser.json({ limit: '50mb' }));

app.use('/static', express.static('static'));

app.get('/main.html', (req, res) => {
  res.sendFile(`${__dirname}/templates/main.html`);
});

app.get('/text.html', (req, res) => {
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

app.get('/main.html', (req, res) => {
  res.sendFile(`${__dirname}/templates/main.html`);
});

const fs = require('fs');

app.get('/get_cm_json', (req, res) => {
  const data = JSON.parse(fs.readFileSync('cm_data.json'));
  res.json(data);
});

app.get(new RegExp(/^\/.*_matrix.html$/), (req, res) => {
  const themeName = req.url.match(/^\/(.*?)_matrix.html$/)[1];
  if (themesNames.map(name => name.replace(/ /g, '_')).includes(themeName)) {
    res.sendFile(`${__dirname}/templates/${themeName}_matrix.html`);
  } else {
    res.sendStatus(404);
  }
});

app.get('/get_text', (req, res) => {
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
              if (obj.codes !== '') {
                const trainSentence = obj.original_sentence;
                const themes = obj.themes;
                jsonObj.push({
                  trainSentence,
                  themes: themes.replace(/;/g, ',')
                });
              }
            }

            for (let i = 0; i < predictObj.length; i++) {
              const obj = predictObj[i];
              // const position = obj.position;
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
                  // position,
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
        csvtojson().fromFile('text/reorder_exit_train_keywords.csv')
        .then((trainKeywordsObj) => {
          csvtojson().fromFile('text/reorder_exit_predict_keywords.csv')
          .then((predictKeywordsObj) => {
            console.log('creating obj to send to client...');
            Object.keys(analyseObj).forEach((analyseKey) => {
              const analyseRow = analyseObj[analyseKey];

              for (let i = 0; i < themesNames.length; i++) {
                const theme = themesNames[i];
                const text = analyseRow[theme];

                if (text.length > 0) {
                  const word = text.replace(/ \(\d+\)/, '').toLowerCase();
                  const predictSentences = [];
                  const trainSentences = [];

                  const predictKeywordsRow =
                    predictKeywordsObj.filter(obj => obj.word === word)[0];

                  if (typeof predictKeywordsRow !== 'undefined') {
                    const predictIndexes = JSON.parse(predictKeywordsRow.sentences);

                    predictIndexes.forEach((index) => {
                      const predictRow = predictObj[index];
                      if (predictRow[theme] === '1' &&
                      predictRow[`${theme} probability`] > minimumProba) {
                        predictSentences.push(predictRow.original_sentence);
                      }
                    });
                  }

                  const trainKeywordsRow = trainKeywordsObj.filter(obj => obj.word === word)[0];

                  if (typeof trainKeywordsRow !== 'undefined') {
                    const trainIndexes = JSON.parse(trainKeywordsRow.sentences);

                    trainIndexes.forEach((index) => {
                      const trainRow = trainObj[index];
                      if (trainRow[theme] === '1') {
                        trainSentences.push(trainRow.original_sentence);
                      }
                    });
                  }

                  analyseRow[theme] = [text];
                  analyseRow[theme].push(predictSentences);
                  analyseRow[theme].push(trainSentences);
                }
              }
            });

            res.json(analyseObj);
            console.log('obj sent to client.');
          });
        });
      });
    });
  });
});

app.get('/get_train_keywords_data', (req, res) => {
  csvtojson().fromFile('text/reorder_exit_train_analyse.csv')
  .then((analyseObj) => {
    csvtojson().fromFile('text/reorder_exit_train.csv')
    .then((trainObj) => {
      csvtojson().fromFile('text/reorder_exit_train_keywords.csv')
      .then((trainKeywordsObj) => {
        console.log('creating obj to send to client...');
        Object.keys(analyseObj).forEach((analyseKey) => {
          const analyseRow = analyseObj[analyseKey];

          for (let i = 0; i < themesNames.length; i++) {
            const theme = themesNames[i];
            const text = analyseRow[theme];

            if (text.length > 0) {
              const word = text.replace(/ \(\d+\)/, '').toLowerCase();
              const trainSentences = [];

              const trainKeywordsRow = trainKeywordsObj.filter(obj => obj.word === word)[0];

              if (typeof trainKeywordsRow !== 'undefined') {
                const trainIndexes = JSON.parse(trainKeywordsRow.sentences);

                trainIndexes.forEach((index) => {
                  const trainRow = trainObj[index];
                  if (trainRow[theme] === '1') {
                    trainSentences.push(trainRow.original_sentence);
                  }
                });

                analyseRow[theme] = [text];
                analyseRow[theme].push(trainSentences);
              }
            }
          }
        });

        res.json(analyseObj);
        console.log('obj sent to client.');
      });
    });
  });
});

app.get('/get_predict_keywords_data', (req, res) => {
  csvtojson().fromFile('text/reorder_exit_predict_analyse.csv')
  .then((analyseObj) => {
    csvtojson().fromFile('text/reorder_exit_predict.csv')
    .then((predictObj) => {
      csvtojson().fromFile('text/reorder_exit_predict_keywords.csv')
      .then((predictKeywordsObj) => {
        console.log('creating obj to send to client...');
        Object.keys(analyseObj).forEach((analyseKey) => {
          const analyseRow = analyseObj[analyseKey];

          for (let i = 0; i < themesNames.length; i++) {
            const theme = themesNames[i];
            const text = analyseRow[theme];

            if (text.length > 0) {
              const word = text.replace(/ \(\d+\)/, '').toLowerCase();
              const predictSentences = [];

              const predictKeywordsRow = predictKeywordsObj.filter(obj => obj.word === word)[0];

              if (typeof predictKeywordsRow !== 'undefined') {
                const predictIndexes = JSON.parse(predictKeywordsRow.sentences);

                predictIndexes.forEach((index) => {
                  const predictRow = predictObj[index];
                  if (predictRow[theme] === '1') {
                    predictSentences.push(predictRow.original_sentence);
                  }
                });

                analyseRow[theme] = [text];
                analyseRow[theme].push(predictSentences);
              }
            }
          }
        });

        res.json(analyseObj);
        console.log('obj sent to client.');
      });
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
        if (trainRow.codes !== '') {
          for (let i = 0; i < themesNames.length; i++) {
            counts[i] += parseInt(trainRow[themesNames[i]], 10);
          }
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

              if (trainRow.codes !== '' && trainRow.codes.includes(code)) {
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

app.post(new RegExp(/^\/re-classify_.*$/), (req, res) => {
  const t0 = performance.now();
  const pageName = req.url.match(/\/re-classify_(.*?)$/)[1];
  if (pageName === 'keywords' || 'keywords_1') {
    console.log(req.body);
    let trainFilePath;
    let predictFilePath;
    if (pageName === 'keywords') {
      trainFilePath = 'text/reorder_exit_train.csv';
      predictFilePath = 'text/reorder_exit_predict.csv';
    } else {
      trainFilePath = 'text/reorder_exit_train_1.csv';
      predictFilePath = 'text/reorder_exit_predict_1.csv';
    }
    csvtojson().fromFile(trainFilePath)
    .then((trainObj) => {
      csvtojson().fromFile(predictFilePath)
      .then((predictObj) => {
        for (let i = 0; i < req.body.length; i++) {
          const changedDataRow = req.body[i];
          const trainSentences = changedDataRow.movingSentences.trainSentences;
          const predictSentences = changedDataRow.movingSentences.predictSentences;
          const movingColumn = changedDataRow.movingColumn;
          const targetColumn = changedDataRow.targetColumn; 

          console.log(`modifying train obj... (${performance.now() - t0})`);

          Object.keys(trainObj).forEach((trainKey) => {
            const trainRow = trainObj[trainKey];
            if (trainRow.codes !== '') {
              const trainRowSentence =
              trainRow.original_sentence
                .replace(/\t/g, '\\t')
                .replace(/\n/g, '\\n');

              if (trainSentences.includes(trainRowSentence)) {
                trainRow.themes = trainRow.themes.replace(movingColumn, targetColumn);
                trainRow[movingColumn] = '0';
                trainRow[targetColumn] = '1';
              }
            }
          });

          for (let j = 0; j < predictSentences.length; j++) {
            const predictSentence = predictSentences[j]
              .replace(/\\t/g, '\t')
              .replace(/\\n/g, '\n');
            const matchingPredictKey = Object.keys(predictObj)
              .find(predictKey =>
                predictObj[predictKey]
                .original_sentence === predictSentence);

            const cleanedSentence = predictObj[matchingPredictKey].cleaned_sentence;
            const sentenceEmbedding = predictObj[matchingPredictKey].sentence_embedding;

            const allThemes = [targetColumn];

            for (let k = 0; k < themesNames.length; k++) {
              const theme = themesNames[k];
              if (theme !== movingColumn && theme !== targetColumn && predictObj[theme] === '1') {
                allThemes.push(theme);
              }
            }

            const newRow = {
              'file name': 'text/reorder_exit.docx',
              comment_id: '',
              original_sentence: predictSentence,
              cleaned_sentence: cleanedSentence,
              sentence_embedding: sentenceEmbedding,
              codes: '',
              themes: allThemes.sort().join('; '),
            };

            for (let k = 0; k < themesNames.length; k++) {
              const theme = themesNames[k];
              if (theme === targetColumn) {
                newRow[theme] = '1';
              } else {
                newRow[theme] = '0';
              }
            }

            trainObj.push(newRow);
          }
        }

        new ObjectsToCsv(trainObj).toDisk('text/reorder_exit_train_1.csv');

        console.log(`done modifying train! calling reclassify script... (${performance.now() - t0})`);

        const reclassifyProcess = spawn('python',
          ['classify_docx.py', 'text/reorder_exit.docx', 'text/reorder_exit_train_1.csv']);

        // reclassifyProcess.stdout.on('data', (data) => {
        //   console.log(data.toString());
        // });
        //
        // reclassifyProcess.stderr.on('data', (data) => {
        //   console.error(data.toString());
        // });

        reclassifyProcess.on('exit', () => {
          console.log(`reclassify script finished! calling analyseProcess... (${performance.now() - t0})`);
          console.log(`calling analyseProcess... (${performance.now() - t0})`);
          const analyseProcess = spawn('python',
            ['analyse_train_and_predict.py', 'text/reorder_exit_train_1.csv']);

          // analyseProcess.stdout.on('data', (data) => {
          //   console.log(data.toString());
          // });
          //
          // analyseProcess.stderr.on('data', (data) => {
          //   console.error(data.toString());
          // });

          analyseProcess.on('exit', () => {
            console.log(`analyse script finished! (${performance.now() - t0})`);
            console.log(`creating json object to send to client... (${performance.now() - t0})`);
            csvtojson().fromFile('text/reorder_exit_analyse_1.csv')
            .then((newAnalyseObj) => {
              csvtojson().fromFile('text/reorder_exit_predict_1.csv')
              .then((newPredictObj) => {
                csvtojson().fromFile('text/reorder_exit_train_1.csv')
                .then((newTrainObj) => {
                  csvtojson().fromFile('text/reorder_exit_train_keywords_1.csv')
                  .then((newTrainKeywordsObj) => {
                    csvtojson().fromFile('text/reorder_exit_predict_keywords_1.csv')
                    .then((newPredictKeywordsObj) => {
                      console.log(`creating obj to send to client... (${performance.now() - t0})`);
                      Object.keys(newAnalyseObj).forEach((newAnalyseKey) => {
                        const newAnalyseRow = newAnalyseObj[newAnalyseKey];

                        for (let i = 0; i < themesNames.length; i++) {
                          const theme = themesNames[i];
                          const text = newAnalyseRow[theme];

                          if (text.length > 0) {
                            const word = text.replace(/ \(\d+\)/, '').toLowerCase();
                            const predictSentences = [];
                            const trainSentences = [];

                            const predictKeywordsRow =
                              newPredictKeywordsObj.filter(obj => obj.word === word)[0];

                            if (typeof predictKeywordsRow !== 'undefined') {
                              const predictIndexes = JSON.parse(predictKeywordsRow.sentences);

                              predictIndexes.forEach((index) => {
                                const predictRow = newPredictObj[index];
                                if (predictRow[theme] === '1' &&
                                predictRow[`${theme} probability`] > minimumProba) {
                                  predictSentences.push(predictRow.original_sentence);
                                }
                              });
                            }

                            const trainKeywordsRow =
                              newTrainKeywordsObj.filter(obj => obj.word === word)[0];

                            if (typeof trainKeywordsRow !== 'undefined') {
                              const trainIndexes = JSON.parse(trainKeywordsRow.sentences);

                              trainIndexes.forEach((index) => {
                                const trainRow = newTrainObj[index];
                                // dont send back moved predicted sentences used in train csv
                                // to avoid duplicate sentences
                                if (trainRow[theme] === '1' && trainRow.codes !== '') {
                                  trainSentences.push(trainRow.original_sentence);
                                }
                              });
                            }

                            newAnalyseRow[theme] = [text];
                            newAnalyseRow[theme].push(predictSentences);
                            newAnalyseRow[theme].push(trainSentences);
                          }
                        }
                      });

                      res.json(newAnalyseObj);
                      const t1 = performance.now();
                      console.log(`done creating obj and sent to client! (${performance.now() - t0})`);
                      console.log(`Re-classify server call took ${t1 - t0} milliseconds.`);
                    });
                  });
                });
              });
            });
          });
        });
      });
    });
  }
});


const server = app.listen(3000, () => {
   const host = server.address().address;
   const port = server.address().port;

   console.log(`App listening at http://${host}:${port}`);
});
