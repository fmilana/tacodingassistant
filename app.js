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
   res.sendFile( __dirname + '/templates/index.html');
});

app.post('/load_docx', upload.single('doc'), function(req, res, next) {
  console.log('file:', req.file);
  console.log('originalname: ', req.file.originalname);

  var docPath = 'uploads/' + req.file.originalname;
  var csvPath = docPath.replace('.docx', '_comments.csv');

  // mammoth.extractRawText({path: docPath})
  //   .then(function(result) {
  //     var text = result.value;
  //     text = text.replace(/’/g, "'");
  //
  //     var spawn = childProcess.spawn;
  //     var pythonProcess = spawn('python', ['export_docx_comments.py', docPath, csvPath]);
  //
  //     pythonProcess.on('exit', function() {
  //       csvtojson().fromFile(csvPath)
  //         .then((jsonObj) => {
  //           jsonObj.push({'whole_text': text});
  //           res.json(jsonObj);
  //         });
  //     });
  //   });

  extractText(docPath).then((text) => {
    var spawn = childProcess.spawn;
    var pythonProcess = spawn('python', ['export_docx_comments.py', docPath, csvPath]);

    pythonProcess.on('exit', function() {
      csvtojson().fromFile(csvPath)
        .then((jsonObj) => {
          jsonObj.push({'whole_text': text});
          res.json(jsonObj);
        });
    });
  });
});

app.post('/code_docx', function(req, res, next) {
  docPath = 'uploads/' + req.body.fileName;
  predPath = docPath.replace('.docx', '_predict.csv')

  var spawn = childProcess.spawn;
  var pythonProcess = spawn('python', ['classify_docx.py', docPath]);

  pythonProcess.on('exit', function() {
    csvtojson().fromFile(predPath)
      .then((jsonObj) => {
        res.json(jsonObj);
      });
  });
});

var server = app.listen(3000, function() {
   var host = server.address().address;
   var port = server.address().port;

   console.log('Example app listening at http://%s:%s', host, port);
});
