/* global d3 window qt QWebChannel importLib textLib codesTableLib trainTableLib predictTableLib allTableLib confusionTablesLib wordDelimiter */

let themes = [];

const tabToContainerDict = {
  'text-button': 'text-container',
  'codes-button': 'codes-table-container',
  'all-keywords-button': 'all-table-container',
  'predict-keywords-button': 'predict-table-container',
  'train-keywords-button': 'train-table-container'
};

let setupBackend;
let textBackend;
let codesTableBackend;
let allTableBackend;
let predictTableBackend;
let trainTableBackend;
let reclassifyBackend;
let confusionTablesBackend;
let logBackend;
// eslint-disable-next-line no-unused-vars
let importBackend;

// eslint-disable-next-line prefer-const
let threadStartId = 'text-button'; // which page started threads
let tabId = 'text-button';
let currentTabId = 'text-button';

// eslint-disable-next-line no-unused-vars
let regexp = null;

let software = null;

let transcriptPath = null;
let nvivoCodesPath = null;
let MAXQDASegmentsPath = null;
let dedooseExcerptsPath = null;
let themeCodeTablePath = null;


const log = function (message) {
  const msTime = new Date().getTime();
  const dateTime = new Date(msTime);
  logBackend.log(`[${dateTime.toLocaleString()} (${msTime})]: ${message}`);
};


const onImportData = function (data) {
  switch (data[0]) {
    case 'transcript':
      // transcript file
      transcriptPath = data[1];
      if (transcriptPath !== '') {
        d3.select('#import-transcript-button')
          .text(/[^/]*$/.exec(transcriptPath)[0]);
        d3.select('#import-transcript-next-button')
          .property('disabled', false);
      }
      break;
    case 'NVivoCodesFolder':
      // nvivo code folder
      nvivoCodesPath = data[1];
      if (nvivoCodesPath !== '') {
        d3.select('#import-nvivo-codes-folder-button')
          .text(/[^/]*$/.exec(nvivoCodesPath)[0]);
        d3.select('#import-nvivo-codes-folder-next-button')
          .property('disabled', false);
      }
      break;
    case 'MAXQDASegments':
      // maxqda segments
      MAXQDASegmentsPath = data[1];
      if (MAXQDASegmentsPath !== '') {
        d3.select('#import-maxqda-document-button')
          .text(/[^/]*$/.exec(MAXQDASegmentsPath)[0]);
        d3.select('#import-maxqda-document-next-button')
          .property('disabled', false);
      }
      break;
    case 'DedooseExcerpts':
      // dedoose excerpts
      dedooseExcerptsPath = data[1];
      if (dedooseExcerptsPath !== '') {
        d3.select('#import-dedoose-excerpts-button')
          .text(/[^/]*$/.exec(dedooseExcerptsPath)[0]);
        d3.select('#import-dedoose-excerpts-next-button')
          .property('disabled', false);
      }
      break;
    case 'codeThemeTable':
      // code table
      themeCodeTablePath = data[1];
      if (themeCodeTablePath !== '') { 
        // d3.select('#import-theme-code-table-button')
        //   .text(/[^/]*$/.exec(themeCodeTablePath)[0]);
        // d3.select('#import-theme-code-table-next-button')
        //   .property('disabled', false);
        d3.select('#import-loading-code-theme-table-container')
          .style('display', 'none');
        d3.select('#import-edit-code-theme-table-container')
          .style('display', 'block');
        d3.select('#import-edit-code-theme-table-path')
          .text(`${themeCodeTablePath}`);

        d3.selectAll('.dynamic-stepper')
          .text(function(d) {
            if (data[2] === 'fromWord') {
              return 'Enter delimiter';
            } else if (data[2] === 'fromNVivo') {
              return 'Select codes folder';
            } else if (data[2] === 'fromMAXQDA') {
              return 'Import coded segments';
            } else if (data[2] === 'fromDedoose') {
              return 'Import excerpts';
            }
          });
      }
      break;
    default:
      // keywords
      const valid = data[2];
      if (valid) {
        d3.select('#import-container')
          .remove();

        d3.select('#setup-container')
          .style('display', 'block');

        setupBackend.set_up(transcriptPath, software, window.wordDelimiter, nvivoCodesPath, MAXQDASegmentsPath, dedooseExcerptsPath, themeCodeTablePath);
      } else {
        // error message
        alert('Please check your filtered keywords or regular expression');
        log('import error');
      }
  }
};


const onSetup = function () {
  d3.select('#setup-container')
    .remove();

  d3.select('.navbar-top')
    .style('display', 'block');
  
    // Navbar functionality
    d3.select('.navbar-top')
      .selectAll('a')
      .on('click', function () {
        tabId = d3.select(this).attr('id');   
        if (tabId !== currentTabId) {
          d3.selectAll('.dropbtn').classed('btn-active', false);
          d3.select(`#${currentTabId}`).classed('btn-active', false);
          d3.select(`#${tabId}`).classed('btn-active', true);
  
          if (d3.select(this.parentNode).classed('dropdown-content')) {
            d3.select(this.parentNode.parentNode).select('.dropbtn').classed('btn-active', true);
          }
  
          if (currentTabId.endsWith('-cm-button')) {
            d3.select('#confusion-tables-container').node().scrollTop = 0;
            d3.select('#confusion-tables-container')
              .style('display', 'none');
          }
  
          d3.select(`#${tabToContainerDict[currentTabId]}`)
            .style('display', 'none');
  
          if (tabId.endsWith('-cm-button')) {
            d3.select('#confusion-tables-container')
              .style('display', 'block');
          }
  
          d3.select(`#${tabToContainerDict[tabId]}`)
            .style('display', 'block');

          // generate comments AFTER text appears (to get positions of spans for comments)
          if (tabId === 'text-button' && d3.selectAll('rect').size() === 0) {
            textLib.generateComments();
          }
  
          log(`switched to ${tabId}`);
  
          currentTabId = tabId;
        }
      });

  d3.select('#text-container')
    .style('display', 'block');

  textBackend.get_text(false);
};


const onTextData = function (data) {
  textLib.loadText(data, () => {
    log('setup finished, text finished loading');

    if (threadStartId === 'text-button') {
      codesTableBackend.get_table();
      allTableBackend.get_table(false); 
      predictTableBackend.get_table(false);
      trainTableBackend.get_table(false);
      confusionTablesBackend.get_data();
    } 
  });
};


const onCodesTableData = function (data) {
  codesTableLib.loadTable(data, () => {
    log('codes finished loading');
  });
};


const onAllTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    allTableLib.loadReclassifiedTable(data, () => {
      log('reclassified all table finished loading');
      if (threadStartId === 'all-keywords-button') {
        predictTableBackend.get_table(true);
        trainTableBackend.get_table(true);
        textBackend.get_text(true);
        confusionTablesBackend.get_data();
      } 
    });
  } else {
    allTableLib.loadTable(data, () => {
      log('all table finished loading');
    });
  }
};


const onPredictTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    predictTableLib.loadReclassifiedTable(data, () => {
      log('reclassified predict table finished loading');
      if (threadStartId === 'predict-keywords-button') {
        allTableBackend.get_table(true);
        trainTableBackend.get_table(true);
        textBackend.get_text(true);
        confusionTablesBackend.get_data();
      } 
    });
  } else {
    predictTableLib.loadTable(data, () => {
      log('predict table finished loading');
    });
  }
};


const onTrainTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    trainTableLib.loadReclassifiedTable(data, () => {
      log('reclassified train table finished loading');
      if (threadStartId === 'train-keywords-button') {
        allTableBackend.get_table(true);
        predictTableBackend.get_table(true);
        textBackend.get_text(true);
        confusionTablesBackend.get_data();
      }
    });
  } else {
    trainTableLib.loadTable(data, () => {
      log('train table finished loading');
    });
  }
};


const onReclassified = function () {
  console.log('onReclassified!');
  console.log(`threadStartId = ${threadStartId}`);

  switch (threadStartId) {
    case 'all-keywords-button':
      console.log('=======================================================');
      console.log('calling all_table thread first');
      allTableBackend.get_table(true);
      break;
    case 'predict-keywords-button':
      predictTableBackend.get_table(true);
      break;
    case 'train-keywords-button':
      trainTableBackend.get_table(true);
      break;
    default:
      break;
  }
};


const onConfusionTablesData = function (data) {
  confusionTablesLib.loadTables(data, () => {
    log('confusion tables finished loading');
  });
  if (tabId.endsWith('-cm-button')) {
    d3.select(`#${tabToContainerDict[tabId]}`)
      .style('display', 'block');
  }
};


d3.select(window).on('load', () => {  
  importLib.setupImportPage();

  // set up qt web channel
  try {
    if (qt !== undefined) {
      new QWebChannel(qt.webChannelTransport, (channel) => {
        setupBackend = channel.objects.setupBackend;
        textBackend = channel.objects.textBackend;
        codesTableBackend = channel.objects.codesTableBackend;
        allTableBackend = channel.objects.allTableBackend;
        predictTableBackend = channel.objects.predictTableBackend;
        trainTableBackend = channel.objects.trainTableBackend;
        reclassifyBackend = channel.objects.reclassifyBackend;
        confusionTablesBackend = channel.objects.confusionTablesBackend;
        logBackend = channel.objects.logBackend;
        importBackend = channel.objects.importBackend;
        // connect signals from the external object to callback functions
        setupBackend.signal.connect(onSetup);
        textBackend.signal.connect(onTextData);
        codesTableBackend.signal.connect(onCodesTableData);
        allTableBackend.signal.connect(onAllTableData);
        predictTableBackend.signal.connect(onPredictTableData);
        trainTableBackend.signal.connect(onTrainTableData);
        reclassifyBackend.signal.connect(onReclassified);
        confusionTablesBackend.signal.connect(onConfusionTablesData);
        importBackend.signal.connect(onImportData);
        // call functions on the external objects
        // textBackend.get_text(false); // false-> not reclassified data
        log('app launched');
      });
    }
  } catch (error) {
    console.log('something went wrong when setting up the Qt connection');
    console.log(error);
    // fetch('/get_cm_json')
    //   .then((res) => res.json().then((data) => {
    //     onConfusionTablesData(data);
    //   }));
  }
});
