/* global d3 window qt QWebChannel importLib textLib codesTableLib trainTableLib predictTableLib allTableLib confusionTablesLib */

// hard-coded
const themes = [
  'practices',
  'social',
  'study vs product',
  'system use',
  'system perception',
  'value judgements'
];

const tabToContainerDict = {
  'text-button': 'text-container',
  'codes-button': 'codes-table-container',
  'all-keywords-button': 'all-table-container',
  'predict-keywords-button': 'predict-table-container',
  'train-keywords-button': 'train-table-container'
};

let textBackend;
let codesTableBackend;
let allTableBackend;
let predictTableBackend;
let trainTableBackend;
let reclassifyBackend;
let confusionTablesBackend;
let logBackend;

let threadStartId = 'text-button'; // which page started threads
let tabId = 'text-button';
let currentTabId = 'text-button';


const onTextData = function (data) {
  textLib.loadText(data, () => {
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
  codesTableLib.loadTable(data);
};


const onAllTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    allTableLib.loadReclassifiedTable(data, () => {
      if (threadStartId === 'all-keywords-button') {
        predictTableBackend.get_table(true);
        trainTableBackend.get_table(true);
        textBackend.get_text(true);
        confusionTablesBackend.get_data();
      } 
    });
  } else {
    allTableLib.loadTable(data);
  }
};


const onPredictTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    predictTableLib.loadReclassifiedTable(data, () => {
      if (threadStartId === 'predict-keywords-button') {
        allTableBackend.get_table(true);
        trainTableBackend.get_table(true);
        textBackend.get_text(true);
        confusionTablesBackend.get_data();
      } 
    });
  } else {
    predictTableLib.loadTable(data);
  }
};


const onTrainTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    trainTableLib.loadReclassifiedTable(data, () => {
      if (threadStartId === 'train-keywords-button') {
        allTableBackend.get_table(true);
        predictTableBackend.get_table(true);
        textBackend.get_text(true);
        confusionTablesBackend.get_data();
      }
    });
  } else {
    trainTableLib.loadTable(data);
  }
};


const onReclassified = function () {
  console.log('onReclassified!');
  console.log(`threadStartId = ${threadStartId}`);

  switch (threadStartId) {
    case 'all-keywords-button':
      console.log('=======================================================');
      console.log('calling ALLTABLE THREAD FIRST!!');
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
  confusionTablesLib.loadTables(data);
  if (tabId.endsWith('-cm-button')) {
    d3.select(`#${tabToContainerDict[tabId]}`)
      .style('display', 'block');
  }
};


d3.select(window).on('load', () => {  
  importLib.setupImportPage();

  for (let i = 0; i < themes.length; i++) {
    tabToContainerDict[`${themes[i].replace(/ /g, '-')}-cm-button`] = `${themes[i].replace(/ /g, '-')}-table-container`;
  }

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

        logBackend.log(`[${new Date().getTime()}]: switched to ${tabId}`);

        currentTabId = tabId;
      }
    });

  // set up qt web channel
  try {
    if (qt !== undefined) {
      new QWebChannel(qt.webChannelTransport, (channel) => {
        textBackend = channel.objects.textBackend;
        codesTableBackend = channel.objects.codesTableBackend;
        allTableBackend = channel.objects.allTableBackend;
        predictTableBackend = channel.objects.predictTableBackend;
        trainTableBackend = channel.objects.trainTableBackend;
        reclassifyBackend = channel.objects.reclassifyBackend;
        confusionTablesBackend = channel.objects.confusionTablesBackend;
        logBackend = channel.objects.logBackend;
        // connect signals from the external object to callback functions
        textBackend.signal.connect(onTextData);
        codesTableBackend.signal.connect(onCodesTableData);
        allTableBackend.signal.connect(onAllTableData);
        predictTableBackend.signal.connect(onPredictTableData);
        trainTableBackend.signal.connect(onTrainTableData);
        reclassifyBackend.signal.connect(onReclassified);
        confusionTablesBackend.signal.connect(onConfusionTablesData);
        // call functions on the external objects
        // textBackend.get_text(false); // false-> not reclassified data
        logBackend.log(`[${new Date().getTime()}]: app launched`);
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
