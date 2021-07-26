/* global d3 window qt QWebChannel textLib trainTableLib predictTableLib allTableLib $*/

let textBackend;
let allTableBackend;
let predictTableBackend;
let trainTableBackend;
let reclassifyBackend;

let currentTabId = 'text-button';


const onTextData = function (data) {
  textLib.loadText(data);
};


const onAllTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    allTableLib.loadReclassifiedTable(data);
  } else {
    allTableLib.loadTable(data);
  }
};


const onPredictTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    predictTableLib.loadReclassifiedTable(data);
  } else {
    predictTableLib.loadTable(data);
  }
};


const onTrainTableData = function (dataAndReclassified) {
  const data = dataAndReclassified[0];
  const reclassified = dataAndReclassified[1];

  if (reclassified) {
    trainTableLib.loadReclassifiedTable(data);
  } else {
    trainTableLib.loadTable(data);
  }
};


const onReclassified = function () {
  allTableBackend.get_table(true); // true-> reclassified
  predictTableBackend.get_table(true);
  trainTableBackend.get_table(true);
};


d3.select(window).on('load', () => {  
  const tabToContainerDict = {
    'text-button': 'text-container',
    'all-keywords-button': 'all-table-container',
    'predict-keywords-button': 'predict-table-container',
    'train-keywords-button': 'train-table-container'
  };

  // Navbar functionality
  d3.select('.navbar-top')
    .selectAll('a')
    .on('click', function () {
      const tabId = d3.select(this).attr('id');   
      if (tabId !== currentTabId) {
        d3.selectAll('.dropbtn').classed('btn-active', false);
        d3.select(`#${currentTabId}`).classed('btn-active', false);
        d3.select(`#${tabId}`).classed('btn-active', true);

        if (d3.select(this.parentNode).classed('dropdown-content')) {
          d3.select(this.parentNode.parentNode).select('.dropbtn').classed('btn-active', true);
        }

        d3.select(`#${tabToContainerDict[currentTabId]}`)
          .style('display', 'none');

        d3.select(`#${tabToContainerDict[tabId]}`)
          .style('display', 'block');

        currentTabId = tabId;
      }
    });

  // set up qt web channel
  try {
    if (qt !== undefined) {
      new QWebChannel(qt.webChannelTransport, (channel) => {
        textBackend = channel.objects.textBackend;
        allTableBackend = channel.objects.allTableBackend;
        predictTableBackend = channel.objects.predictTableBackend;
        trainTableBackend = channel.objects.trainTableBackend;
        reclassifyBackend = channel.objects.reclassifyBackend;
        // connect signals from the external object to callback functions
        textBackend.signal.connect(onTextData);
        allTableBackend.signal.connect(onAllTableData);
        predictTableBackend.signal.connect(onPredictTableData);
        trainTableBackend.signal.connect(onTrainTableData);
        reclassifyBackend.signal.connect(onReclassified);
        // call functions on the external objects
        textBackend.get_text();
        allTableBackend.get_table(false); // false-> not reclassified
        predictTableBackend.get_table(false);
        trainTableBackend.get_table(false);
      });
    }
  } catch (error) {
    console.log('something went wrong when setting up the Qt connection');
    console.log(error);
  }
});
