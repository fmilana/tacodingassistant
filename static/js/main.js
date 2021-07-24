/* global d3 window qt QWebChannel textLib tableLib $*/

let textBackend;
let tableBackend;
let reclassifyBackend;

let currentTabId = 'text-button';


const onTextData = function (data) {
  textLib.loadText(data);
};


const onTableData = function (data) {
  tableLib.loadTable('keywords', data);
};


const onReclassifyData = function (data) {
  tableLib.loadReclassifiedTable(data); // pass table name?
};


d3.select(window).on('load', () => {  
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

        if (currentTabId === 'text-button') {
          d3.select('#text-container')
            .style('display', 'none');
        } else if (currentTabId === 'all-keywords-button') {
          d3.select('#table-container')
            .style('display', 'none');
        }

        if (tabId === 'text-button') {
          d3.select('#text-container')
            .style('display', 'flex');
        } else if (tabId === 'all-keywords-button') {
          let tableContainer = d3.select('#table-container');
          if (tableContainer.empty()) {
            tableContainer = d3.select('body')
              .append('div')
                .attr('id', 'table-container');

            tableContainer
              .append('button')
                .attr('id', 're-classify-button')
                .attr('type', 'button')
                .property('disabled', true)
                .text('Re-classify');
            
            tableContainer
              .append('h1')
                .attr('id', 'table-title')
                .text('Keywords');

            const binDiv = tableContainer
              .append('div')
                .attr('id', 'bin-div');
            
            binDiv
              .append('h1')
              .attr('id', 'bin-title')
              .text('Bin');
            
            binDiv
              .append('p')
              .text('Drag here to remove theme');

            tableBackend.get_table();

            d3.select('#loading-gif')
              .style('display', 'block');
          } else {
            tableContainer.style('display', 'block');
          }
        }

        currentTabId = tabId;
      }
    });

  // set up qt web channel
  try {
    if (qt !== undefined) {
      new QWebChannel(qt.webChannelTransport, (channel) => {
        textBackend = channel.objects.textBackend;
        tableBackend = channel.objects.tableBackend;
        reclassifyBackend = channel.objects.reclassifyBackend;
        // connect signals from the external object to callback functions
        textBackend.signal.connect(onTextData);
        tableBackend.signal.connect(onTableData);
        reclassifyBackend.signal.connect(onReclassifyData);
        // call a function on the external object
        textBackend.get_text();
      });
    }
  } catch (error) {
    console.log('something went wrong when setting up the Qt connection');
    console.log(error);
  }
});
