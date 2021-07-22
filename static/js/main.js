/* global d3 window qt QWebChannel textLib tableLib $*/

let textBackend;
let tableBackend;

let currentTabId = 'text-button';


const onTextData = function (data) {
  textLib.loadText(data);
};


const onTableData = function (data) {
  tableLib.loadTable('keywords', data);
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
          const tableContainer = d3.select('#table-container');
          if (tableContainer.empty()) {
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
        // connect signals from the external object to callback functions
        textBackend.signal.connect(onTextData);
        tableBackend.signal.connect(onTableData);
        // call a function on the external object
        textBackend.get_text();
      });
    }
  } catch (error) {
    console.log('something went wrong when setting up the Qt connection');
    console.log(error);
  }
});
