var get_data = function() {
  fetch('/get_data')
  .then(function(res) {
    return res.json().then(function(data) {

      var columns = [
        'practices',
        'social',
        'study vs product',
        'system perception',
        'system use',
        'value judgements',
      ];

  		var table = d3.select('body').append('table').attr('class', 'center');
  		var thead = table.append('thead');
  		var	tbody = table.append('tbody');

  		// append the header row
  		thead.append('tr')
  		  .selectAll('th')
  		  .data(columns).enter()
  		  .append('th')
  		    .text(function (column) {
            return column;
          });

  		// create a row for each object in the data
  		var rows = tbody.selectAll('tr')
  		  .data(data)
  		  .enter()
  		  .append('tr');

  		// create a cell in each row for each column
  		var cells = rows.selectAll('td')
  		  .data(function (row) {
  		    return columns.map(function (column) {
  		      return {column: column, value: row[column]};
  		    });
  		  })
  		  .enter()
  		  .append('td')
  		    .text(function (d) {
            return d.value;
          });

    });
  });
}

get_data();
