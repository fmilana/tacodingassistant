/* global d3 document $ textBackend logBackend */
// eslint-disable-next-line no-unused-vars
const importLib = (function () {
  const setupImportPage = function () {
    let transcriptCheckboxValue = false;
    let codesCheckboxValue = false;

    // d3.select('#transcript-chooser')
    //   .on('change', () => {
    //     console.log('change!!');
    //     console.log(JSON.stringify(this));
    //     console.log(`this.files = ${this.files}`);
    //     console.log(`this.files[0] = ${this.files[0]}`);
    //   });

    document.getElementById('transcript-chooser')
      .addEventListener('change', () => {
        const file = $('#transcript-chooser')[0].files[0];
        // continue
      }, false);

    d3.select('#transcript-checkbox')
      .on('click', () => {
        transcriptCheckboxValue = !transcriptCheckboxValue;
        if (transcriptCheckboxValue) {
          d3.select('#codes-label')
            .style('color', '#BFBFBF');
          d3.select('#codes-chooser')
            .property('disabled', true);
          d3.select('#codes-checkbox')
            .attr('disabled', 'disabled');
          d3.select('#codes-checkbox-label')
            .style('color', '#BFBFBF');

          if (codesCheckboxValue) {
            d3.select('#theme-code-label')
            .style('color', '#000000');
            d3.select('#theme-code-chooser')
              .attr('disabled', null);
          }
        } else {
          d3.select('#codes-label')
            .style('color', '#000000');
          d3.select('#codes-chooser')
            .property('disabled', false);
          d3.select('#codes-checkbox')
            .attr('disabled', null);
          d3.select('#codes-checkbox-label')
          .style('color', '#000000');

          if (codesCheckboxValue) {
            d3.select('#theme-code-label')
              .style('color', '#BFBFBF');
            d3.select('#theme-code-chooser')
              .property('disabled', true);
          }
        }
      });

    d3.select('#codes-checkbox')
      .on('click', () => {
        codesCheckboxValue = !codesCheckboxValue;
        if (codesCheckboxValue) {
          d3.select('#theme-code-label')
            .style('color', '#BFBFBF');
          d3.select('#theme-code-chooser')
            .property('disabled', true);
        } else {
          d3.select('#theme-code-label')
            .style('color', '#000000');
          d3.select('#theme-code-chooser')
            .attr('disabled', null);
        }
      });

    // if (d3.select('#import-container').empty()) {
    //   console.log('import container selection empty');
    // } else {
    //   console.log('import container not empty');
    // }

    d3.select('#finish-import-button')
      .on('click', function () {
        let logData = '';

        if (transcriptCheckboxValue) {
          logData = `[${new Date().getTime()}]: import page closed with document containing codes`;
        } else if (codesCheckboxValue) {
          logData = `[${new Date().getTime()}]: import page closed with hierarchical codes documents`;
        } else {
          logData = `[${new Date().getTime()}]: import page closed with codes documents and a theme-code table`;
        }

        logData += `. Filtered keywords: "${$('#filter-textarea').val()}"`;

        logBackend.log(logData);

        // run scripts on imported files
        // + add themes to navbar (cm's)
        d3.select('#import-container')
          .remove();

        d3.select('.navbar-top')
          .style('display', 'block');

        d3.select('#text-container')
          .style('display', 'block');

        textBackend.get_text(false);
      });
  };

  return { setupImportPage };
}());
