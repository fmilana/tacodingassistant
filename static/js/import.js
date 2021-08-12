/* global d3 document $ textBackend logBackend writeFileBackend importBackend fileChooserBackend */
// eslint-disable-next-line no-unused-vars
const importLib = (function () {
  const setupImportPage = function () {
    let transcriptCheckboxValue = false;
    let codesCheckboxValue = false;

    let transcriptFile = null;

    // document.getElementById('transcript-chooser')
    //   .addEventListener('change', () => {
    //     const file = $('#transcript-chooser')[0].files[0];
    //     // writeFileBackend.write_file(file);
    //     transcriptFile = file;
    //     // continue
    //   }, false);

    d3.select('#transcript-chooser')
      .on('click', () => {
        fileChooserBackend.open_transcript_chooser();
      });

    d3.select('#codes-chooser')
      .on('click', () => {
        fileChooserBackend.open_codes_chooser();
      });

    d3.select('#theme-code-chooser')
      .on('click', () => {
        fileChooserBackend.open_theme_code_table_chooser();
      });

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
              .property('disabled', false);
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
            .property('disabled', false);
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
          logData = `[${new Date().getTime()}]: import document containing codes`;
        } else if (codesCheckboxValue) {
          logData = `[${new Date().getTime()}]: import hierarchical codes documents`;
        } else {
          logData = `[${new Date().getTime()}]: import codes documents and a theme-code table`;
        }

        const filterKeywords = $('#filter-textarea').val();
        const regularExpression = $('#regexp-checkbox').is(':checked');
        const caseInsensitive = $('#case-insensitive-checkbox').is(':checked');

        logData += `. Filtered keywords: "${filterKeywords}"`;

        logBackend.log(logData);

        // console.log(`typeof transcriptFile = ${transcriptFile}`);
        // console.log(`filename = ${transcriptFile.name}`);

        // writeFileBackend.write_file(transcriptFile);

        importBackend.save_regexp(filterKeywords, regularExpression, caseInsensitive);
      });
  };

  return { setupImportPage };
}());
