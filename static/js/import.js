/* global d3 document $ textBackend log writeFileBackend importBackend transcriptPath codesPath */
// eslint-disable-next-line no-unused-vars

let wordDelimiter = '';

const importLib = (function () {
  const setupImportPage = function () {
    let noWordDelimiterCheckBoxValue = false; 
    let editedCheckBoxValue = false;
    let wordCheckboxValue = false;
    let nvivoCheckboxValue = false;
    // let yesHierarchicalCheckboxValue = false;
    // let noHierarchicalCheckboxValue = false;

    const containersStack = [];

    d3.select('#import-transcript-button')
      .on('click', () => {
        importBackend.open_transcript_chooser();
      });

    d3.select('#import-theme-code-table-button')
      .on('click', () => {
        importBackend.open_theme_code_table_chooser();
      });

    d3.select('#import-codes-folder-button')
      .on('click', () => {
        importBackend.open_codes_chooser();
      });

    //next-buttons//
    d3.select('#import-transcript-next-button')
      .on('click', () => {
        d3.select('#import-transcript-container')
          .style('display', 'none');
        d3.select('#import-word-or-nvivo-container')
          .style('display', 'block');
        containersStack.push('#import-transcript-container');
      });

    d3.select('#import-word-or-nvivo-next-button')
      .on('click', () => {
        let divToShow = '';
        if (wordCheckboxValue) {
          // divToShow = '#import-loading-code-theme-table-container';
          divToShow = "#import-word-delimiter-container";
          codesPath = '';
        } else {
          divToShow = '#import-codes-folder-container';
        }
        d3.select('#import-word-or-nvivo-container')
          .style('display', 'none');
        d3.select(divToShow)
          .style('display', 'block');
        containersStack.push('#import-word-or-nvivo-container');
      });

    d3.select('#import-word-delimiter-next-button')
      .on('click', () => {
        // save delimiter
        if (!noWordDelimiterCheckBoxValue) {
          window.wordDelimiter = d3.select('#delimiter-textarea').node().value;
        } else {
          window.wordDelimiter = '';
        }

        d3.select('#import-word-delimiter-container')
          .style('display', 'none');
          d3.select('#import-loading-code-theme-table-container')
          .style('display', 'block');

        importBackend.create_code_table_csv_from_document(transcriptPath, window.wordDelimiter);
        
        containersStack.push('#import-word-delimiter-container');
      });

    d3.select('#import-codes-folder-next-button')
      .on('click', () => {
        d3.select('#import-codes-folder-container')
          .style('display', 'none');
        d3.select('#import-loading-code-theme-table-container')
          .style('display', 'block');

        importBackend.create_code_table_csv_from_folder(transcriptPath, codesPath);
        
        containersStack.push('#import-codes-folder-container');
      });

    // d3.select('#import-theme-code-table-next-button')
    //   .on('click', () => {
    //     d3.select('#import-theme-code-table-container')
    //       .style('display', 'none');
    //     d3.select('#import-keywords-container')
    //       .style('display', 'block');
    //     containersStack.push('#import-theme-code-table-container');
    //   });

    // d3.select('#import-hierarchical-next-button')
    //   .on('click', () => {
    //     let divToShow = '';
    //     if (yesHierarchicalCheckboxValue) {
    //       divToShow = '#import-keywords-container';
    //     } else {
    //       divToShow = '#import-theme-code-table-container';
    //     }
    //     d3.select('#import-hierarchical-container')
    //       .style('display', 'none');
    //     d3.select(divToShow)
    //       .style('display', 'block');
    //     containersStack.push('#import-hierarchical-container');
    //   });
    // ////////////////

    d3.select('#delimiter-textarea')
      .on('input', () => {
        if (d3.select('#delimiter-textarea').node().value.length > 0) {
          d3.select('#import-word-delimiter-next-button')
            .property('disabled', false);
        } else {
          d3.select('#import-word-delimiter-next-button')
            .property('disabled', true);
        }
      });

    d3.select('#no-word-delimiter-checkbox')
      .on('click', () => {
        noWordDelimiterCheckBoxValue = !noWordDelimiterCheckBoxValue;
        if (noWordDelimiterCheckBoxValue) {
          d3.select('#delimiter-textarea')
            .property('value', '')
            .property('disabled', true);
          d3.select('#import-word-delimiter-next-button')
            .property('disabled', false);
        } else {
          d3.select('#delimiter-textarea')
            .property('disabled', false);
          d3.select('#import-word-delimiter-next-button')
            .property('disabled', true);
        }
        d3.select('#delimiter-textarea')
          .property('disabled', noWordDelimiterCheckBoxValue);
      });

    d3.select('#edit-code-theme-table-checkbox')
      .on('click', () => {
        editedCheckBoxValue = !editedCheckBoxValue;
        d3.select('#import-edit-code-theme-table-next-button')
          .property('disabled', !editedCheckBoxValue);
      });

    d3.select('#import-edit-code-theme-table-next-button')
      .on('click', () => {
        d3.select('#import-edit-code-theme-table-container')
          .style('display', 'none');
        d3.select('#import-keywords-container')
          .style('display', 'block');
        containersStack.push('#import-edit-code-theme-table-container');

        if (containersStack.includes('#import-word-delimiter-container')) {
          d3.select('.dynamic-stepper')
            .text('Enter delimiter');
        } else {
          d3.select('.dynamic-stepper')
            .text('Select codes folder');
        }

        // d3.select('#dynamic-stepper')
        //   .text(function(d) {
        //     if (containersStack.includes('#import-word-delimiter-container')) {
        //       console.log('RETURNING Enter delimiter');
        //       return 'Enter delimiter';
        //     } else {
        //       console.log('RETURNING Select codes folder');
        //       return 'Select codes folder';
        //     }
        //   });
      });

    //back-buttons//
    d3.select('#import-word-or-nvivo-back-button')
      .on('click', () => {
        d3.select('#import-word-or-nvivo-container')
          .style('display', 'none');
        d3.select(containersStack.pop())
          .style('display', 'block');
      });

    d3.select('#import-word-delimiter-back-button')
      .on('click', () => {
        d3.select('#import-word-delimiter-container')
          .style('display', 'none');
        d3.select(containersStack.pop())
          .style('display', 'block');  
      })

    // d3.select('#import-theme-code-table-back-button')
    //   .on('click', () => {
    //     d3.select('#import-theme-code-table-container')
    //       .style('display', 'none');
    //     d3.select(containersStack.pop())
    //       .style('display', 'block');
    //   });

    d3.select('#import-edit-code-theme-table-back-button')
      .on('click', () => {
        d3.select('#import-edit-code-theme-table-container')
          .style('display', 'none');
        d3.select(containersStack.pop())
          .style('display', 'block');
      });

    d3.select('#import-codes-folder-back-button')
      .on('click', () => {
        d3.select('#import-codes-folder-container')
          .style('display', 'none');
        d3.select(containersStack.pop())
          .style('display', 'block');
      });

    // d3.select('#import-hierarchical-back-button')
    //   .on('click', () => {
    //     d3.select('#import-hierarchical-container')
    //       .style('display', 'none');
    //     d3.select(containersStack.pop())
    //       .style('display', 'block');
    //   });

    d3.select('#import-keywords-back-button')
      .on('click', () => {
        d3.select('#import-keywords-container')
          .style('display', 'none');
        d3.select(containersStack.pop())
          .style('display', 'block');
      });
    //////////////

    d3.select('#import-word-checkbox')
      .on('click', () => {
        wordCheckboxValue = !wordCheckboxValue;
        if (wordCheckboxValue && nvivoCheckboxValue) {
          d3.select('#import-nvivo-checkbox')
            .property('checked', false);
          nvivoCheckboxValue = false;
        }
        d3.select('#import-word-or-nvivo-next-button')
          .property('disabled', (!wordCheckboxValue && !nvivoCheckboxValue));
      });

    d3.select('#import-nvivo-checkbox')
      .on('click', () => {
        nvivoCheckboxValue = !nvivoCheckboxValue;
        if (nvivoCheckboxValue && wordCheckboxValue) {
          d3.select('#import-word-checkbox')
            .property('checked', false);
          wordCheckboxValue = false;
        }
        d3.select('#import-word-or-nvivo-next-button')
          .property('disabled', (!wordCheckboxValue && !nvivoCheckboxValue));
      });

    // d3.select('#yes-checkbox')
    //   .on('click', () => {
    //     yesHierarchicalCheckboxValue = !yesHierarchicalCheckboxValue;
    //     if (yesHierarchicalCheckboxValue && noHierarchicalCheckboxValue) {
    //       d3.select('#no-checkbox')
    //         .property('checked', false);
    //       noHierarchicalCheckboxValue = false;
    //     }
    //     d3.select('#import-hierarchical-next-button')
    //       .property('disabled', (!yesHierarchicalCheckboxValue && !noHierarchicalCheckboxValue));
    //   });

    // d3.select('#no-checkbox')
    //   .on('click', () => {
    //     noHierarchicalCheckboxValue = !noHierarchicalCheckboxValue;
    //     if (noHierarchicalCheckboxValue && yesHierarchicalCheckboxValue) {
    //       d3.select('#yes-checkbox')
    //         .property('checked', false);
    //       yesHierarchicalCheckboxValue = false;
    //     }
    //     d3.select('#import-hierarchical-next-button')
    //       .property('disabled', (!noHierarchicalCheckboxValue && !yesHierarchicalCheckboxValue));
    //   });

    d3.select('#import-keywords-next-button')
      .on('click', function () {
        let logString = '';

        if (wordCheckboxValue) {
          logString = 'import document containing codes from Word';
        } 
        // else if (nvivoCheckboxValue && yesHierarchicalCheckboxValue) {
        //   logString = 'import document and hierarchical codes from NVivo';
        // } 
        else {
          logString = 'import document, codes from NVivo and theme-code lookup table';
        }

        const filterKeywords = $('#filter-textarea').val();
        const regularExpression = $('#regexp-checkbox').is(':checked');
        const caseInsensitive = $('#case-insensitive-checkbox').is(':checked');

        log(logString);

        // console.log(`typeof transcriptFile = ${transcriptFile}`);
        // console.log(`filename = ${transcriptFile.name}`);

        // writeFileBackend.write_file(transcriptFile);

        importBackend.save_regexp(filterKeywords, regularExpression, caseInsensitive);
      });
  };

  return { setupImportPage };
}());
