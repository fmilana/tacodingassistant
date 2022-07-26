/* global d3 document $ textBackend log writeFileBackend importBackend transcriptPath nvivoCodesPath */
// eslint-disable-next-line no-unused-vars

let wordDelimiter = '';

const importLib = (function () {
  const setupImportPage = function () {
    let noWordDelimiterCheckBoxValue = false; 
    let editedCheckBoxValue = false;
    let wordCheckboxValue = false;
    let nvivoCheckboxValue = false;
    let maxqdaCheckboxValue = false;
    let dedooseCheckboxValue = false;

    const containersStack = [];

    d3.select('#import-transcript-button')
      .on('click', () => {
        importBackend.open_transcript_chooser();
      });

    d3.select('#import-theme-code-table-button')
      .on('click', () => {
        importBackend.open_theme_code_table_chooser();
      });

    d3.select('#import-nvivo-codes-folder-button')
      .on('click', () => {
        importBackend.open_nvivo_codes_chooser();
      });

    d3.select('#import-maxqda-document-button')
      .on('click', () => {
        importBackend.open_maxqda_document_chooser();
      })

    //next-buttons//
    d3.select('#import-transcript-next-button')
      .on('click', () => {
        d3.select('#import-transcript-container')
          .style('display', 'none');
        d3.select('#import-software-container')
          .style('display', 'block');
        containersStack.push('#import-transcript-container');
      });

    d3.select('#import-software-next-button')
      .on('click', () => {
        let divToShow = '';
        if (wordCheckboxValue) {
          // divToShow = '#import-loading-code-theme-table-container';
          divToShow = "#import-word-delimiter-container";
          nvivoCodesPath = '';
        } else if (nvivoCheckboxValue) {
          divToShow = '#import-nvivo-codes-folder-container';
        } else if (maxqdaCheckboxValue) {
          divToShow = '#import-maxqda-document-container';
        }
        d3.select('#import-software-container')
          .style('display', 'none');
        d3.select(divToShow)
          .style('display', 'block');
        containersStack.push('#import-software-container');
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

        importBackend.create_code_table_csv_from_word(transcriptPath, window.wordDelimiter);

        software = 'Word';
        
        containersStack.push('#import-word-delimiter-container');
      });

    d3.select('#import-nvivo-codes-folder-next-button')
      .on('click', () => {
        d3.select('#import-nvivo-codes-folder-container')
          .style('display', 'none');
        d3.select('#import-loading-code-theme-table-container')
          .style('display', 'block');

        importBackend.create_code_table_csv_from_nvivo(transcriptPath, nvivoCodesPath);

        software = 'NVivo';
        
        containersStack.push('#import-nvivo-codes-folder-container');
      });

    d3.select('#import-maxqda-document-next-button')
      .on('click', () => {
        d3.select('#import-maxqda-document-container')
          .style('display', 'none');
        d3.select('#import-loading-code-theme-table-container')
          .style('display', 'block');

        importBackend.create_code_table_csv_from_maxqda(transcriptPath, MAXQDADocumentPath);

        software = 'MAXQDA';
        
        containersStack.push('#import-maxqda-document-container');
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
        } else if (containersStack.includes('#import-nvivo-codes-folder-container')) {
          d3.select('.dynamic-stepper')
            .text('Select codes folder');
        } else if (containersStack.includes('#import-maxqda-document-container')) {
          d3.select('.dynamic-stepper')
            .text('Import coded segments');
        }
      });

    //back-buttons//
    d3.select('#import-software-back-button')
      .on('click', () => {
        d3.select('#import-software-container')
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

    d3.select('#import-nvivo-codes-folder-back-button')
      .on('click', () => {
        d3.select('#import-nvivo-codes-folder-container')
          .style('display', 'none');
        d3.select(containersStack.pop())
          .style('display', 'block');
      });

    d3.select('#import-maxqda-document-back-button')
      .on('click', () => {
        d3.select('#import-maxqda-document-container')
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

        if (wordCheckboxValue) {
          if (nvivoCheckboxValue) {
            d3.select('#import-nvivo-checkbox')
              .property('checked', false);
            nvivoCheckboxValue = false;
          } else if (maxqdaCheckboxValue) {
            d3.select('#import-maxqda-checkbox')
              .property('checked', false);
            maxqdaCheckboxValue = false;
          } else if (dedooseCheckboxValue) {
            d3.select('#import-dedoose-checkbox')
              .property('checked', false);
            dedooseCheckboxValue = false;
          }
        } else {
          d3.select('#import-word-checkbox')
            .property('checked', false);
        }

        d3.select('#import-software-next-button')
          .property('disabled', (!wordCheckboxValue && !nvivoCheckboxValue && !maxqdaCheckboxValue && !dedooseCheckboxValue));
      });

    d3.select('#import-nvivo-checkbox')
      .on('click', () => {
        nvivoCheckboxValue = !nvivoCheckboxValue;

        if (nvivoCheckboxValue) {
          if (wordCheckboxValue) {
            d3.select('#import-word-checkbox')
              .property('checked', false);
            wordCheckboxValue = false;
          } else if (maxqdaCheckboxValue) {
            d3.select('#import-maxqda-checkbox')
              .property('checked', false);
            maxqdaCheckboxValue = false;
          } else if (dedooseCheckboxValue) {
            d3.select('#import-dedoose-checkbox')
              .property('checked', false);
            dedooseCheckboxValue = false;
          }
        } else {
          d3.select('#import-nvivo-checkbox')
            .property('checked', false);
        }

        d3.select('#import-software-next-button')
          .property('disabled', (!wordCheckboxValue && !nvivoCheckboxValue && !maxqdaCheckboxValue && !dedooseCheckboxValue));
      });

    d3.select('#import-maxqda-checkbox')
      .on('click', () => {
        maxqdaCheckboxValue = !maxqdaCheckboxValue;

        if (maxqdaCheckboxValue) {
          if (wordCheckboxValue) {
            d3.select('#import-word-checkbox')
              .property('checked', false);
            wordCheckboxValue = false;
          } else if (nvivoCheckboxValue) {
            d3.select('#import-nvivo-checkbox')
              .property('checked', false);
            nvivoCheckboxValue = false;
          } else if (dedooseCheckboxValue) {
            d3.select('#import-dedoose-checkbox')
              .property('checked', false);
            dedooseCheckboxValue = false;
          }
        } else {
          d3.select('#import-maxqda-checkbox')
            .property('checked', false);
        }

        d3.select('#import-software-next-button')
          .property('disabled', (!wordCheckboxValue && !nvivoCheckboxValue && !maxqdaCheckboxValue && !dedooseCheckboxValue));
      });

    d3.select('#import-dedoose-checkbox')
      .on('click', () => {
        dedooseCheckboxValue = !dedooseCheckboxValue;

        if (dedooseCheckboxValue) {
          if (wordCheckboxValue) {
            d3.select('#import-word-checkbox')
              .property('checked', false);
            wordCheckboxValue = false;
          } else if (nvivoCheckboxValue) {
            d3.select('#import-nvivo-checkbox')
              .property('checked', false);
            nvivoCheckboxValue = false;
          } else if (maxqdaCheckboxValue) {
            d3.select('#import-maxqda-checkbox')
              .property('checked', false);
            maxqdaCheckboxValue = false;
          }
        } else {
          d3.select('#import-dedoose-checkbox')
            .property('checked', false);
        }

        d3.select('#import-software-next-button')
          .property('disabled', (!wordCheckboxValue && !nvivoCheckboxValue && !maxqdaCheckboxValue && !dedooseCheckboxValue));
      });

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
