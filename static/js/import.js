/* global d3 document $ textBackend log writeFileBackend importBackend transcriptPath nvivoCodesPath */
// eslint-disable-next-line no-unused-vars

let wordDelimiter = '';

const importLib = (function () {
  const setupImportPage = function () {
    let noWordDelimiterCheckBoxValue = false;
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
        importBackend.open_maxqda_segments_chooser();
      });
    
    d3.select('#import-dedoose-excerpts-button')
      .on('click', () => {
        importBackend.open_dedoose_excerpts_chooser();
      });

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
        } else if (dedooseCheckboxValue) {
          divToShow = '#import-dedoose-excerpts-container';
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

        importBackend.get_codes_from_word(transcriptPath, window.wordDelimiter);

        software = 'Word';
        
        containersStack.push('#import-word-delimiter-container');
      });

    d3.select('#import-nvivo-codes-folder-next-button')
      .on('click', () => {
        d3.select('#import-nvivo-codes-folder-container')
          .style('display', 'none');
        d3.select('#import-loading-code-theme-table-container')
          .style('display', 'block');

        importBackend.get_codes_from_nvivo(transcriptPath, nvivoCodesPath);

        software = 'NVivo';
        
        containersStack.push('#import-nvivo-codes-folder-container');
      });

    d3.select('#import-maxqda-document-next-button')
      .on('click', () => {
        d3.select('#import-maxqda-document-container')
          .style('display', 'none');
        d3.select('#import-loading-code-theme-table-container')
          .style('display', 'block');

        importBackend.get_codes_from_maxqda(transcriptPath, MAXQDASegmentsPath);

        software = 'MAXQDA';
        
        containersStack.push('#import-maxqda-document-container');
      });

      d3.select('#import-dedoose-excerpts-next-button')
        .on('click', () => {
          d3.select('#import-dedoose-excerpts-container')
            .style('display', 'none');
          d3.select('#import-loading-code-theme-table-container')
            .style('display', 'block');

          importBackend.get_codes_from_dedoose(transcriptPath, dedooseExcerptsPath);

          software = 'Dedoose';
          
          containersStack.push('#import-dedoose-excerpts-container');
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

    //codethemetable functionality
    //https://stackoverflow.com/questions/61492659/how-can-i-drag-and-drop-cell-contents-within-a-table-with-dynamically-added-rows
    $(function() {
      initDragAndDrop();

      // https://stackoverflow.com/questions/14964253/how-to-dynamically-add-a-new-column-to-an-html-table
      $('#add-theme-button').on('click', function() {
        setupThemeCodeColumnCount = setupThemeCodeColumnCount + 1;

        [...$('#theme-code-table tr')].forEach((row, i) => {
          let cell;
          if (i === 0) {
            cell = document.createElement('th');
            cell.innerHTML = `<div contenteditable>Theme ${setupThemeCodeColumnCount}</div>`;
          } else {
            cell = document.createElement('td');
          }

          row.appendChild(cell);

          if (setupThemeCodeColumnCount === 8) {
            $('#add-theme-button').prop('disabled', true);
          }

          if (setupThemeCodeColumnCount === 3) {
            $('#remove-theme-button').prop('disabled', false);
          }
        });

        clearDragAndDrop();
        initDragAndDrop();
      });

      $('#remove-theme-button').on('click', function() {
        setupThemeCodeColumnCount = setupThemeCodeColumnCount - 1;

        $('#theme-code-table tr').find('th:last-child, td:last-child').remove();
        
        if (setupThemeCodeColumnCount === 7) {
          $('#add-theme-button').prop('disabled', false);
        }

        if (setupThemeCodeColumnCount === 2) {
          $('#remove-theme-button').prop('disabled', true);
        }
      });
    });

    const clearDragAndDrop = function () {
      $('.code').off();
      $('#theme-code-table td').off('dragenter dragover drop');
    };

    const initDragAndDrop = function () {
      $('.code').on('dragstart', function(event) {
        var dt = event.originalEvent.dataTransfer;
        dt.setData('Text', $(this).attr('id'));
      });
      $('#theme-code-table td').on('dragenter dragover drop', function(event) {
        event.preventDefault();
        if (event.type === 'drop') {
          var data = event.originalEvent.dataTransfer.getData('Text', $(this).attr('id'));
          de = $('#' + data).detach();
          de.appendTo($(this));
        }
      });
    };
    // ............................. //

    d3.select('#import-edit-code-theme-table-next-button')
      .on('click', () => {
        d3.select('#import-edit-code-theme-table-container')
          .style('display', 'none');
        d3.select('#import-keywords-container')
          .style('display', 'block');
        containersStack.push('#import-edit-code-theme-table-container');

        d3.select('.dynamic-stepper')
          .text(function(d) {
            if (containersStack.includes('#import-word-delimiter-container')) {
              return 'Enter delimiter';
            } else if (containersStack.includes('#import-nvivo-codes-folder-container')) {
              return 'Select codes folder';
            } else if (containersStack.includes('#import-maxqda-document-container')) {
              return 'Import coded segments';
            } else if (containersStack.includes('#import-dedoose-excerpts-container')) {
              return 'Import excerpts';
            }
          });
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

    d3.select('#import-dedoose-excerpts-back-button')
      .on('click', () => {
        d3.select('#import-dedoose-excerpts-container')
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
