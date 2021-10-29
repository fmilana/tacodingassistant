/* global d3 */
const textLib = (function () {
  const loadText = function (data, callback) {
    const startTime = new Date().getTime();

    const text = highlightSentences(data[0], data[1], data[2]);

    const textContainer = d3.select('#text-container');

    textContainer.select('#loading-gif')
      .style('display', 'none');

    const textContainerRow = textContainer
      .append('div')
      .classed('row', true);
    
    textContainerRow
      .append('div')
        .classed('col-9', true)
        .append('div')
          .attr('id', 'text-box');
    
    textContainerRow
      .append('div')
        .classed('col-3', true)
        .append('div')
          .attr('id', 'comments-box');

    // if (d3.select('#comments-box').empty()) {
    //   console.log('')
    // }

    textContainerRow.select('#text-box')
      .selectAll('p')
      .data([text])
      .enter()
      .append('p')
        .attr('id', 'text-paragraph')
        .html(d => d);

    // scroll text container to top after re-classification?
    // textContainerRow.select('#text-box').node().scrollTop = 0;

    if (d3.select('#text-container').style('display') === 'block') {    
      generateComments();
    }

    const endTime = new Date().getTime();
    console.log(`Text (JavaScript) => ${((endTime - startTime) / 1000).toFixed(2)} seconds`);

    console.log('calling callback (other threads)...');
    
    callback();
  };


  const highlightSentences = function (text, trainData, predictData) {
    const mapObj = {};

    for (let i = 0; i < trainData.length; i++) {
      const entry = trainData[i];
      const trainSentence = entry[0];
      const themes = entry[1];
      //better way?
      if (trainSentence.length > 6) {
        mapObj[trainSentence] = `<span data-tooltip="${themes}"` +
        `style="background-color: #dbdbdb">${trainSentence}</span>`;
      }
    }

    for (let i = 0; i < predictData.length; i++) {
      const entry = predictData[i];
      const predictSentence = entry[0];
      const themes = entry[1];
      // better way?
      if (predictSentence.length > 6) {
        mapObj[predictSentence] = `<span data-tooltip="${themes}"` +
        `style="background-color: #a8e5ff">${predictSentence}</span>`;
      }
    }

    // // 2 methods: https://stackoverflow.com/questions/15604140/replace-multiple-strings-with-multiple-other-strings
    //
    // // Regexp method (bugged: 366 undefined's) ------------------------------
    // const escapedMapObjKeys = Object.keys(mapObj)
    //   .map(key => key.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&'));
    // const re = new RegExp(escapedMapObjKeys.join('|'), 'gim');
    //
    // return text.replace(re, (matched) => mapObj[matched]);

    // split-join method -------------------------------------------

    ////////////// -------> to-do: use position

    const entries = Object.entries(mapObj);

    const highlightedText = entries.reduce(
      // replace all the occurrences of the keys in the text into an index placholder using split-join
      (_text, [key], i) => _text.split(key).join(`{${i}}`),
      // manipulate all exisitng index placeholder-like formats, in order to prevent confusion
      text.replace(/\{(?=\d+\})/g, '{-')
    )
    // replace all index placeholders to the desired replacement values
    .replace(/\{(\d+)\}/g, (_, i) => entries[i][1])
    // undo the manipulation of index placeholder -like formats
    .replace(/\{-(?=\d+\})/g, '{');

    return highlightedText;
  };


  const generateComments = function () {
    const commentsObj = [];

    d3.select('#text-container')
      .selectAll('span')
      .each(function () {
        const obj = {
          themes: d3.select(this).attr('data-tooltip'),
          y: this.getBoundingClientRect().y - 50 + d3.select('#text-container').node().scrollTop, // navbar offset (50) + scroll offset
          color: d3.select(this).style('background-color')
        };
        commentsObj.push(obj);
      });

    let lastThemes = '';
    let lastY = 0;
    let lastCanvas;

    let lastColor;

    let stacked = 0;

    Object.keys(commentsObj).forEach((key) => {
      const obj = commentsObj[key];

      let y = obj.y;
      const color = obj.color;
      const themes = obj.themes;

      const skipText = (themes === lastThemes) && ((lastY === (y - 28) || (lastY === (y - 33))));
      const concatCanvas = (y === lastY);

      if (!concatCanvas) {
        const canvas = d3.select('#text-container')
          .select('#comments-box')
          .append('svg')
          .style('position', 'absolute')
          .style('top', () => {
            if (skipText) {
              y -= 5;
              return y.toString().concat('px');
            }
            return y.toString().concat('px');
          })
          .attr('width', '400')
          .append('g');

        // rect
        canvas.append('rect')
          .attr('height', () => {
            if (skipText) {
              return '30';
            }
            return '25';
          })
          .attr('width', '6')
          .attr('x', '0')
          .style('fill', color);

        lastColor = color;

        // text
        canvas.append('text')
          .attr('x', '20')
          .attr('y', '17')
          .text(() => {
            if (!skipText && (y !== lastY)) {
              return themes;
            }
            return '';
          });

        lastCanvas = canvas;
        stacked = 1;
      } else {
        const existingThemes = lastCanvas.select('text').text().split(', ');

        if (color !== lastColor) {
          lastCanvas.append('rect')
            .attr('height', '25')
            .attr('width', '6')
            .attr('x', 10 * stacked)
            .style('fill', color);

          lastColor = color;

          stacked += 1;
        }

        const newThemes = themes.split(', ');

        for (let i = 0; i < newThemes.length; i++) {
          if (!existingThemes.includes(newThemes[i])) {
            existingThemes.push(newThemes[i]);
          }
        }

        const concatThemes = existingThemes.join(', ');

        lastCanvas.select('text')
          .attr('x', 10 + (10 * stacked))
          .text(concatThemes);
      }

      lastThemes = themes;
      lastY = y;
    });
  };

  return { loadText, generateComments };
}());
