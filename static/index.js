/* global d3 fetch */

const getHtml = function () {
  d3.select('#loading-gif')
    .style('display', 'block');

  fetch('/get_html')
    .then((res) => res.json().then((jsonObj) => {
        let text = '';
        const trainObjects = [];
        const predictObjects = [];

        for (let i = 0; i < jsonObj.length; i++) {
          if (Object.prototype.hasOwnProperty.call(jsonObj[i], 'trainSentence')) {
            trainObjects.push(jsonObj[i]);
          } else if (Object.prototype.hasOwnProperty.call(jsonObj[i], 'predictSentence')) {
            predictObjects.push(jsonObj[i]);
          } else if (Object.prototype.hasOwnProperty.call(jsonObj[i], 'wholeText')) {
            text = jsonObj[i].wholeText;
          }
        }
        console.log('read all train, predict sentences + whole text');
        console.log('trainObjects length:', trainObjects.length);
        console.log('predictObjects length:', predictObjects.length);

        text = highlightSentences(text, trainObjects, predictObjects);

        d3.select('#loading-gif')
            .style('display', 'none');

        console.log('updating html...');

        d3.select('#text-box')
          .selectAll('p')
          .data([text])
          .enter()
          .append('p')
          .attr('id', 'text-paragraph')
          .html(data => data);

        console.log('updated html!');

        console.log('generating comments...');

        generateComments();
      }));
};


const highlightSentences = function (text, trainObjects, predictObjects) {
  const mapObj = {};

  for (let i = 0; i < trainObjects.length; i++) {
    const obj = trainObjects[i];
    const trainSentence = obj.trainSentence;
    const themes = obj.themes;
    //better way?
    if (trainSentence.length > 6) {
      mapObj[trainSentence] = `<span data-tooltip="${themes}"` +
      `style="background-color: #dbdbdb">${trainSentence}</span>`;
    }
  }

  for (let i = 0; i < predictObjects.length; i++) {
    const obj = predictObjects[i];
    // const position = obj.position
    const predictSentence = obj.predictSentence;
    const themes = obj.themes;
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

  d3.selectAll('span')
    .each(function () {
      const obj = {
        themes: d3.select(this).attr('data-tooltip'),
        y: this.getBoundingClientRect().y,
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

    let y = obj.y.toString();
    const color = obj.color;
    const themes = obj.themes;

    const skipText = (themes === lastThemes) && ((lastY === (y - 28) || (lastY === (y - 33))));
    const concatCanvas = (y === lastY);

    if (!concatCanvas) {
      const canvas = d3.select('#comments-box')
        .append('svg')
        .style('position', 'absolute')
        .style('top', () => {
          if (skipText) {
            y = parseFloat(y) - 5;
            return y.toString().concat('px');
          }
          return y.concat('px');
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


getHtml();
