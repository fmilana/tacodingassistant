# Thematic Analysis Coding Assistant

Using XGBoost to classify word2vec embeddings and generate coding suggestions.

1) Classify the .docx:
```
python classify_docx.py text/reorder_exit.docx
```
2) Generate the word frequency .csv:
```
python analyse_predictions.py
```
3) Run the local server:
```
node app.js
```
4) Visit http://localhost:3000/index.html for text classification and http://localhost:3000/words.html for word frequencies for each theme.
