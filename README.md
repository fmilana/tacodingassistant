# Thematic Analysis Coding Assistant

Using XGBoost to classify word2vec embeddings and generate coding suggestions.

1) Install Python packages:
```
pip install -r requirements.txt
```
2) Install Node.js packages:
```
npm i body-parser csvtojson doxtract express
```
3) Classify the .docx:
```
python classify_docx.py text/reorder_exit.docx
```
4) Generate the word frequency .csv:
```
python analyse_predictions.py
```
5) Run the local server:
```
node app.js
```
6) Visit http://localhost:3000/index.html for text classification and http://localhost:3000/words.html for word frequencies for each theme.
