# Thematic Analysis Coding Assistant

Using XGBoost to classify word2vec embeddings and generate coding suggestions.

1) Install Python packages:
```
pip install -r requirements.txt
```
2) Install Node.js packages:
```
npm i body-parser csvtojson doxtract express objects-to-csv
```
3) Classify the .docx:
```
python classify_docx.py text/reorder_exit.docx
```
4) Generate the word frequency .csv's:
```
python analyse_train_and_predict.py
```
5) Run the local server:
```
node app.js
```
6) Visit:
* http://localhost:3000/index.html for text classification
* http://localhost:3000/keywords.html for predicted and trained word frequencies
* http://localhost:3000/predict_keywords.html for predicted word frequencies
* http://localhost:3000/train_keywords.html for trained word frequencies
* http://localhost:3000/train_codes.html for trained codes
* http://localhost:3000/practices_matrix.html for "practices" classification confusion matrix
* http://localhost:3000/social_matrix.html for "social" classification confusion matrix
* http://localhost:3000/study_vs_product_matrix.html for "study vs product" classification confusion matrix
* http://localhost:3000/system_perception_matrix.html for "system perception" classification confusion matrix
* http://localhost:3000/system_use_matrix.html for "system use" classification confusion matrix
* http://localhost:3000/value_judgements_matrix.html for "value judgements" classification confusion matrix
