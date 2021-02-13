# Thematic Analysis Coding Assistant

Using k-means clustering on word2vec embeddings to generate coding suggestions.

1) Train the word2vec model:
```
python word_embedding.py
```
2) Run the local server:
```
flask run
```
3) Launch the Qt application:
```
python qt.py
```
4) Load a .txt file and press the "Code" button.

4) Visualise word embedding in Tensorflow's [Embedding Projector](https://projector.tensorflow.org/) by uploading vectors.tsv and metadata.tsv

KMeans classification dumped in "classification.txt".
