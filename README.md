# Thematic Analysis Coding Assistant

Using k-means clustering on word2vec embeddings to generate coding suggestions.

1) Run the local server:
```
flask run
```
2) Launch the Qt application:
```
python qt.py
```
3) Load a .txt file and press the "Code" button.

4) Generate _tensor.tsv and _metadata.tsv files for Tensorflow's [Embedding Projector](https://projector.tensorflow.org/):
```
python -m gensim.scripts.word2vec2tensor -i data/word2vecformat.model -o data/
```
KMeans classification dumped in "classification.txt".
