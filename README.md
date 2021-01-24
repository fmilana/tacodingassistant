# Thematic Analysis Coding Assistant

Using k-means clustering on word2vec embeddings to generate coding suggestions.

To train the word2vec model:
```
python train.py
```
To generate _tensor.tsv and _metadata.tsv files for Tensorboard Projector:
```
python -m gensim.scripts.word2vec2tensor -i data/word2vecformat.model -o data/
```
To classify the embeddings:
```
python cluster.py
```