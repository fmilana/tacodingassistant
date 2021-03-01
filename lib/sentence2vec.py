import re
import csv
import numpy as np
import gensim.downloader
from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec
from nltk import word_tokenize


class Sentence2Vec:
    vector_sentence_dict = {}

    def __init__(self, model_name):
        print('loading model...')
        self.model = gensim.downloader.load(model_name)
        print('done!')

    def get_vector(self, sentence):
        # convert to lowercase, ignore all special characters - keep only
        # alpha-numericals and spaces
        sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())
        # get word vectors from model
        word_vectors = [self.model.wv[word] for word in word_tokenize(sentence)
                   if word in self.model.wv]
        # create empty sentence vector
        sentence_vector = np.zeros(self.model.vector_size)
        # sentence vector equals average of word vectors
        if (len(word_vectors) > 0):
            sentence_vector = (np.array([sum(word_vector) for word_vector
                                in zip(*word_vectors)])) / sentence_vector.size

        self.vector_sentence_dict[sentence_vector.tobytes()] = sentence

        return sentence_vector

    def get_sentence(self, vector):
        return self.vector_sentence_dict.get(vector.tobytes())

    def similarity(self, x, y):
        # calculates similarity based on distance between vectors
        x_vector = self.get_vector(x)
        y_vector = self.get_vector(y)
        score = 0
        if x_vector.size > 0 and y_vector.size > 0:
            score = dot(x_vector, y_vector) / (norm(x_vector) * norm(y_vector))
        return score
