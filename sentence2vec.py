import sys
import os
import re
import csv
import numpy as np
import pickle
import gensim.downloader
from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec
from nltk import word_tokenize
from datetime import datetime


class Sentence2Vec:
    # https://github.com/RaRe-Technologies/gensim-data
    model_name = 'glove-twitter-50'
    model_file_path = 'embeddings/word2vec_model.pickle'

    vector_sentence_dict = {}

    def __init__(self):
        start = datetime.now()
        if os.path.exists(self.model_file_path):
            print('loading word embeddings from disk...')
            with open(self.model_file_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print('downloading word embeddings...')
            self.model = gensim.downloader.load(self.model_name)
            with open(self.model_file_path, 'wb') as f:
                pickle.dump(self.model, f)
        print(f'done in {datetime.now() - start}')


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
