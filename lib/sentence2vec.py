import re
import csv
import numpy as np
from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec
from nltk import word_tokenize


class Sentence2Vec:
    word_vector_dict = {}
    vector_sentence_dict = {}

    def __init__(self, words_vectors_file):
        reader = csv.reader(open(words_vectors_file), delimiter='\t')
        for row in reader:
            self.word_vector_dict[row[0]] = np.fromstring(row[1][1:-1], sep=',')

    def get_vector(self, sentence):
        # convert to lowercase, ignore all special characters - keep only
        # alpha-numericals and spaces
        sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())
        # get word vectors from dict
        word_vectors = [self.word_vector_dict[word] for word in
            word_tokenize(sentence) if word in self.word_vector_dict.keys()]

        # create empty sentence vector
        sentence_vector = np.zeros(
            next(iter(self.word_vector_dict.values())).size)

        if len(word_vectors) > 0:
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
