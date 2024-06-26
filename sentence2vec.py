import os
import re
import numpy as np
import pickle
import gensim.downloader
from nltk import word_tokenize
from datetime import datetime

from path_util import resource_path


class Sentence2Vec:
    # https://github.com/RaRe-Technologies/gensim-data
    model_name = 'glove-twitter-50'
    model_file_path = resource_path('data/embeddings/glove_model.pkl')

    vector_sentence_dict = {}

    def __init__(self):
        start = datetime.now()
        if os.path.exists(self.model_file_path):
            print('loading word embeddings from disk...')
            with open(self.model_file_path, 'rb') as f:
                self.model = pickle.load(f)
                f.close()
        else:
            print(f'downloading glove model ({self.model_name})...')
            self.model = gensim.downloader.load(self.model_name)
            os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
            with open(self.model_file_path, 'wb') as f:
                pickle.dump(self.model, f)
                f.close()
        print(f'done in {datetime.now() - start}')


    def get_vector(self, sentence):
        # convert to lowercase, ignore all special characters - keep only
        # alpha-numericals and spaces
        sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())
        # get word vectors from model
        word_vectors = [self.model.wv[word] for word in word_tokenize(sentence) if word in self.model.wv]
        # create empty sentence vector
        sentence_vector = np.zeros(self.model.vector_size)
        # sentence vector equals average of word vectors
        if (len(word_vectors) > 0):
            sentence_vector = (np.array([sum(word_vector) for word_vector in zip(*word_vectors)])) / sentence_vector.size

        self.vector_sentence_dict[sentence_vector.tobytes()] = sentence

        return sentence_vector


    def get_sentence(self, vector):
        return self.vector_sentence_dict.get(vector.tobytes())