import os
import re
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from nltk import sent_tokenize
from lib.sentence2vec import Sentence2Vec


class Classifier():
    model = None
    kmeans = None
    sentence_embeddings = None
    cluster_label_dict = {}
    original_sentence_dict = {}

    def __init__(self, model_name):
        self.model = Sentence2Vec(model_name)

    def clean_text(self, text):
        text = text.lower()
        # remove interviewer
        text = re.sub(r'iv[0-9]*[ \t].*', '', text)
        # remove interview format
        regexp = (r'p[0-9]+\w*|speaker key|r*user\s*\d+( - study \d+)*|'
                  '(iv[0-9]*|ie|um|a[0-9]+)\t|'
                  '(interviewer|interviewee|person [0-9]|participant)|'
                  '\d{2}:\d{2}:\d{2}|\[(.*?)\]|\[|\]')
        text = re.sub(regexp, '', text)
        # replace "..." at the end of a line with "."
        text = re.sub(r'\.\.\.[\r\n]', '.', text)
        # replace multiple spaces or newlines with one space
        text = re.sub(r' +|[\r\n\t]+', ' ', text)
        return text


    def clean_sentence(self, sentence):
        return re.sub(r'[^A-Za-z ]+', '', sentence)

    def classify(self, text):
        text = self.clean_text(text)
        clean_sentences = []

        for sentence in sent_tokenize(text):
            clean_sentence = self.clean_sentence(sentence)
            self.original_sentence_dict[clean_sentence] = sentence
            if not re.match('[.,…:;–\'’!?-]', clean_sentence):
                clean_sentences.append(clean_sentence)

        self.sentence_embeddings = np.array([self.model.get_vector(sentence)
            for sentence in clean_sentences])

        # elbow method
        inertias = []
        kmeans_fit_list = []
        for k in range(1, 15):
            kmeans = KMeans(n_clusters=k)
            kmeans_fit = kmeans.fit(self.sentence_embeddings)
            kmeans_fit_list.append(kmeans_fit)
            inertias.append(kmeans_fit.inertia_)

        knee_locator = KneeLocator(
            range(1, len(inertias)+1),
            inertias,
            curve='convex',
            direction='decreasing'
        )
        optimum_k = knee_locator.knee
        self.kmeans = kmeans_fit_list[optimum_k-1]

        # get the indices of the points for each corresponding cluster
        self.cluster_label_dict = {i: np.where(self.kmeans.labels_ == i)
                       for i in range(self.kmeans.n_clusters)}


    def get_cluster_from_label(self, label):
        for key, labels in self.cluster_label_dict.items():
            if label in labels[0]:
                return key


    def get_confidence(self, x_vector, y_vector):
        normd = norm(x_vector) * norm(y_vector)
        if x_vector.size > 0 and y_vector.size > 0 and not normd == 0:
            score = np.dot(x_vector, y_vector)/normd
        else:
            score = 0
        return score


    def get_cluster_label_dict(self):
        return self.cluster_label_dict


    def get_output_dict(self):
        output_dict = {}
        for num in range(1, self.sentence_embeddings.shape[0]):
            sentence = self.model.get_sentence(self.sentence_embeddings[num])
            cluster = self.get_cluster_from_label(num)
            confidence = self.get_confidence(self.sentence_embeddings[num],
                self.kmeans.cluster_centers_[self.get_cluster_from_label(num)])
            original_sentence = self.original_sentence_dict[sentence]
            output_dict[original_sentence] = [cluster, confidence]
        # print(output_dict)
        return output_dict
