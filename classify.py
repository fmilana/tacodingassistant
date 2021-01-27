import numpy as np
import matplotlib.pyplot as plt
import re
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk import sent_tokenize
from lib.sentence2vec import Sentence2Vec
from train import clean_text, clean_sentence


model = Sentence2Vec('./data/word2vec.model')

file_name = 'joint_groupbuy_jhim'
# file_name = 'joint_reorder_exit'
text = clean_text(open('text\\' + file_name + '.txt', 'r').read())
sentences = [clean_sentence(sentence) for sentence in sent_tokenize(text)
             if clean_sentence(sentence)
             and not re.match('[.,…:;–\'’!?-]', clean_sentence(sentence))]

sentence_embeddings = np.array([model.get_vector(sentence)
                                for sentence in sentences])

kmeans = KMeans(n_clusters=8)
kmeans.fit(sentence_embeddings)

# get the indices of the points for each corresponding cluster
mydict = {i: np.where(kmeans.labels_ == i) for i in range(kmeans.n_clusters)}
# transform the dictionary into list
dictlist = []
for cluster, labels in mydict.items():
    dictlist.append([cluster, labels])


def get_cluster_from_label(label):
    for key, labels in mydict.items():
        if label in labels[0]:
            return key


def confidence(x_vector, y_vector):
    if x_vector.size > 0 and y_vector.size > 0:
        score = np.dot(x_vector, y_vector) / (norm(x_vector) * norm(y_vector))
    return score


def testsentence(num):
    print('-----------------------------------------------------------------')
    print('sentence: "' + model.get_sentence(sentence_embeddings[num]) + '"')
    print('cluster: ' + str(get_cluster_from_label(num)))
    print('confidence: ' + str(confidence(sentence_embeddings[num],
           kmeans.cluster_centers_[get_cluster_from_label(num)])))


for i in range(15):
    testsentence(i)
