import numpy as np
import matplotlib.pyplot as plt
import re
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk import sent_tokenize
from lib.sentence2vec import Sentence2Vec
from train import clean


model = Sentence2Vec('./data/word2vec.model')

text = open('text\\joint_groupbuy_jhim.txt', 'r').read()
sentences = [clean(sentence) for sentence in sent_tokenize(text)]

sentence_embeddings = np.array(
    [model.get_vector(sentence) for sentence in sentences
    if not re.match('[.,…:;–\'’!?-]', sentence)])

kmeans = KMeans(n_clusters=8)
kmeans.fit(sentence_embeddings)

# !! Get the indices of the points for each corresponding cluster
mydict = {i: np.where(kmeans.labels_ == i) for i in range(kmeans.n_clusters)}

# Transform the dictionary into list
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




# # reduce dimensionality to 2
# pca = PCA(n_components=2)
# reduced_embeddings = pca.fit_transform(sentence_embeddings)
#
# kmeans_reduced = KMeans(n_clusters=8)
# kmeans.fit(reduced_embeddings)
#
# # plot clustering #
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min = reduced_embeddings[:, 0].min() - 1
# x_max = reduced_embeddings[:, 0].max() + 1
# y_min = reduced_embeddings[:, 1].min() - 1
# y_max = reduced_embeddings[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans_reduced.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation="nearest",
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired, aspect="auto", origin="lower")
#
# plt.plot(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans_reduced.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
#             color="w", zorder=10)
# plt.title("K-means clustering on PCA-reduced sentence embeddings\n"
#           "Centroids are marked with white cross")
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
