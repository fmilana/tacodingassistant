import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


csv_file_path = 'text/joint_groupbuy_jhim.csv'


def generate_data_from_csv():
    df = pd.read_csv(csv_file_path, encoding='Windows-1252')
    # convert embedding string to np array
    df.iloc[:, 4] = df.iloc[:, 4].apply(lambda x: np.fromstring(
        x.replace('\n','')
        .replace('[','')
        .replace(']','')
        .replace('  ',' '), sep=' '))
    # create matrix from embedding array column
    embedding_matrix = np.array(df.iloc[:, 4].tolist())
    codes_array = df.iloc[:, 5].to_numpy()

    print(f'embedding_matrix shape = {embedding_matrix.shape}')
    print(f'codes_array shape = {codes_array.shape}')

    le = preprocessing.LabelEncoder()
    codes_encoded = le.fit_transform(codes_array)

    return embedding_matrix, codes_encoded


def knn_classify(sentence_embedding_matrix):
    training_embedding_matrix, codes_encoded = generate_data_from_csv()

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(training_embedding_matrix, codes_encoded)
    print(f'predict = {clf.predict(sentence_embedding_matrix)}')
    print(f'predict_proba = {clf.predict_proba(sentence_embedding_matrix)}')
    print(f'score = {clf.score(sentence_embedding_matrix, codes_encoded)}')
