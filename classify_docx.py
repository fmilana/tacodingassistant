import csv
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from skmultilearn.adapt import MLkNN, BRkNNaClassifier, MLARAM
from skmultilearn.problem_transform import (
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import multilabel_confusion_matrix


train_file_path = 'text/reorder_exit_train.csv'
predict_file_path = 'text/reorder_exit_predict.csv'
categories_file_path = 'text/reorder_categories.csv'

coded_df = pd.read_csv(train_file_path, encoding='Windows-1252')
cat_df = pd.read_csv(categories_file_path)


def generate_training_and_testing_data(many_together):
    # convert embedding string to np array
    if not many_together:
        coded_df['sentence embedding'] = coded_df['sentence embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(']','')
                .replace('  ',' '), sep=' '))

    # split into training and testing
    by_themes = coded_df.groupby('themes')

    training_list = []
    testing_list = []
    # we now iterate by codes
    for name, group in by_themes:
        training = group.sample(frac=.8)
        testing = group.loc[~group.index.isin(training.index)]
        training_list.append(training)
        testing_list.append(testing)
    # create two new dataframes from the lists
    train_df = pd.concat(training_list)
    test_df = pd.concat(testing_list)

    # create matrices from embedding array columns
    train_embedding_matrix = np.array(train_df['sentence embedding'].tolist())
    test_embedding_matrix = np.array(test_df['sentence embedding'].tolist())
    # create matrices from theme binary columns
    train_themes_binary_matrix = train_df.iloc[:, 7:].to_numpy()
    test_themes_binary_matrix = test_df.iloc[:, 7:].to_numpy()

    # train_themes_list = train_df['themes'].tolist()
    test_themes_list = test_df['themes'].tolist()

    return (train_embedding_matrix, test_embedding_matrix,
        train_themes_binary_matrix, test_themes_binary_matrix, test_themes_list)


def add_classification_to_csv(clf, prediction_output):
    predict_df = pd.read_csv(predict_file_path, encoding='Windows-1252')

    themes_list = cat_df.category.unique()

    if isinstance(prediction_output, scipy.sparse.spmatrix):
        new_df = pd.DataFrame.sparse.from_spmatrix(data=prediction_output,
            columns=themes_list)
    else:
        new_df = pd.DataFrame(data=prediction_output, columns=themes_list)

    predict_df = predict_df.merge(new_df, left_index=True, right_index=True)
    predict_df.to_csv(predict_file_path, index=False)


def get_confusion_matrix(Y_true, Y_predicted, test_themes_list):
    all_cms = multilabel_confusion_matrix(Y_true, Y_predicted.toarray())
    
    themes_list = []
    for themes in test_themes_list:
        if ';' in themes:
            for theme in themes.split('; '):
                if theme not in themes_list:
                    themes_list.append(theme)
        else:
            if themes not in themes_list:
                themes_list.append(themes)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for axes, cm, theme in zip(ax.flatten(), all_cms, themes_list):
        plot_multilabel_confusion_matrix(cm, axes, theme, ['Y', 'N'])
    fig.tight_layout()
    plt.show()


def plot_multilabel_confusion_matrix(cm, axes, theme, class_names, fontsize=14):
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)

    heatmap = sns.heatmap(cm_df, annot=True, fmt='d', cbar=False, ax=axes)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
        ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
        ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(theme)


def classify(sentence_embedding_matrix, clf, many_together):
    (X_train, X_test,
    Y_train, Y_test, test_themes_list) = generate_training_and_testing_data(
        many_together)

    # scale data to [0-1] to avoid negative data passed to MultinomialNB
    if isinstance(clf, MultinomialNB):
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    # elif isinstance(clf, BRkNNaClassifier):
    #     parameters = {'k': range(1, 5)}
    #     clf = GridSearchCV(clf, parameters, scoring='f1_macro')
    #     clf.fit(X_train, Y_train)
    #     print (clf.best_params_, clf.best_score_)
    #     return
    # elif isinstance(clf, MLARAM):
    #     parameters = {'vigilance': [0.8, 0.85, 0.9, 0.95, 0.99],
    #         'threshold': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]}
    #     clf = GridSearchCV(clf, parameters, scoring='f1_macro')
    #     clf.fit(X_train, Y_train)
    #     print(clf.best_params_, clf.best_score_)
    #     return

    clf.fit(X_train, Y_train)

    test_score = clf.score(X_test, Y_test)

    prediction_output = clf.predict(sentence_embedding_matrix)

    if not many_together:
        add_classification_to_csv(clf, prediction_output)

    get_confusion_matrix(Y_test, clf.predict(X_test), test_themes_list)

    return test_score


# move to app.py?
import docx
import pandas as pd
from nltk import sent_tokenize
from lib.sentence2vec import Sentence2Vec
from preprocess import (
    clean_sentence,
    remove_interview_format,
    remove_interviewer,
    remove_stop_words)


model = Sentence2Vec()

def get_text(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

docx_file_path = 'text/reorder_exit.docx'
predict_file_path = 'text/reorder_exit_predict.csv'

text = get_text(docx_file_path)
text = remove_interviewer(text)

writer = csv.writer(open(predict_file_path, 'w', newline=''))
writer.writerow(['file name', 'original sentence', 'cleaned_sentence',
    'sentence embedding'])

coded_original_sentences = coded_df['original sentence'].tolist()

all_original_sentences = sent_tokenize(text)

uncoded_original_sentences = [sentence for sentence in all_original_sentences
    if sentence not in coded_original_sentences]

sentence_embedding_list = []

for sentence in uncoded_original_sentences:
    cleaned_sentence = clean_sentence(remove_stop_words(
        remove_interview_format(sentence)))
    sentence_embedding = model.get_vector(cleaned_sentence)

    writer.writerow([docx_file_path, sentence, cleaned_sentence,
        sentence_embedding])

    sentence_embedding_list.append(sentence_embedding)

sentence_embedding_matrix = np.stack(sentence_embedding_list, axis=0)

# sklearn classifiers:
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = MultinomialNB()
# clf = GaussianNB()
# clf = tree.DecisionTreeClassifier()
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = MLPClassifier(alpha=1, max_iter=1000)
# clf = AdaBoostClassifier()

# scikit multilabel classifiers:
# clf = MLkNN(k=3, s=0.5)
# clf = BRkNNaClassifier(k=3)
# clf = MLARAM(threshold=0.04, vigilance=0.99)
# clf = BinaryRelevance(
#     classifier=KNeighborsClassifier(n_neighbors=1)
# )
# clf = BinaryRelevance(
#     classifier=tree.DecisionTreeClassifier()
# )
# clf = BinaryRelevance(
#     classifier=RandomForestClassifier(max_depth=2, random_state=0)
# )
# clf = BinaryRelevance(
#     classifier=MLPClassifier(alpha=1, max_iter=1000)
# )
# clf = ClassifierChain(
#     classifier=KNeighborsClassifier(n_neighbors=1)
# )
# clf = ClassifierChain(
#     classifier=tree.DecisionTreeClassifier()
# )
# clf = ClassifierChain(
#     classifier=RandomForestClassifier(max_depth=2, random_state=0)
# )
# clf = ClassifierChain(
#     classifier=MLPClassifier(alpha=1, max_iter=1000)
# )

# classify(sentence_embedding_matrix, clf, False)



# coded_df['sentence embedding'] = coded_df['sentence embedding'].apply(
#     lambda x: np.fromstring(
#         x.replace('\n','')
#         .replace('[','')
#         .replace(']','')
#         .replace('  ',' '), sep=' '))
#
#
# scores = []
# for i in range(5):
#     clf = MLkNN(k=1, s=0.5)
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'MLkNN(k=1) >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = MLkNN(k=3, s=0.5)
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'MLkNN(k=3) >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = BRkNNaClassifier(k=1)
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'BRkNN(k=1) >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = BRkNNaClassifier(k=3)
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'BRkNN(k=3) >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = MLARAM(threshold=0.04, vigilance=0.99)
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'MLARAM >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = BinaryRelevance(classifier=tree.DecisionTreeClassifier())
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'BinaryRelevance Decision Tree >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = BinaryRelevance(classifier=RandomForestClassifier(max_depth=2, random_state=0))
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'BinaryRelevance Random Forest >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = BinaryRelevance(classifier=MLPClassifier(alpha=1, max_iter=1000))
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'BinaryRelevance MLP >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=1))
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'ClassifierChain KNN(k=1) >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = ClassifierChain(classifier=tree.DecisionTreeClassifier())
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'ClassifierChain Decision Tree >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = ClassifierChain(classifier=RandomForestClassifier(max_depth=2, random_state=0))
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# # print(f'ClassifierChain Random Forest >>>>>> {sum(scores)/len(scores)}')
#
# scores = []
# for i in range(5):
#     clf = ClassifierChain(classifier=MLPClassifier(alpha=1, max_iter=1000))
#     scores.append(classify(sentence_embedding_matrix, clf, True))
# print(f'ClassifierChain MLP >>>>>> {sum(scores)/len(scores)}')

clf = ClassifierChain(classifier=MLPClassifier(alpha=1, max_iter=1000))
classify(sentence_embedding_matrix, clf, False)
