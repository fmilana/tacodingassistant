import csv
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import tree
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.neural_network import MLPClassifier
from skmultilearn.adapt import MLkNN, BRkNNaClassifier, MLARAM
from skmultilearn.problem_transform import (
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, multilabel_confusion_matrix

train_file_path = 'text/reorder_exit_train.csv'
predict_file_path = 'text/reorder_exit_predict.csv'
categories_file_path = 'text/reorder_categories.csv'

coded_df = pd.read_csv(train_file_path, encoding='Windows-1252')
cat_df = pd.read_csv(categories_file_path)


def generate_training_and_testing_data(many_together):
    # themes_list = []
    # for i, col in enumerate(coded_df.columns):
    #     if i >= 7:
    #         themes_list.append(col)
    themes_list = cat_df.category.unique()
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
    # we now iterate by themes
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
    test_cleaned_sentences = test_df['cleaned sentence'].tolist()
    # create matrices from theme binary columns
    train_themes_binary_matrix = train_df.iloc[:, 7:].to_numpy()
    test_themes_binary_matrix = test_df.iloc[:, 7:].to_numpy()

    return (train_embedding_matrix, test_embedding_matrix,
        train_themes_binary_matrix, test_themes_binary_matrix,
        test_cleaned_sentences, themes_list)


def add_classification_to_csv(clf, prediction_output, prediction_proba):
    predict_df = pd.read_csv(predict_file_path, encoding='Windows-1252')

    themes_list = cat_df.category.unique()

    if isinstance(prediction_output, scipy.sparse.spmatrix):
        out_df = pd.DataFrame.sparse.from_spmatrix(data=prediction_output,
            columns=themes_list)
    else:
        out_df = pd.DataFrame(data=prediction_output, columns=themes_list)

    proba_cols = [f'{theme} probability' for theme in themes_list]

    if isinstance(prediction_proba, scipy.sparse.spmatrix):
        proba_df = pd.DataFrame.sparse.from_spmatrix(data=prediction_proba,
            columns=proba_cols)
    else:
        proba_df = pd.DataFrame(data=prediction_proba, columns=proba_cols)

    new_df = pd.concat([out_df, proba_df], axis=1)

    predict_df = predict_df.merge(new_df, left_index=True, right_index=True)
    predict_df.to_csv(predict_file_path, index=False)


def plot_heatmaps(clf_name, Y_true, Y_predicted, sentences_dict, themes_list):
    all_cms = multilabel_confusion_matrix(Y_true, Y_predicted.toarray())

    all_label_cms = get_keyword_labels(sentences_dict, themes_list)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for axes, labels, cm, theme in zip(ax.flatten(), all_label_cms, all_cms,
        themes_list):
        plot_multilabel_confusion_matrix(cm, labels, axes, theme, ['N', 'Y'])

    fig.suptitle(clf_name, fontsize=16)
    fig.tight_layout()

    plt.show()


def get_keyword_labels(sentences_dict, themes_list):
    word_freq_dict = {}
    all_cms = []

    more_stop_words = ['like', 'yes', 'actually', 'something', 'going', 'could',
        'would', 'oh', 'things', 'think', 'know', 'really', 'well', 'kind',
        'always', 'mean', 'maybe', 'get', 'guess', 'bit', 'much', 'go', 'one',
        'thing', 'probably']

    for category in sentences_dict:
        sentence_list = sentences_dict[category]
        joined_sentences = ''
        joined_vocab = []

        for sentence in sentence_list:
            if isinstance(sentence, str):
                joined_sentences += (' ' + sentence)

                for word in set(word_tokenize(sentence)):
                    joined_vocab.append(word)

        if len(joined_sentences) > 0:
            counter_freq = Counter([word for word in word_tokenize(joined_sentences)
                if word not in more_stop_words])

            counter_vocab = Counter(joined_vocab)

            first_most_freq = counter_freq.most_common(3)[0]
            second_most_freq = counter_freq.most_common(3)[1]
            third_most_freq = counter_freq.most_common(3)[2]
            word_freq_dict[category] = (f'{first_most_freq[0]} ' +
                f'(f: {first_most_freq[1]}, s: {counter_vocab[first_most_freq[0]]})\n' +
                f'{second_most_freq[0]} (f: {second_most_freq[1]}, s: {counter_vocab[second_most_freq[0]]})\n' +
                f'{third_most_freq[0]} (f: {third_most_freq[1]}, s: {counter_vocab[third_most_freq[0]]})')
        else:
            word_freq_dict[category] = ''

    for theme in themes_list:
        true_negative_keyword = word_freq_dict[theme + ' true_negatives']
        false_positive_keyword = word_freq_dict[theme + ' false_positives']
        false_negative_keyword = word_freq_dict[theme + ' false_negatives']
        true_positive_keyword = word_freq_dict[theme + ' true_positives']
        # create 2x2 keyword confusion matrix array for each theme
        all_cms.append(np.array([[true_negative_keyword, false_positive_keyword],
                                [false_negative_keyword, true_positive_keyword]]))

    all_cms = np.dstack(all_cms)
    all_cms = np.transpose(all_cms, (2, 0, 1))

    return all_cms


def plot_multilabel_confusion_matrix(cm, labels, axes, theme, class_names, fontsize=14):
    annot = (np.asarray([f'{count}\n {keyword}'
        for keyword, count in zip(labels.flatten(), cm.flatten())])
        ).reshape(2, 2)

    heatmap = sns.heatmap(cm, annot=annot, fmt='', cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=axes)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
        ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
        ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(theme)


def classify(sentence_embedding_matrix, clf, clf_name, many_together):
    (X_train, X_test, Y_train, Y_test,
    test_cleaned_sentences, themes_list) = generate_training_and_testing_data(
        many_together)

    # scale data to [0-1] to avoid negative data passed to MultinomialNB
    # if isinstance(clf, MultinomialNB):
    #     scaler = MinMaxScaler()
    #     X_train = scaler.fit_transform(X_train)
    #     X_test = scaler.fit_transform(X_test)
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

    # test_score = clf.score(X_test, Y_test)
    scores = []

    test_pred = clf.predict(X_test).toarray()
    for col in range(test_pred.shape[1]):
        equals = np.equal(test_pred[:, col], Y_test[:, col])
        score = np.sum(equals)/equals.size
        # print(f'{target_names[col]}: {np.sum(equals)}/{equals.size} = {score}')
        scores.append(score)
    # print(classification_report(test_pred, Y_test, target_names=target_names))

    prediction_output = clf.predict(sentence_embedding_matrix)

    prediction_output = prediction_output.astype(int)

    prediction_proba = clf.predict_proba(sentence_embedding_matrix)

    if not many_together:
        add_classification_to_csv(clf, prediction_output, prediction_proba)

    sentences_dict = {}

    # for col in range(test_pred.shape[1]):
    for col, class_name in enumerate(themes_list):
        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []

        for row in range(test_pred.shape[0]):
            if test_pred[row, col] == 1 and Y_test[row, col] == 1:
                true_positives.append(test_cleaned_sentences[row])
            elif test_pred[row, col] == 0 and Y_test[row, col] == 0:
                true_negatives.append(test_cleaned_sentences[row])
            elif test_pred[row, col] == 1 and Y_test[row, col] == 0:
                false_positives.append(test_cleaned_sentences[row])
            elif test_pred[row, col] == 0 and Y_test[row, col] == 1:
                false_negatives.append(test_cleaned_sentences[row])

        sentences_dict[class_name + ' true_positives'] = true_positives
        sentences_dict[class_name + ' true_negatives'] = true_negatives
        sentences_dict[class_name + ' false_positives'] = false_positives
        sentences_dict[class_name + ' false_negatives'] = false_negatives

    plot_heatmaps(clf_name, Y_test, clf.predict(X_test), sentences_dict,
        themes_list)

    # print(f'sentences_dict = {sentences_dict}')

    return (scores, true_positives, true_negatives, false_positives,
        false_negatives)


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

file = open(predict_file_path, 'w', newline='')
writer = csv.writer(file, delimiter=',')
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

file.close()

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


#
# coded_df['sentence embedding'] = coded_df['sentence embedding'].apply(
#     lambda x: np.fromstring(
#         x.replace('\n','')
#         .replace('[','')
#         .replace(']','')
#         .replace('  ',' '), sep=' '))
# #
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
# for i in range(20):
#     clf = ClassifierChain(classifier=MLPClassifier(alpha=1, max_iter=1000))
#     scores.append(classify(sentence_embedding_matrix, clf,
#         'ClassifierChain MLP', True))
# print(f'ClassifierChain MLP >>>>>> {sum(scores)/len(scores)}')

# scores = []
# for i in range(20):
#     clf = ClassifierChain(classifier=AdaBoostClassifier(
#         n_estimators=2, learning_rate=0.4))
#     scores.append(classify(sentence_embedding_matrix, clf,
#         'ClassifierChain AdaBoost', True))
# print(f'ClassifierChain AdaBoost >>>>>>> {sum(scores)/len(scores)}')

# class_names = ['practices', 'social', 'study vs product',
#     'system perception', 'system use', 'value judgements']
# practices_score = []
# social_score = []
# study_vs_product_score = []
# system_perception_score = []
# system_use_score = []
# value_judgements_score = []
# iter = 20
# for i in range(iter):
#     clf = ClassifierChain(classifier=GradientBoostingClassifier(n_estimators=2,
#         learning_rate=0.4, max_depth=1))
#     scores = classify(sentence_embedding_matrix, clf,
#         'CCkNN(k=3)', True)
#     practices_score.append(scores[0])
#     social_score.append(scores[1])
#     study_vs_product_score.append(scores[2])
#     system_perception_score.append(scores[3])
#     system_use_score.append(scores[4])
#     value_judgements_score.append(scores[5])
#     print(f'{i+1}/{iter}')
# print(f'practices: {sum(practices_score)/len(practices_score)}')
# print(f'social: {sum(social_score)/len(social_score)}')
# print(f'study vs product: {sum(study_vs_product_score)/len(study_vs_product_score)}')
# print(f'system perception: {sum(system_perception_score)/len(system_perception_score)}')
# print(f'system use: {sum(system_use_score)/len(system_use_score)}')
# print(f'value judgements: {sum(value_judgements_score)/len(value_judgements_score)}')




# clf = ClassifierChain(classifier=AdaBoostClassifier())
#
# parameters = {
#     'classifier__n_estimators': [2, 10, 25, 50],
#     'classifier__learning_rate': [0.3, 0.4, 0.5, 0.6, 0.7]
# }
#
# grid_search_cv = GridSearchCV(clf, parameters)
#
# # for param in grid_search_cv.get_params().keys():
# #     print(param)
#
# X_train, _, Y_train, _, _, _ = generate_training_and_testing_data(False)
#
# grid_search_cv.fit(X_train, Y_train)
# print(grid_search_cv.best_score_)
# print(grid_search_cv.best_params_)





clf = ClassifierChain(classifier=MLPClassifier(alpha=1, max_iter=1000))
classify(sentence_embedding_matrix, clf, 'ClassifierChain MLP', False)

# clf = ClassifierChain(classifier=RandomForestClassifier(max_depth=2, random_state=0))
# classify(sentence_embedding_matrix, clf, 'ClassifierChain Random Forest', False)

# clf = ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=1))
# classify(sentence_embedding_matrix, clf, 'ClassifierChain kNN(k=1)', False)

# clf = ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=3))
# classify(sentence_embedding_matrix, clf, 'ClassifierChain kNN(k=3)', False)

# clf = ClassifierChain(classifier=GradientBoostingClassifier(n_estimators=2,
#         learning_rate=0.4, max_depth=1))
# classify(sentence_embedding_matrix, clf, 'ClassifierChain Gradient Boosting',
#     False)
# #
# clf = ClassifierChain(classifier=AdaBoostClassifier(n_estimators=2,
#     learning_rate=0.4))
# classify(sentence_embedding_matrix, clf, 'ClassifierChain AdaBoost', False)
