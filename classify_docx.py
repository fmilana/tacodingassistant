import sys
import csv
import re
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
from xgboost import XGBClassifier

from export_docx_themes_with_embeddings import Export


doc_path = sys.argv[1]

export = Export(doc_path)
export.process()


train_file_path = doc_path.replace('.docx', '_train.csv')
predict_file_path = doc_path.replace('.docx', '_predict.csv')
categories_file_path = 'text/reorder_categories.csv'

coded_df = pd.read_csv(train_file_path, encoding='Windows-1252')
cat_df = pd.read_csv(categories_file_path)

print('inside classify_docx.py')

def get_sample_weights(Y_train):
    sample_weights = []
    sums = Y_train.sum(axis=0)
    num_most_common = np.amax(sums)
    class_weights = []

    for x in sums:
        class_weights.append(num_most_common/x)

    for row in range(Y_train.shape[0]):
        sample_weight = 1

        for col in range(Y_train.shape[1]):
            if Y_train[row][col] == 1:
                class_weight = class_weights[col]
                if class_weight > sample_weight:
                    sample_weight = class_weight

        sample_weights.append(sample_weight)

    return np.asarray(sample_weights)


def generate_training_and_testing_data(oversample, many_together):
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

    # oversample minority classes
    if oversample:
        Y_train = train_df.iloc[:, 7:].to_numpy()

        class_dist = [x/Y_train.shape[0] for x in Y_train.sum(axis=0)]

        print(f'class distribution before oversampling = {class_dist}')

        sample_weights = get_sample_weights(Y_train)
        sample_weights = np.rint(sample_weights)

        train_df = train_df.reindex(train_df.index.repeat(sample_weights))

        Y_train_os = train_df.iloc[:, 7:].to_numpy()

        class_dist_os = [x/Y_train_os.shape[0] for x in Y_train_os.sum(axis=0)]

        print(f'class distribution after oversampling = {class_dist_os}')

        train_df.to_csv('text/reorder_exit_oversampled.csv', index=False,
            header=True)

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

    stop_words = open('text/analysis_stopwords.txt', 'r').read().split(',')

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
                if word not in stop_words])

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

    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

    heatmap = sns.heatmap(cm, cmap=cmap, annot=annot, fmt='', cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=axes)
    sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
        ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
        ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(theme)


def classify(sentence_embedding_matrix, clf, clf_name, oversample, many_together):
    (X_train, X_test, Y_train, Y_test,
    test_cleaned_sentences, themes_list) = generate_training_and_testing_data(
        oversample, many_together)

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

    # plot_heatmaps(clf_name, Y_test, clf.predict(X_test), sentences_dict,
    #     themes_list)

    ##### evaluate accuracy and f-measure per class ######

    accuracies = []
    f_measures = []

    for col in range(test_pred.shape[1]):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for row in range(test_pred.shape[0]):
            if test_pred[row][col] == 1 and Y_test[row][col] == 1:
                tp += 1
            elif test_pred[row][col] == 1 and Y_test[row][col] == 0:
                fp += 1
            elif test_pred[row][col] == 0 and Y_test[row][col] == 1:
                fn += 1
            elif test_pred[row][col] == 0 and Y_test[row][col] == 0:
                tn += 1

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if (precision + recall) > 0:
            f_measure = (2 * precision * recall) / (precision + recall)
        else:
            f_measure = 0

        accuracies.append((tp + tn)/(tp + tn + fp + fn))
        f_measures.append(f_measure)

    return (scores, true_positives, true_negatives, false_positives,
        false_negatives, accuracies, f_measures)


# move to app.py?
import docx
import pandas as pd
from nltk import sent_tokenize
from lib.sentence2vec import Sentence2Vec
from preprocess import (
    clean_sentence,
    # remove_interview_format,
    # remove_interviewer,
    remove_stop_words)


model = export.model


def get_text(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)


text = get_text(doc_path)
text = text.replace("’", "'")

file = open(predict_file_path, 'w', newline='')
writer = csv.writer(file, delimiter=',')
writer.writerow(['position', 'original sentence', 'cleaned_sentence',
    'sentence embedding'])

coded_original_sentences = coded_df['original sentence'].tolist()

# text = remove_interview_format(text, lower=False)

all_original_sentences = sent_tokenize(text)

uncoded_original_sentence_position_dict = {}

# i = 0

position = 0
for sentence in all_original_sentences:
    if sentence not in coded_original_sentences:
        if sentence in uncoded_original_sentence_position_dict:
            uncoded_original_sentence_position_dict[sentence].append(position)
        else:
            uncoded_original_sentence_position_dict[sentence] = [position]
    position += len(re.sub('\n', '', sentence)) + 1

    # if i < 10:
    #     print(f'{i} sentence: {sentence[0:50]}')
    #     print(f'{i} len(sentence): {len(re.sub('\n', ' ', sentence))}')
    #     i += 1

sentence_embedding_list = []

# i = 0

for sentence in uncoded_original_sentence_position_dict.keys():
    # if i < 20:
    #     print(f'sentence: {sentence[0:20]}')
    #     print(f'position: {uncoded_original_sentence_position_dict[sentence]}')
    #     i += 1

    position = uncoded_original_sentence_position_dict[sentence]

    # cleaned_sentence = clean_sentence(remove_stop_words(
    #     remove_interview_format(sentence)))
    cleaned_sentence = remove_stop_words(clean_sentence(sentence))
    sentence_embedding = model.get_vector(cleaned_sentence)

    writer.writerow([', '.join(str(i) for i in position), sentence, cleaned_sentence,
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


# --------------------WHEN MANY TOGETHER
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



# clf = MLkNN(k=1, s=0.5)
# _, _, _, _, _, f_measures = classify(sentence_embedding_matrix, clf, 'MLkNN(k=1)', True, False)

# clf = MLkNN(k=3, s=0.5)
# _, _, _, _, _, f_measures = classify(sentence_embedding_matrix, clf, 'MLkNN(k=3)', False)

# clf = MLARAM(threshold=0.04, vigilance=0.99)
# _, _, _, _, _, f_measures = classify(sentence_embedding_matrix, clf, 'MLARAM', False)

# clf = ClassifierChain(classifier=tree.DecisionTreeClassifier())
# _, _, _, _, _, f_measures = classify(sentence_embedding_matrix, clf, 'ClassifierChain Decision Tree', False)

# clf = ClassifierChain(classifier=MLPClassifier(alpha=1, max_iter=1000))
# _, _, _, _, _, f_measures = classify(sentence_embedding_matrix, clf, 'ClassifierChain MLP', False)

# clf = ClassifierChain(classifier=RandomForestClassifier(max_depth=2, random_state=0))
# _, _, _, _, _, f_measures = classify(sentence_embedding_matrix, clf, 'ClassifierChain Random Forest', False)

# clf = ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=1))
# _, _, _, _, _, f_measures = classify(sentence_embedding_matrix, clf, 'ClassifierChain kNN(k=1)', False)

# clf = ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=3))
# _, _, _, _, _, f_measures = classify(sentence_embedding_matrix, clf, 'ClassifierChain kNN(k=3)', False)

# clf = ClassifierChain(classifier=GradientBoostingClassifier(n_estimators=2,
#         learning_rate=0.4, max_depth=1))
# _, _, _, _, _, accuracies, f_measures = classify(sentence_embedding_matrix, clf, 'ClassifierChain Gradient Boosting oversample',
#     True, False)

# clf = ClassifierChain(classifier=AdaBoostClassifier(n_estimators=2,
#     learning_rate=0.4))
# _, _, _, _, _, accuracies, f_measures = classify(sentence_embedding_matrix, clf, 'ClassifierChain AdaBoost oversample', True, False)

clf = ClassifierChain(classifier=XGBClassifier())
_, _, _, _, _, accuracies, f_measures = classify(sentence_embedding_matrix,
    clf, 'ClassifierChain XGBoost oversample', True, False)
#
# clf = ClassifierChain(classifier=XGBClassifier(scale_pos_weight=50))
# _, _, _, _, _, accuracies, f_measures = classify(sentence_embedding_matrix,
#     clf, 'title', True, False)

print(f'accuracy = {accuracies}')
print(f'f_measures = {f_measures}')





# iter = 20
#
# themes = ['practices', 'social', 'study vs product', 'system perception',
#     'system use', 'value_jugements']
#
# accuracies_dict = {theme: [] for theme in themes}
# f_measures_dict = {theme: [] for theme in themes}
#
# clf = ClassifierChain(classifier=XGBClassifier(scale_pos_weight=50))
# for i in range(iter):
#     _, _, _, _, _, accuracies, f_measures = classify(sentence_embedding_matrix,
#         clf, 'title', True, True)
#
#     for j, theme in enumerate(themes):
#         accuracies_dict[theme].append(accuracies[j])
#         f_measures_dict[theme].append(f_measures[j])
#
#     print(f'{i+1}/{iter} done')
#
# for theme in accuracies_dict:
#     accuracies_dict[theme] = sum(accuracies_dict[theme])/iter
#     f_measures_dict[theme] = sum(f_measures_dict[theme])/iter
#
# print(f'accuracies = {[accuracies_dict[theme] for theme in themes]}')
# print(f'f_measures = {[f_measures_dict[theme] for theme in themes]}')





# clf = ClassifierChain(classifier=XGBClassifier())
#
# parameters = {
#     'classifier__n_estimators': [2],
#     'classifier__colsample_bytree': [0.6, 0.7, 0.8],
#     'classifier__max_depth': [20, 30, 40, 50],
#     'classifier__reg_alpha': [1.3, 1.4, 1.5],
#     'classifier__reg_lambda': [1.3, 1.4, 1.5],
#     'classifier__subsample': [0.9]
# }
#
# grid_search_cv = GridSearchCV(clf, parameters, scoring=['accuracy', 'f1'],
#     refit='f1')
#
# X_train, _, Y_train, _, _, _ = generate_training_and_testing_data(
#     oversample=False,
#     many_together=False)
#
# grid_search_cv.fit(X_train, Y_train)
#
# print(f'best scores: {grid_search_cv.best_score_}')
# print(f'best params: {grid_search_cv.best_params_}')
# # print(f'results: {grid_search_cv.cv_results_}')
