import sys
import csv
import re
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import seaborn as sns
import docx
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from lib.sentence2vec import Sentence2Vec
from preprocess import (
    clean_sentence,
    remove_stop_words)
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

    clf.fit(X_train, Y_train)

    scores = []

    test_pred = clf.predict(X_test).toarray()
    for col in range(test_pred.shape[1]):
        equals = np.equal(test_pred[:, col], Y_test[:, col])
        score = np.sum(equals)/equals.size
        scores.append(score)

    prediction_output = clf.predict(sentence_embedding_matrix)

    prediction_output = prediction_output.astype(int)

    prediction_proba = clf.predict_proba(sentence_embedding_matrix)

    if not many_together:
        add_classification_to_csv(clf, prediction_output, prediction_proba)

    sentences_dict = {}

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

all_original_sentences = sent_tokenize(text)

uncoded_original_sentence_position_dict = {}

position = 0
for sentence in all_original_sentences:
    if sentence not in coded_original_sentences:
        if sentence in uncoded_original_sentence_position_dict:
            uncoded_original_sentence_position_dict[sentence].append(position)
        else:
            uncoded_original_sentence_position_dict[sentence] = [position]
    position += len(re.sub('\n', '', sentence)) + 1

sentence_embedding_list = []

for sentence in uncoded_original_sentence_position_dict.keys():
    position = uncoded_original_sentence_position_dict[sentence]

    cleaned_sentence = remove_stop_words(clean_sentence(sentence))
    sentence_embedding = model.get_vector(cleaned_sentence)

    writer.writerow([', '.join(str(i) for i in position), sentence, cleaned_sentence,
        sentence_embedding])

    sentence_embedding_list.append(sentence_embedding)

file.close()

sentence_embedding_matrix = np.stack(sentence_embedding_list, axis=0)


clf = ClassifierChain(classifier=XGBClassifier())
_, _, _, _, _, accuracies, f_measures = classify(sentence_embedding_matrix,
    clf, 'ClassifierChain XGBoost oversample', True, False)

print(f'accuracy = {accuracies}')
print(f'f_measures = {f_measures}')
