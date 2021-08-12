import csv
import re
import os
import pickle
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import docx
import pandas as pd
from datetime import datetime
from nltk import sent_tokenize, word_tokenize
from lib.sentence2vec import Sentence2Vec
from preprocess import (
    clean_sentence,
    remove_stop_words)
from collections import Counter
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import multilabel_confusion_matrix
from xgboost import XGBClassifier
from export_docx_themes_with_embeddings import Export


regexp = None

train_file_path = None
predict_file_path = None

cat_df = None
original_train_df = None # save train_df copy here before test split and oversampling
train_df = None
moved_predict_df = None


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
    themes_list = list(cat_df)
    global train_df
    global original_train_df
    original_train_df = train_df.copy()
    # convert embedding string to np array
    if not many_together:
        train_df['sentence_embedding'] = train_df['sentence_embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(']','')
                .replace('  ',' '), sep=' '))

    # split into training and testing
    by_themes = train_df.groupby('themes')

    training_list = []
    testing_list = []
    # iterate by themes
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
    train_embedding_matrix = np.array(train_df['sentence_embedding'].tolist())
    test_embedding_matrix = np.array(test_df['sentence_embedding'].tolist())
    test_cleaned_sentences = test_df['cleaned_sentence'].tolist()
    # create matrices from theme binary columns
    train_themes_binary_matrix = train_df.iloc[:, 7:].to_numpy()
    test_themes_binary_matrix = test_df.iloc[:, 7:].to_numpy()

    return (train_embedding_matrix, test_embedding_matrix,
        train_themes_binary_matrix, test_themes_binary_matrix,
        test_cleaned_sentences, themes_list)


def add_classification_to_csv(clf, prediction_output, prediction_proba):
    themes_list = list(cat_df)

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

    predict_df = pd.read_csv(predict_file_path, encoding='utf-8')

    predict_df = predict_df.merge(new_df, left_index=True, right_index=True)

    global moved_predict_df
    if moved_predict_df is not None:
        global train_df
        # add moved predictions so they still show up in table as predictions
        moved_predict_df = moved_predict_df.drop([
            'file name',
            'comment_id',
            'codes',
            'themes'], axis=1)

        for theme in themes_list:
            moved_predict_df[f'{theme} probability'] = moved_predict_df[theme]

        predict_df = predict_df.append(moved_predict_df)

        # remove moved predictions from train
        train_df = train_df[train_df['codes'].notna()]

    predict_df.to_csv(predict_file_path, index=False, encoding='utf-8-sig')
    moved_predict_df = None


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


def write_cms_to_csv(sentences_dict, themes_list):
    cm_col_names = [
        'true_positives',
        'false_positives',
        'true_negatives',
        'false_negatives'
    ]

    for theme in themes_list:
        with open(f'text/cm/reorder_exit_{theme.replace(" ", "_")}_cm.csv',
            'w', newline='', encoding='utf-8') as file:

            writer = csv.writer(file, delimiter=',')

            lengths = []

            for col_name in cm_col_names:
                sentences = sentences_dict[f'{theme} {col_name}']
                lengths.append(len(sentences))

            writer.writerow(f'{col_name.replace("_", " ").title()} ({lengths[i]})'
                for i, col_name in enumerate(cm_col_names))

            all_sentences_lists = []

            for col_name in cm_col_names:
                sentences = sentences_dict[f'{theme} {col_name}']
                original_sentences = []

                for sentence in sentences:
                    # to-do: better way than .any()
                    # str() used because sometimes bool
                    original_sentence = str(original_train_df.loc[
                        original_train_df['cleaned_sentence'] == sentence]['original_sentence'].any())

                    if len(original_sentence) > 0:
                        # # remove interview artifacts (not stopwords)
                        # original_sentence = clean_sentence(original_sentence,
                        #     keep_alphanum=True)
                        original_sentence.replace('…', '...')

                        if len(original_sentence) > 0:
                            original_sentences.append(original_sentence)

                emptyToAdd = max(lengths) - len(original_sentences)

                for i in range(emptyToAdd):
                    original_sentences.append('')

                all_sentences_lists.append(original_sentences)

            zipped = zip(*[sentences for sentences in all_sentences_lists])

            for row in zipped:
                writer.writerow(row)


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


def plot_multilabel_confusion_matrix(cm, labels, axes, theme, class_names,
    fontsize=14):
    annot = (np.asarray([f'{count}\n {keyword}'
        for keyword, count in zip(labels.flatten(), cm.flatten())])
        ).reshape(2, 2)

    cmap = sns.color_palette('ch:start=.2,rot=-.3', as_cmap=True)

    heatmap = sns.heatmap(cm, cmap=cmap, annot=annot, fmt='', cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=axes)
    sns.color_palette('ch:start=.2,rot=-.3', as_cmap=True)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
        ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
        ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(theme)


def classify(sentence_embedding_matrix, clf, clf_name, oversample,
    many_together):
    print('running classify function...')
    start_function = datetime.now()

    print('running generate_training_and_testing_data...')
    start_gen = datetime.now()
    (X_train, X_test, Y_train, Y_test,
    test_cleaned_sentences, themes_list) = generate_training_and_testing_data(
        oversample, many_together)
    print(f'generate_training_and_testing_data run in {datetime.now() - start_gen}')

    print('fitting clf...')
    start_fit = datetime.now()

    clf.fit(X_train, Y_train)

    print(f'done fitting clf in {datetime.now() - start_fit}')

    print(f'generating confusion matrices...')
    start_cm = datetime.now()

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

    write_cms_to_csv(sentences_dict, themes_list)

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

    print(f'confusion matrices created in {datetime.now() - start_cm}')

    print(f'classify function run in {datetime.now() - start_function}')

    return (scores, true_positives, true_negatives, false_positives,
        false_negatives, accuracies, f_measures)


def get_text(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)


def run_classifier(doc_path, interviewer_regexp, modified_train_file_path=None):
    global regexp
    global train_file_path
    global predict_file_path
    global cat_df
    global train_df
    global moved_predict_df

    global regexp
    regexp = interviewer_regexp

    print('inside script.')
    start_script = datetime.now()

    if modified_train_file_path is not None:
        train_file_path = modified_train_file_path
        predict_file_path = modified_train_file_path.replace('train', 'predict')
        model = Sentence2Vec()
    else:
        train_file_path = doc_path.replace('.docx', '_train.csv')
        predict_file_path = doc_path.replace('.docx', '_predict.csv')

        export = Export(doc_path)
        export.process(regexp)
        model = export.model

    cat_df = pd.read_csv('text/reorder_exit_codes.csv', encoding='utf-8-sig')
    train_df = pd.read_csv(train_file_path, encoding='utf-8')

    text = get_text(doc_path).replace("’", "'")

    print('writing sentences to predict csv...')
    start_writing = datetime.now()

    with open(predict_file_path, 'w', newline='', encoding='utf-8') as predict_file:
        writer = csv.writer(predict_file, delimiter=',')
        writer.writerow(['original_sentence', 'cleaned_sentence',
            'sentence_embedding'])

        train_df = pd.read_csv(train_file_path, encoding='utf-8')
        train_original_sentences = train_df['original_sentence'].tolist()

        # save which have been moved to train as re-classification
        # (before oversampling!!)
        moved_predict_df = train_df.loc[train_df['codes'].isna()]

        train_moved_sentences = moved_predict_df['original_sentence'].tolist()

        all_original_sentences = sent_tokenize(text)

        uncoded_original_sentences = []

        for sentence in all_original_sentences:
            if (sentence not in train_moved_sentences and
                sentence not in train_original_sentences and
                re.sub(regexp, '', sentence, flags=re.IGNORECASE)
                .strip() not in train_original_sentences and
                sentence[:-1] not in train_original_sentences): # to-do: better way
                uncoded_original_sentences.append(sentence)     # to check for "."

        sentence_embedding_list = []

        cleaned_sentence_embedding_dict = {}

        start_emb = datetime.now()

        if os.path.exists('text/embeddings.pickle'):
            with open('text/embeddings.pickle', 'rb') as handle:
                dict = pickle.load(handle)
                for sentence in uncoded_original_sentences:
                    try:
                        cleaned_sentence = dict[sentence][0]
                        sentence_embedding = dict[sentence][1]
                    except KeyError:
                        cleaned_sentence = remove_stop_words(
                            clean_sentence(sentence, regexp))
                        sentence_embedding = model.get_vector(cleaned_sentence)

                    writer.writerow([sentence, cleaned_sentence,
                        sentence_embedding])

                    sentence_embedding_list.append(sentence_embedding)

                    cleaned_sentence_embedding_dict[sentence] = [cleaned_sentence,
                        sentence_embedding]
        else:
            for sentence in uncoded_original_sentences:
                cleaned_sentence = remove_stop_words(clean_sentence(sentence, regexp))
                sentence_embedding = model.get_vector(cleaned_sentence)

                writer.writerow([sentence, cleaned_sentence, sentence_embedding])

                sentence_embedding_list.append(sentence_embedding)

                cleaned_sentence_embedding_dict[sentence] = [cleaned_sentence,
                    sentence_embedding]

        print(f'done writing data in csv in {datetime.now() - start_emb}')

        predict_file.close()


    # save sentence, cleaned_sentence, sentence_embedding dict to pickle
    with open('text/embeddings.pickle', 'wb') as handle:
        pickle.dump(cleaned_sentence_embedding_dict, handle,
            protocol=pickle.HIGHEST_PROTOCOL)
    #-------------------------------------------------------------------

    sentence_embedding_matrix = np.stack(sentence_embedding_list, axis=0)

    print(f'done writing in {datetime.now() - start_writing}')

    clf = ClassifierChain(classifier=XGBClassifier())

    _, _, _, _, _, accuracies, f_measures = classify(sentence_embedding_matrix,
        clf, 'ClassifierChain XGBoost oversample', True, False)

    print(f'accuracy per class = {accuracies}')
    print(f'f_measure per class = {f_measures}')


    print(f'script finished in {datetime.now() - start_script}')
