import csv
import re
import os
import pickle
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
# import seaborn as sns
import docx
import pandas as pd
from datetime import datetime
from nltk import sent_tokenize, word_tokenize, download, data
from path_util import resource_path
from sentence2vec import Sentence2Vec
from preprocess import (
    clean_sentence,
    remove_stop_words)
from augment import get_minority_samples, MLSMOTE
from collections import Counter
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import multilabel_confusion_matrix
from path_util import resource_path

# if running on MacOS, copy libomp.dylib and libomp.a in /usr/local/lib
from sys import platform
from pathlib import Path
from os.path import isfile
from shutil import copy

if platform == 'darwin':
    dylib_path = Path(resource_path('/usr/local/lib/libomp.dylib'))
    a_path = Path(resource_path('/usr/local/lib/libomp.a'))
    if not isfile(dylib_path): # if .dylib does not already exist
        copy(resource_path('libomp/libomp.dylib'), str(dylib_path))
        print('copied libomp.dylib to /usr/local/lib/')
    if not isfile(a_path): # if .a does not already exist
        copy(resource_path('libomp/libomp.a'), str(a_path))
        print('copied libomp.a to /usr/local/lib/')

from xgboost import XGBClassifier
import import_codes_from_document
import import_codes_from_folder


class ClassifyDocx:

    sentence2vec_model = None

    doc_path = ''
    cat_path = ''
    codes_folder_path = ''
    regexp = ''

    train_file_path = ''
    predict_file_path = ''

    cat_df = None
    original_train_df = None # save train_df copy here before test split and oversampling
    train_df = None
    moved_predict_df = None

    themes = None

    def __init__(self):
        # download nltk resources
        os.makedirs(os.path.dirname(resource_path('data/nltk/')), exist_ok=True)
        download('punkt', download_dir=resource_path('data/nltk/'))
        download('stopwords', download_dir=resource_path('data/nltk/'))
        data.path.append(resource_path('data/nltk/'))
        self.sentence2vec_model = Sentence2Vec()


    def set_up(self, transcript_path, codes_folder_path, theme_code_table_path, filter_regexp):
        self.doc_path = transcript_path
        self.codes_folder_path = codes_folder_path
        self.cat_path = theme_code_table_path
        self.regexp = filter_regexp


    """ def get_sample_weights(self, Y_train):
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

        return np.asarray(sample_weights) """


    def generate_training_and_testing_data(self, oversample, many_together):
        # themes_list = list(self.cat_df)
        self.original_train_df = self.train_df.copy()
        # convert embedding string to np array
        if not many_together:
            self.train_df['sentence_embedding'] = self.train_df['sentence_embedding'].apply(
                lambda x: np.fromstring(
                    x.replace('\n','')
                    .replace('[','')
                    .replace(']','')
                    .replace('  ',' '), sep=' '))

        # split into training and testing
        by_themes = self.train_df.groupby('themes')

        training_list = []
        testing_list = []
        # iterate by themes
        for name, group in by_themes:
            training = group.sample(frac=.8)
            testing = group.loc[~group.index.isin(training.index)]
            training_list.append(training)
            testing_list.append(testing)
        # create two new dataframes from the lists
        self.train_df = pd.concat(training_list)
        test_df = pd.concat(testing_list)

        # # oversample minority classes
        # if oversample:
        #     Y_train = self.train_df.iloc[:, 7:].to_numpy()
        #     class_dist = [x/Y_train.shape[0] for x in Y_train.sum(axis=0)]
        #     print(f'class distribution BEFORE MLSMOTE = {class_dist}')
        #     print(f'Y_train.shape[0] BEFORE MLSMOTE = {Y_train.shape[0]}')

        #     X = pd.DataFrame(np.array(self.train_df['sentence_embedding'].tolist()))
        #     Y = self.train_df.iloc[:, 7:]

        #     X_sub, Y_sub = get_minority_samples(X, Y)
        #     X_res, Y_res = MLSMOTE(X_sub, Y_sub, 300, 5)

        #     # Y_res.to_csv(resource_path('data/augmented_samples.csv'), index=False)            

        #     train_embedding_matrix = X.append(X_res).to_numpy()      # append augmented samples
        #     train_themes_binary_matrix = Y.append(Y_res).to_numpy()  # to original dataframes

        #     class_dist_os = [x/train_themes_binary_matrix.shape[0] for x in train_themes_binary_matrix.sum(axis=0)]
        #     print(f'class distribution AFTER MLSMOTE = {class_dist_os}')
        #     print(f'Y_train.shape[0] AFTER MLSMOTE = {train_themes_binary_matrix.shape[0]}')
        
        # else:
        train_embedding_matrix = np.array(self.train_df['sentence_embedding'].tolist())
        train_themes_binary_matrix = self.train_df.iloc[:, 7:].to_numpy()

        print(f'=========================> np.shape(train_embedding_matrix) = {np.shape(train_embedding_matrix)}')
        print(f'=========================> np.shape(train_themes_binary_matrix) = {np.shape(train_themes_binary_matrix)}')

        test_embedding_matrix = np.array(test_df['sentence_embedding'].tolist())
        test_cleaned_sentences = test_df['cleaned_sentence'].tolist()
        test_themes_binary_matrix = test_df.iloc[:, 7:].to_numpy()

        return (train_embedding_matrix, test_embedding_matrix,
            train_themes_binary_matrix, test_themes_binary_matrix,
            test_cleaned_sentences, self.themes)


    def add_classification_to_csv(self, prediction_output, prediction_proba):
        print(np.shape(prediction_output))
        if isinstance(prediction_output, scipy.sparse.spmatrix):
            out_df = pd.DataFrame.sparse.from_spmatrix(data=prediction_output,
                columns=self.themes)
        else:
            out_df = pd.DataFrame(data=prediction_output, columns=self.themes)

        proba_cols = [f'{theme} probability' for theme in self.themes]

        if isinstance(prediction_proba, scipy.sparse.spmatrix):
            proba_df = pd.DataFrame.sparse.from_spmatrix(data=prediction_proba,
                columns=proba_cols)
        else:
            proba_df = pd.DataFrame(data=prediction_proba, columns=proba_cols)

        new_df = pd.concat([out_df, proba_df], axis=1)

        predict_df = pd.read_csv(self.predict_file_path, encoding='utf-8')

        predict_df = predict_df.merge(new_df, left_index=True, right_index=True)

        if self.moved_predict_df is not None:
            # add moved predictions so they still show up in table as predictions
            self.moved_predict_df = self.moved_predict_df.drop([
                'file_name',
                'comment_id',
                'codes',
                'themes'], axis=1)

            for theme in self.themes:
                self.moved_predict_df[f'{theme} probability'] = self.moved_predict_df[theme]

            predict_df = predict_df.append(self.moved_predict_df)

            # remove moved predictions from train
            self.train_df = self.train_df[self.train_df['codes'].notna()]

        predict_df.to_csv(self.predict_file_path, index=False, encoding='utf-8-sig')
        self.moved_predict_df = None


    # def plot_heatmaps(self.clf_name, Y_true, Y_predicted, sentences_dict, themes_list):
    #     all_cms = self.multilabel_confusion_matrix(Y_true, Y_predicted.toarray())

    #     all_label_cms = self.get_keyword_labels(sentences_dict, themes_list)

    #     fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    #     for axes, labels, cm, theme in zip(ax.flatten(), all_label_cms, all_cms,
    #         themes_list):
    #         self.plot_multilabel_confusion_matrix(cm, labels, axes, theme, ['N', 'Y'])

    #     fig.suptitle(clf_name, fontsize=16)
    #     fig.tight_layout()

    #     plt.show()


    def write_cms_to_csv(self, sentences_dict, themes_list):
        cm_col_names = [
            'true_positives',
            'false_positives',
            'true_negatives',
            'false_negatives'
        ]

        for theme in themes_list:
            # start_path = re.search(r'^(.*[\\\/])', self.doc_path).group(0)
            end_path = re.search(r'([^\/]+).$', self.doc_path).group(0)
            end_path = end_path.replace('.docx', f'_{theme.replace(" ", "_")}_cm.csv')

            theme_cm_path = resource_path(f'data/cm/{end_path}')

            os.makedirs(os.path.dirname(theme_cm_path), exist_ok=True)
            with open(theme_cm_path, 'w', newline='', encoding='utf-8') as file:
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
                        original_sentence = str(self.original_train_df.loc[
                            self.original_train_df['cleaned_sentence'] == sentence]['original_sentence'].any())

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


    def get_keyword_labels(self, sentences_dict, themes_list):
        word_freq_dict = {}
        all_cms = []

        stop_words = open(resource_path('data/analysis_stopwords.txt'), 'r').read().split(',')

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


    # def plot_multilabel_confusion_matrix(self, cm, labels, axes, theme, 
    #     class_names, fontsize=14):
    #     annot = (np.asarray([f'{count}\n {keyword}'
    #         for keyword, count in zip(labels.flatten(), cm.flatten())])
    #         ).reshape(2, 2)

    #     cmap = sns.color_palette('ch:start=.2,rot=-.3', as_cmap=True)

    #     heatmap = sns.heatmap(cm, cmap=cmap, annot=annot, fmt='', cbar=False,
    #         xticklabels=class_names, yticklabels=class_names, ax=axes)
    #     sns.color_palette('ch:start=.2,rot=-.3', as_cmap=True)

    #     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
    #         ha='right', fontsize=fontsize)
    #     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
    #         ha='right', fontsize=fontsize)
    #     axes.set_ylabel('True label')
    #     axes.set_xlabel('Predicted label')
    #     axes.set_title(theme)


    def classify(self, sentence_embedding_matrix, clf, clf_name, oversample,
        many_together):
        print('running classify function...')
        start_function = datetime.now()

        print('running generate_training_and_testing_data...')
        start_gen = datetime.now()
        (X_train, X_test, Y_train, Y_test,
        test_cleaned_sentences, themes_list) = self.generate_training_and_testing_data(
            oversample, many_together)
        print(f'generate_training_and_testing_data run in {datetime.now() - start_gen}')

        print('fitting clf...')
        start_fit = datetime.now()

        print(f'======================================> np.shape(X_train) = {np.shape(X_train)}')
        print(f'======================================> np.shape(Y_train) = {np.shape(Y_train)}')

        clf.fit(X_train, Y_train)

        doc_file_name = re.search(r'([^\/]+).$', self.doc_path).group(0).replace('.docx', '')

        # save xgboost model in logs
        model_counter = 0
        while True:
            model_path = resource_path(f'logs/models/{doc_file_name}_xgbmodel_{model_counter}.pickle')
            if os.path.exists(model_path):
                model_counter += 1
            else:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as handle:
                    pickle.dump(clf, handle, protocol=4)
                break

        print(f'done fitting clf in {datetime.now() - start_fit}')

        print(f'generating confusion matrices...')
        start_cm = datetime.now()

        scores = []

        test_pred = clf.predict(X_test).toarray()

        for col in range(test_pred.shape[1]):
            equals = np.equal(test_pred[:, col], Y_test[:, col])
            score = np.sum(equals)/equals.size
            scores.append(score)

        print(f'======================================> np.shape(sentence_embedding_matrix) = {np.shape(sentence_embedding_matrix)}')

        prediction_output = clf.predict(sentence_embedding_matrix)

        print(f'======================================> np.shape(prediction_output) 1 = {np.shape(prediction_output)}')

        prediction_output = prediction_output.astype(int)

        print(f'======================================> np.shape(prediction_output) 2 = {np.shape(prediction_output)}')

        prediction_proba = clf.predict_proba(sentence_embedding_matrix)

        if not many_together:
            self.add_classification_to_csv(prediction_output, prediction_proba)

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

        self.write_cms_to_csv(sentences_dict, themes_list)

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


    def get_text(self, file_path):
        doc = docx.Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return '\n'.join(full_text)


    def run_classifier(self, modified_train_file_path=None):
        print('inside run_classifier.')
        start_script = datetime.now()

        if self.themes is None:
            # if from word
            if self.codes_folder_path == '':
                self.themes = import_codes_from_document.import_codes(self.sentence2vec_model, self.doc_path, self.cat_path, self.regexp)
            # if from nvivo
            else:
                self.themes = import_codes_from_folder.import_codes(self.sentence2vec_model, self.doc_path, self.codes_folder_path, self.cat_path, self.regexp)

        if modified_train_file_path is not None:
            self.train_file_path = modified_train_file_path
            self.predict_file_path = modified_train_file_path.replace('train', 'predict')
        else:
            self.train_file_path = self.doc_path.replace('.docx', '_train.csv')
            self.predict_file_path = self.doc_path.replace('.docx', '_predict.csv')

        if self.cat_path != '':
            self.cat_df = pd.read_csv(self.cat_path, encoding='utf-8-sig')

        self.train_df = pd.read_csv(self.train_file_path, encoding='utf-8')

        text = self.get_text(self.doc_path).replace("’", "'")

        print('writing sentences to predict csv...')
        start_writing = datetime.now()

        with open(self.predict_file_path, 'w', newline='', encoding='utf-8') as predict_file:
            writer = csv.writer(predict_file, delimiter=',')
            writer.writerow(['original_sentence', 'cleaned_sentence',
                'sentence_embedding'])

            train_original_sentences = self.train_df['original_sentence'].tolist()

            # save which have been moved to train as re-classification
            # (before oversampling!!)
            self.moved_predict_df = self.train_df.loc[self.train_df['codes'].isna()]

            train_moved_sentences = self.moved_predict_df['original_sentence'].tolist()

            all_original_sentences = sent_tokenize(text)

            uncoded_original_sentences = []

            for sentence in all_original_sentences:
                if (sentence not in train_moved_sentences and
                    sentence not in train_original_sentences and
                    re.sub(self.regexp, '', sentence, flags=re.IGNORECASE)
                    .strip() not in train_original_sentences and
                    sentence[:-1] not in train_original_sentences): # to-do: better way
                    uncoded_original_sentences.append(sentence)     # to check for "."

            sentence_embedding_list = []

            cleaned_sentence_embedding_dict = {}

            start_emb = datetime.now()

            if os.path.exists(resource_path('embeddings/embeddings.pickle')):
                with open(resource_path('embeddings/embeddings.pickle'), 'rb') as handle:
                    dict = pickle.load(handle)
                    for sentence in uncoded_original_sentences:
                        try:
                            cleaned_sentence = dict[sentence][0]
                            sentence_embedding = dict[sentence][1]
                        except KeyError:
                            cleaned_sentence = remove_stop_words(clean_sentence(sentence, self.regexp))
                            sentence_embedding = self.sentence2vec_model.get_vector(cleaned_sentence)

                        writer.writerow([sentence, cleaned_sentence,
                            sentence_embedding])

                        sentence_embedding_list.append(sentence_embedding)

                        cleaned_sentence_embedding_dict[sentence] = [cleaned_sentence, sentence_embedding]
            else:
                for sentence in uncoded_original_sentences:
                    cleaned_sentence = remove_stop_words(clean_sentence(sentence, self.regexp))
                    sentence_embedding = self.sentence2vec_model.get_vector(cleaned_sentence)

                    writer.writerow([sentence, cleaned_sentence, sentence_embedding])

                    sentence_embedding_list.append(sentence_embedding)

                    cleaned_sentence_embedding_dict[sentence] = [cleaned_sentence, sentence_embedding]

                with open(resource_path('embeddings/embeddings.pickle'), 'wb') as handle:
                    pickle.dump(cleaned_sentence_embedding_dict, handle, protocol=4)

            print(f'done writing data in csv in {datetime.now() - start_emb}')

            predict_file.close()


        # save sentence, cleaned_sentence, sentence_embedding dict to pickle
        with open(resource_path('embeddings/embeddings.pickle'), 'wb') as handle:
            pickle.dump(cleaned_sentence_embedding_dict, handle, protocol=4)
        #-------------------------------------------------------------------

        sentence_embedding_matrix = np.stack(sentence_embedding_list, axis=0)

        print(f'done writing in {datetime.now() - start_writing}')

        clf = ClassifierChain(classifier=XGBClassifier())

        _, _, _, _, _, accuracies, f_measures = self.classify(sentence_embedding_matrix,
            clf, 'ClassifierChain XGBoost oversample', True, False)

        print(f'accuracy per class = {accuracies}')
        print(f'f_measure per class = {f_measures}')


        print(f'script finished in {datetime.now() - start_script}')

        return self.themes
