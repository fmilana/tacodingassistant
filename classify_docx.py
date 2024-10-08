import csv
import re
import os
import pickle
import pandas as pd
import numpy as np
import scipy
import docx
import pandas as pd
import warnings
from datetime import datetime
from nltk import sent_tokenize, word_tokenize, download, data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import jaccard_score, f1_score
from path_util import resource_path
from sentence2vec import Sentence2Vec
from preprocess import clean_sentence, remove_stop_words
from augment import get_minority_samples, MLSMOTE
from collections import Counter
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier
from import_codes import import_codes_from_word, import_codes_from_nvivo, import_codes_from_maxqda, import_codes_from_dedoose


class ClassifyDocx:
    warnings.filterwarnings(action='ignore', category=UserWarning)

    sentence2vec_model = None

    software_used = ''

    doc_path = ''
    cat_path = ''
    nvivo_codes_folder_path = ''
    maxqda_document_path = ''
    dedoose_excerpts_path = ''
    delimiter = ''
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


    def set_up(self, transcript_path, software, word_delimiter, nvivo_codes_folder_path, maxqda_doc_path, dedoose_excerpts_path, theme_code_table_path, filter_regexp):
        self.doc_path = transcript_path
        self.software_used = software
        self.delimiter = word_delimiter
        self.nvivo_codes_folder_path = nvivo_codes_folder_path
        self.maxqda_document_path = maxqda_doc_path
        self.dedoose_excerpts_path = dedoose_excerpts_path
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
    

    def log_scores(self, weighted_f1_score, weighted_jaccard_score):
        with open(resource_path('logs/app.log'), 'a+', encoding='utf-8') as f:
            line = f'=====================================> MODEL SCORES: weighted f1 = {weighted_f1_score} weighted jaccard = {weighted_jaccard_score}\n'
            f.write(line)
            f.close()

    
    def generate_training_and_testing_data(self, oversample):
        self.original_train_df = self.train_df.copy()
        
        X = np.array([np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ') 
                      for x in self.train_df['sentence_embedding']])
        Y = self.train_df.iloc[:, 7:].to_numpy()

        X_train, X_test, Y_train, Y_test, train_indices, test_indices = train_test_split(
            X, Y, np.arange(X.shape[0]), test_size=0.2, random_state=42)

        if oversample:
            class_dist = Y_train.sum(axis=0) / Y_train.shape[0]
            print('checking for minority classes in train split...')
            X_sub, Y_sub = get_minority_samples(pd.DataFrame(X_train), pd.DataFrame(Y_train))
            if X_sub.shape[0] > 0:
                print('minority classes found.')
                print('oversampling...')
                try:
                    X_res, Y_res = MLSMOTE(X_sub, Y_sub, round(X_train.shape[0]))
                    X_train = np.concatenate((X_train, X_res.to_numpy()))
                    Y_train = np.concatenate((Y_train, Y_res.to_numpy()))
                    print('oversampled.')
                    class_dist_os = Y_train.sum(axis=0) / Y_train.shape[0]
                    print(f'class distribution BEFORE MLSMOTE: {class_dist}')
                    print(f'class distribution AFTER MLSMOTE: {class_dist_os}')
                except ValueError:
                    print('could not oversample because n_samples < n_neighbors in some classes')
            else:
                print('no minority classes.')

        test_cleaned_sentences = self.train_df.iloc[test_indices]['cleaned_sentence'].tolist()

        return X_train, X_test, Y_train, Y_test, test_cleaned_sentences, self.themes


    def add_classification_to_csv(self, prediction_output, prediction_proba):
        print(f'np.shape(prediction_output) = {np.shape(prediction_output)}')
        if isinstance(prediction_output, scipy.sparse.spmatrix):
            out_df = pd.DataFrame.sparse.from_spmatrix(data=prediction_output, columns=self.themes)
        else:
            out_df = pd.DataFrame(data=prediction_output, columns=self.themes)

        proba_cols = [f'{theme} probability' for theme in self.themes]

        if isinstance(prediction_proba, scipy.sparse.spmatrix):
            proba_df = pd.DataFrame.sparse.from_spmatrix(data=prediction_proba,
                columns=proba_cols)
        else:
            proba_df = pd.DataFrame(data=prediction_proba, columns=proba_cols)

        new_df = pd.concat([out_df, proba_df], axis=1)

        predict_df = pd.read_csv(self.predict_file_path, encoding='utf-8-sig', encoding_errors='replace')

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

        predict_df.to_csv(self.predict_file_path, index=False, encoding='utf-8-sig', errors='replace')
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
        cm_col_names = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']

        for theme in themes_list:
            # start_path = re.search(r'^(.*[\\\/])', self.doc_path).group(0)
            end_path = re.search(r'([^\/]+).$', self.doc_path).group(0)
            end_path = end_path.replace('.docx', f'_{theme.replace(" ", "_")}_cm.csv')

            theme_cm_path = resource_path(f'data/documents/confusion_tables/{end_path}')

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
                        # original_sentence = str(self.original_train_df.loc[self.original_train_df['cleaned_sentence'] == sentence]['original_sentence'].any())

                        # pandas 1.3.0
                        match_df = self.original_train_df.loc[self.original_train_df['cleaned_sentence'] == sentence, 'original_sentence']
                        if len(match_df) == 0:
                            continue
                        elif len (match_df) > 1:
                            match_df = match_df.iloc[:1]
                        original_sentence = match_df.item()

                        if len(original_sentence) > 0:
                            # # remove interview artifacts (not stopwords)
                            # original_sentence = clean_sentence(original_sentence,
                            #     keep_alphanum=True)
                            original_sentence.replace('…', '...')

                            if len(original_sentence) > 0:
                                original_sentences.append(original_sentence)

                    emptyToAdd = max(lengths) - len(original_sentences)

                    for _ in range(emptyToAdd):
                        original_sentences.append('')

                    all_sentences_lists.append(original_sentences)

                zipped = zip(*[sentences for sentences in all_sentences_lists])

                for row in zipped:
                    writer.writerow(row)
                file.close()


    def get_keyword_labels(self, sentences_dict, themes_list):
        word_freq_dict = {}
        all_cms = []

        stop_words = open(resource_path('data/stopwords/analysis_stopwords.txt'), 'r', encoding='utf-8').read().split(',')

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
                counter_freq = Counter([word for word in word_tokenize(joined_sentences) if word not in stop_words])

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


    def train_and_export_model(self, X, Y, themes_list):
        clf = XGBClassifier(
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        chains = [ClassifierChain(clf, order='random', random_state=i) for i in range(10)]
        
        for chain in chains:
            chain.fit(X, Y)
        
        model_file_path = resource_path('data/models/chains.pkl')
        themes_file_path = resource_path('data/models/themes.txt')

        with open(model_file_path, 'wb') as file:  
            pickle.dump(chains, file)

        with open(themes_file_path, 'w') as file:
            file.write('\n'.join(themes_list))


    def classify(self, sentence_embedding_matrix, chains, minimum_proba, oversample=True):
        print('running classify function...')
        start_function = datetime.now()

        print('running generate_training_and_testing_data...')
        start_gen = datetime.now()
        X_train, X_test, Y_train, Y_test, test_cleaned_sentences, themes_list = self.generate_training_and_testing_data(oversample)
        print(f'generate_training_and_testing_data run in {datetime.now() - start_gen}')

        print('fitting clf...')
        start_fit = datetime.now()

        X = np.concatenate((X_train, X_test))
        Y = np.concatenate((Y_train, Y_test))

        label_encoder = LabelEncoder()
        for col in range(Y.shape[1]):
            Y[:, col] = label_encoder.fit_transform(Y[:, col])

        print(f'np.shape(X) = {np.shape(X)}')
        print(f'np.shape(Y) = {np.shape(Y)}')

        # train and export model for transfer learning in transfer branch
        self.train_and_export_model(X, Y, themes_list)
        # ------------------------------------------------------------------

        for i, chain in enumerate(chains):
            chain.fit(X_train, Y_train)
            print(f'{i+1}/{len(chains)} chains fit')

        # doc_file_name = re.search(r'([^\/]+).$', self.doc_path).group(0).replace('.docx', '')

        # save xgboost model in logs
        # model_counter = 0
        # while True:
        #     model_path = resource_path(f'logs/models/{doc_file_name}_xgbmodel_{model_counter}.pkl')
        #     if os.path.exists(model_path):
        #         model_counter += 1
        #     else:
        #         os.makedirs(os.path.dirname(model_path), exist_ok=True)
        #         with open(model_path, 'wb') as handle:
        #             pickle.dump(clf, handle, protocol=4)
        #             handle.close()
        #         break

        print(f'done fitting clf in {datetime.now() - start_fit}')

        print(f'generating confusion matrices...')
        start_cm = datetime.now()

        Y_test_pred = np.mean([chain.predict_proba(X_test) for chain in chains], axis=0)
        Y_test_pred = (Y_test_pred >= minimum_proba).astype(int)

        weighted_f1_score = f1_score(Y_test, Y_test_pred, average='weighted')
        weighted_jaccard_score = jaccard_score(Y_test, Y_test_pred, average='weighted')

        prediction_proba = np.mean([chain.predict_proba(sentence_embedding_matrix) for chain in chains], axis=0)
        prediction_output = (prediction_proba >= minimum_proba).astype(int)

        self.add_classification_to_csv(prediction_output, prediction_proba)

        sentences_dict = {f'{class_name} {category}': [] for class_name in themes_list for category in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']}

        for col, class_name in enumerate(themes_list):
            for row in range(Y_test_pred.shape[0]):
                category = f'{["true", "false"][Y_test_pred[row, col] != Y_test[row, col]]}_{["negative", "positive"][Y_test_pred[row, col]]}'
                sentences_dict[f'{class_name} {category}s'].append(test_cleaned_sentences[row])

        self.write_cms_to_csv(sentences_dict, themes_list)

        print(f'confusion matrices created in {datetime.now() - start_cm}')
        print(f'classify function run in {datetime.now() - start_function}')

        return weighted_f1_score, weighted_jaccard_score


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
            # if from Word
            if self.software_used == 'Word':
                self.themes = import_codes_from_word(self.sentence2vec_model, self.doc_path, self.delimiter, self.cat_path, self.regexp)
            # if from NVivo
            elif self.software_used == 'NVivo':
                self.themes = import_codes_from_nvivo(self.sentence2vec_model, self.doc_path, self.nvivo_codes_folder_path, self.cat_path, self.regexp)
            # if from MAXQDA
            elif self.software_used == 'MAXQDA':
                self.themes = import_codes_from_maxqda(self.sentence2vec_model, self.doc_path, self.maxqda_document_path, self.cat_path, self.regexp)
            # if from Dedoose
            elif self.software_used == 'Dedoose':
                self.themes = import_codes_from_dedoose(self.sentence2vec_model, self.doc_path, self.dedoose_excerpts_path, self.cat_path, self.regexp)            

        if modified_train_file_path is not None:
            self.train_file_path = modified_train_file_path
            self.predict_file_path = modified_train_file_path.replace('train', 'predict')
        else:
            self.train_file_path = self.doc_path.replace('.docx', '_train.csv')
            self.predict_file_path = self.doc_path.replace('.docx', '_predict.csv')

        if self.cat_path != '':
            self.cat_df = pd.read_csv(self.cat_path, encoding='utf-8-sig', encoding_errors='replace').applymap(lambda x: x.lower() if type(x) == str else x)
            self.cat_df.columns = self.cat_df.columns.str.lower()

        self.train_df = pd.read_csv(self.train_file_path, encoding='utf-8-sig', encoding_errors='replace')

        text = self.get_text(self.doc_path).replace("’", "'").replace("“", "'").replace("”", "'")

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

            if os.path.exists(resource_path('data/embeddings/embeddings.pkl')):
                with open(resource_path('data/embeddings/embeddings.pkl'), 'rb') as handle:
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

                        handle.close()
            else:
                for sentence in uncoded_original_sentences:
                    cleaned_sentence = remove_stop_words(clean_sentence(sentence, self.regexp))
                    sentence_embedding = self.sentence2vec_model.get_vector(cleaned_sentence)

                    writer.writerow([sentence, cleaned_sentence, sentence_embedding])

                    sentence_embedding_list.append(sentence_embedding)

                    cleaned_sentence_embedding_dict[sentence] = [cleaned_sentence, sentence_embedding]

                with open(resource_path('data/embeddings/embeddings.pkl'), 'wb') as handle:
                    pickle.dump(cleaned_sentence_embedding_dict, handle, protocol=4)
                    handle.close()

            print(f'done writing data in csv in {datetime.now() - start_emb}')

            predict_file.close()


        # save sentence, cleaned_sentence, sentence_embedding dict to pickle
        with open(resource_path('data/embeddings/embeddings.pkl'), 'wb') as handle:
            pickle.dump(cleaned_sentence_embedding_dict, handle, protocol=4)
            handle.close()
        #-------------------------------------------------------------------

        sentence_embedding_matrix = np.stack(sentence_embedding_list, axis=0)

        print(f'done writing in {datetime.now() - start_writing}')

        # to-do: tune best params from gridsearchcv on reorder_exit training data
        # currently using default params
        clf = XGBClassifier(
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
            # n_estimators=100,
            # learning_rate=0.3,
            # gamma=0,
            # max_depth=6,
            # min_child_weight=1,
            # max_delta_step=0,
            # subsample=1,
            # colsample_bytree=1,
            # colsample_bylevel=1,
            # reg_lambda=1,
            # reg_alpha=0,
            # scale_pos_weight=1
        )

        number_of_chains = 10

        chains = [ClassifierChain(clf, order='random', random_state=i) for i in range(number_of_chains)]

        minimum_proba = 0.75

        weighted_f1_score, weighted_jaccard_score = self.classify(sentence_embedding_matrix, chains, minimum_proba)

        print(f'weighted f1 score = {weighted_f1_score}')
        print(f'weighted jaccard score = {weighted_jaccard_score}')

        self.log_scores(weighted_f1_score, weighted_jaccard_score)

        print(f'script finished in {datetime.now() - start_script}')

        return self.themes
