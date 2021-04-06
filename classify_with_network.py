import csv
import docx
import codecs
import pickle
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import sent_tokenize
from preprocess import (
    clean_sentence,
    remove_interviewer,
    remove_interview_format,
    remove_stop_words)


coded_file_path = 'text/reorder_exit_train.csv'
pred_file_path = 'text/reorder_exit_predict.csv'
cat_file_path = 'text/reorder_categories.csv'
docx_file_path = 'text/reorder_exit.docx'

codecs.register_error('strict', codecs.ignore_errors)

coded_df = pd.read_csv(coded_file_path, encoding='Windows-1252')
cat_df = pd.read_csv(cat_file_path, encoding='Windows-1252')


def add_classification_to_csv(pred, pred_proba):
    themes_list = cat_df.category.unique()

    if isinstance(pred, scipy.sparse.spmatrix):
        out_df = pd.DataFrame.sparse.from_spmatrix(data=pred,
            columns=themes_list)
    else:
        out_df = pd.DataFrame(data=pred, columns=themes_list)

    proba_cols = [f'{theme} probability' for theme in themes_list]

    if isinstance(pred_proba, scipy.sparse.spmatrix):
        proba_df = pd.DataFrame.sparse.from_spmatrix(data=pred_proba,
            columns=proba_cols)
    else:
        proba_df = pd.DataFrame(data=pred_proba, columns=proba_cols)

    new_df = pd.concat([out_df, proba_df], axis=1)

    pred_df = pd.read_csv(pred_file_path, encoding='Windows-1252')

    merged_df = pred_df.merge(new_df, left_index=True, right_index=True)
    merged_df.to_csv(pred_file_path, index=False)


def get_data():
    doc = docx.Document(docx_file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)

    text = '\n'.join(full_text)
    text = remove_interviewer(text)

    filtered_df = coded_df[coded_df['cleaned sentence'] != '']
    filtered_df.dropna(inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    coded_original_sentences = filtered_df['original sentence'].tolist()

    all_original_sentences = sent_tokenize(text)

    all_cleaned_sentences = [clean_sentence(remove_stop_words(
        remove_interview_format(sentence))) for sentence
        in all_original_sentences]

    max_sentence_length = len(max(all_cleaned_sentences))

    uncoded_original_sentences =  [sentence for sentence
        in all_original_sentences if sentence not in coded_original_sentences
        and isinstance(sentence, str)]

    uncoded_clean_sentences = []

    with open(pred_file_path, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['file name', 'original sentence', 'cleaned sentence'])

        for sentence in uncoded_original_sentences:
            cleaned_sentence = clean_sentence(remove_stop_words(
                remove_interview_format(sentence)))
            uncoded_clean_sentences.append(cleaned_sentence)
            writer.writerow([docx_file_path, sentence, cleaned_sentence])

        file.close()

    # load same tokenizer used for training
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequences = tokenizer.texts_to_sequences(uncoded_clean_sentences)

    data = pad_sequences(sequences, padding='post', maxlen=max_sentence_length)

    return data


def classify_docx(model):
    data = get_data()
    pred_proba = model.predict(data)
    pred = np.rint(pred_proba)

    add_classification_to_csv(pred, pred_proba)


model = load_model('models/pretrained_lstm_model.h5', compile=False)

classify_docx(model)
