import sys, os
import re
import zipfile
import csv
import docx
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from sklearn.naive_bayes import MultinomialNB
from lib.sentence2vec import Sentence2Vec
from preprocess import clean_text, clean_sentence


model = Sentence2Vec('word2vec-google-news-300')
sentence_embeddings_codes_dict = {}


def get_text(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)


def generate_sentence_embeddings_codes_dict(file_name):
    docx_path = 'text/' + file_name + '.docx'
    csv_path = 'text/' + file_name + '.csv'

    text = get_text(docx_path)

    cleaned_sentences = []
    for sentence in sent_tokenize(text):
        cleaned_sentence = clean_sentence(clean_text(sentence))
        if not re.match('[.,…:;–\'’!?-]', cleaned_sentence):
            cleaned_sentences.append(cleaned_sentence)

    df = pd.read_csv(csv_path, encoding='Windows-1252')
    for cleaned_sentence in cleaned_sentences:
        csv_df = df[df.iloc[:, 3].str.match(r'^' + cleaned_sentence + '$',
            na=False)]
        codes = []
        if not csv_df.empty:
            codes = list(csv_df.iloc[:, 4])
        sentence_embeddings_codes_dict[
            np.array([model.get_vector(cleaned_sentence)]).tobytes()] = codes
