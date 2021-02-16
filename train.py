import tensorflow as tf
import numpy as np
import re
import logging
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec


# pre process data
def clean_text(text):
    text = text.lower()
    # remove interviewer
    text = re.sub(r'iv[0-9]*[ \t].*', '', text)
    # remove interview format
    regexp = (r'p[0-9]+\w*|speaker key|r*user\s*\d+( - study \d+)*|'
              '(iv[0-9]*|ie|um|a[0-9]+)\t|'
              '(interviewer|interviewee|person [0-9]|participant)|'
              '\d{2}:\d{2}:\d{2}|\[(.*?)\]|\[|\]')
    text = re.sub(regexp, '', text)
    # replace "..." at the end of a line with "."
    text = re.sub(r'\.\.\.[\r\n]', '.', text)
    # replace multiple spaces or newlines with one space
    text = re.sub(r' +|[\r\n\t]+', ' ', text)
    return text


def clean_sentence(sentence):
    return re.sub(r'[^A-Za-z ]+', '', sentence)


def train(text):
    # remove interview format
    text = clean_text(text)
    # tokenize and clean sentences
    sentences = [clean_sentence(sentence) for sentence in sent_tokenize(text)]
    # tokenize sentences into words
    sentences_tokenized = [word_tokenize(sentence) for sentence in sentences]
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    # https://radimrehurek.com/gensim/models/word2vec.html
    model = Word2Vec(sentences_tokenized, sg=1, size=128, window=5,
        min_count=1, workers=4, iter=20)
    # save model to file
    model.save('./data/word2vec.model')
    # for word2vec2tensor (tensorboard)
    model.wv.save_word2vec_format('./data/word2vectf.model')
    print('training done')


# file_name = 'joint_groupbuy_jhim'
# train(open('text\\' + file_name + '.txt', 'r').read())

# for tensorboard:
# python -m gensim.scripts.word2vec2tensor -i data/word2vectf.model -o data/
