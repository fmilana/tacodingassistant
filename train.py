import tensorflow as tf
import numpy as np
import re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec


# pre process data
def clean_text(text):
    text = text.lower()
    # remove interviewer
    text = re.sub('iv[0-9]*[ \t].*', '', text)
    # remove interview format
    regexp = ('p[0-9]+\w*|speaker key|r*user\s*\d+( - study \d+)*|'
        '(iv[0-9]*|ie|um|a[0-9]+)\t|'
        '(interviewer|interviewee|person [0-9]|participant)|'
        '\d{2}:\d{2}:\d{2}|\[(.*?)\]|\[|\]')
    text = re.sub(regexp, '', text)
    # replace multiple spaces and newlines with one space
    text = re.sub('\s+', ' ', text)
    return text


def clean_sentence(sentence):
    return re.sub('[.,…:;–\'’!?-]', '', sentence)


# text = open('text\\joint_groupbuy_jhim.txt', 'r').read()
text = clean_text(open('text\\joint_reorder_exit.txt', 'r').read())
# tokenize and clean sentences
sentences = [clean_sentence(sentence) for sentence in sent_tokenize(text)]
# tokenize sentences into words and remove "." or "..." sentences
sentences_tokenized = [word_tokenize(sentence) for sentence in sentences]

# refer to here for all parameters:
# https://radimrehurek.com/gensim/models/word2vec.html
model = Word2Vec(sentences_tokenized, sg=1, size=100, window=5,
                 min_count=1, workers=4, iter=100)

# save model to file
model.save('./data/word2vec.model')
# for word2vec2tensor
model.wv.save_word2vec_format('./data/word2vecformat.model')

# for tensorboard:
# python -m gensim.scripts.word2vec2tensor -i data/word2vecformat.model -o data/
