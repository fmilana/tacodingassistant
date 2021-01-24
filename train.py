import tensorflow as tf
import numpy as np
import re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec


# pre processing data
def clean(sentence):
    sentence = sentence.lower()
    # remove interview format terms
    # regexp = '[.,…:;–\'’!?-]|speaker key|r*user\s*\d+( - study \d+)*|\
    #          \b(iv|ie|p1|p2|p3)\t|\b(interviewer|interviewee|person 1|\
    #          person 2|person 3)\b|\d{2}:\d{2}:\d{2}|\[(.*?)\]'
    regexp = 'speaker key|r*user\s*\d+( - study \d+)*|(iv|ie|p1|p2|p3)\t|(interviewer|interviewee|person 1|person 2|person 3)|\d{2}:\d{2}:\d{2}|\[(.*?)\]'
    sentence = re.sub(regexp, '', sentence)
    # remove stop words
    # sentence = ' '.join([word for word in sentence.split()
    #                    if word not in stopwords.words('english')])

    return sentence


text = open('text\\joint_groupbuy_jhim.txt', 'r').read()
sentences = [clean(sentence) for sentence in sent_tokenize(text)]
# tokenizes and removes "." or "..." sentences
sentences_tokenized = [word_tokenize(sentence) for sentence in sentences
    if not re.match('[.,…:;–\'’!?-]', sentence)]

# refer to here for all parameters:
# https://radimrehurek.com/gensim/models/word2vec.html
model = Word2Vec(sentences_tokenized, sg=1, size=100, window=5,
                 min_count=1, workers=4, iter=1000)

# save model to file
model.save('./data/word2vec.model')
# for word2vec2tensor
model.wv.save_word2vec_format('./data/word2vecformat.model')

# for tensorboard
# python -m gensim.scripts.word2vec2tensor -i data/word2vecformat.model -o data/
