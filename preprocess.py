import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from path_util import resource_path


def remove_stop_words(text):
    stop_words = list(set(stopwords.words('english')))
    extra_stop_words = open(resource_path('data/stopwords/extra_stopwords.txt'), 'r', encoding='utf-8').read().split(',')

    stop_words += extra_stop_words

    word_tokens = [word for word in word_tokenize(text) if word.lower() not in stop_words]
    text = ' '.join(word_tokens)

    return text


def clean_sentence(sentence, keep_alphanum=False):
    sentence = re.sub(r'\t', ' ', sentence)    
    if not keep_alphanum:
        sentence = re.sub(r'[^A-Za-z\s]+', '', sentence)
    # sentence = re.sub(r'[ ]{2,}', ' ', sentence).strip()
    return sentence
