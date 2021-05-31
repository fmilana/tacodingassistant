import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    extra_stop_words = open('text/extra_stopwords.txt', 'r').read().split(',')

    word_tokens = [word for word in word_tokenize(text)
        if word not in stop_words and word not in extra_stop_words]
    text = ' '.join(word_tokens)
    return text


def clean_sentence(sentence, keep_alphanum=False):
    regexp = r'(?i)(^p[0-9]+exit.*$|\biv\d+S*|\bp\d+\S*|a\d|\d\d:\d\d:\d\d|participant|interviewer|interviewee|person \d|^p\d_.*$|\bie\b|\bum\b)[:]*'
    sentence = re.sub(regexp, '', sentence, flags=re.MULTILINE)
    if not keep_alphanum:
        sentence = re.sub(r'[^A-Za-z ]+', '', sentence)
    # sentence = re.sub(r'[ ]{2,}', ' ', sentence).strip()
    return sentence
