import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from path_util import get_correct_path


def remove_stop_words(text):
    stop_words = list(set(stopwords.words('english')))
    extra_stop_words = open(get_correct_path('text/extra_stopwords.txt'), 'r').read().split(',')

    stop_words += extra_stop_words

    word_tokens = [word for word in word_tokenize(text) if word.lower() not in stop_words]
    text = ' '.join(word_tokens)

    return text


def clean_sentence(sentence, filter_regexp, keep_alphanum=False):
    sentence = re.sub(r'\t', ' ', sentence)

    if filter_regexp != '':
        case_insensitive = (filter_regexp[-1] == 'i')

        regexp = re.sub(r'/([^/]*)$', '', filter_regexp[1:]) # strip regexp of / and flags (needed for the js side)

        if case_insensitive:
            regexp = re.compile(regexp, re.IGNORECASE)
        else:
            regexp = re.compile(regexp)

        sentence = re.sub(regexp, '', sentence)
    
    if not keep_alphanum:
        sentence = re.sub(r'[^A-Za-z\s]+', '', sentence)
    # sentence = re.sub(r'[ ]{2,}', ' ', sentence).strip()
    return sentence
