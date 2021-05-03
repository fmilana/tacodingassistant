import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# def remove_interviewer(text):
#     text = re.sub(r'(?i)(p1|iv[0-9]*|a[0-9]+)[ \t].*', '', text)
#     # text = re.sub(r'[ ]{2,}', ' ', text).strip()
#     return text


def remove_interview_format(text, lower=True):
    if lower:
        text = text.lower()
    # remove interview format
    regexp = r'(?i)(^p[0-9]+exit.*$|speaker key|^iv.*$|p[\d]+[a-z]*|a\d|\d\d:\d\d:\d\d|participant|interviewer|interviewee|person \d|^p\d_.*$|\bie\b|\bum\b)[:]*'
    text = re.sub(regexp, '', text, flags=re.MULTILINE)
    # replace '...' at the end of a line with '.'
    text = re.sub(r'\.\.\.[\r\n]', '.', text)
    # replace multiple spaces or newlines with one space
    # text = re.sub(r' +|[\r\n\t]+', ' ', text)
    # replace multiple spaces with one and strip string
    # text = re.sub(r'[ ]{2,}', ' ', text).strip()
    return text


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    extra_stop_words_text = open('text/extra_stopwords.txt', 'r').read()
    extra_stop_words = extra_stop_words_text.split(',')

    word_tokens = [word for word in word_tokenize(text)
        if word not in stop_words and word not in extra_stop_words]
    text = ' '.join(word_tokens)
    return text


def clean_sentence(sentence):
    sentence = re.sub(r'[^A-Za-z ]+', '', sentence)
    sentence = re.sub(r'[ ]{2,}', ' ', sentence).strip()
    return sentence


## testing functions:
# text = 'Yes, that\'s right'
# after_remove_interviewer = remove_interviewer(text)
# after_remove_interview_format = remove_interview_format(after_remove_interviewer)
# after_remove_stopwords = remove_stop_words(after_remove_interview_format)
# after_clean_sentence = clean_sentence(after_remove_stopwords)
#
# print(f'after remove_interviewer: {after_remove_interviewer}')
# print(f'after remove_interview_format: {after_remove_interview_format}')
# print(f'after remove_stop_words: {after_remove_stopwords}')
# print(f'after clean_sentence: {after_clean_sentence}')

# Order:
# 1. remove interviewer
# 2. remove interview format
# 3. remove stopwords
# 4. clean_sentence
