import re
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords


def remove_interviewer(text):
    text = re.sub(r'(?i)iv[0-9]*[ \t].*', '', text)
    text = re.sub(r'[ ]{2,}', ' ', text).strip()
    return text


def remove_interview_format(text):
    text = text.lower()
    # remove interview format
    regexp = (r'p[0-9]+\w*|speaker key|r*user\s*\d+( - study \d+)*|'
              '(iv[0-9]*|ie|um|a[0-9]+)\t|'
              '(interviewer|interviewee|person [0-9]|participant)|'
              '\d{2}:\d{2}:\d{2}|\[(.*?)\]|\[|\]')
    text = re.sub(regexp, '', text)
    # replace '...' at the end of a line with '.'
    text = re.sub(r'\.\.\.[\r\n]', '.', text)
    # replace multiple spaces or newlines with one space
    text = re.sub(r' +|[\r\n\t]+', ' ', text)
    # replace multiple spaces with one and strip string
    text = re.sub(r'[ ]{2,}', ' ', text).strip()
    return text


def remove_stop_words(text):
    stopwords_text = open('text/gist_stopwords.txt', 'r').read()
    stopwords = stopwords_text.split(',')

    word_tokens = [word for word in word_tokenize(text)
        if word not in stopwords]
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
