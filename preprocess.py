import re


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
