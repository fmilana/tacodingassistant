import docx
import numpy as np
from export_docx_comments import process
from nltk import sent_tokenize
from sklearn.naive_bayes import MultinomialNB
from lib.sentence2vec import Sentence2Vec


def get_text(file_name):
    doc = docx.Document(file_name)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)


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


docx_name = 'joint_groupbuy_jhim.docx'
src_dir = 'text/'
dst_dir = 'text/'

process(src_dir, src_dir)

print('docx processed')

text = get_text(src_dir + docx_name)
clean_text = clean_text(text)

model = Sentence2Vec('word2vec-google-news-300')

clean_sentences = []

for sentence in sent_tokenize(clean_text):
    clean_sentence = self.clean_sentence(sentence)
    self.original_sentence_dict[clean_sentence] = sentence
    if not re.match('[.,…:;–\'’!?-]', clean_sentence):
        clean_sentences.append(clean_sentence)

self.sentence_embeddings = np.array([self.model.get_vector(sentence)
    for sentence in clean_sentences])





clf = MultinomialNB
