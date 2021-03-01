import sys, os
import zipfile
import csv
from nltk import sent_tokenize
from bs4 import BeautifulSoup
from preprocess import (
    clean_sentence,
    clean_text,
    remove_interviewer,
    remove_stop_words)
from lib.sentence2vec import Sentence2Vec


model = Sentence2Vec('word2vec-google-news-300')


def transverse(start, end, text):
    if start == end:
        return text
    for node in start.next_siblings:
        if node == end:
            # proper end
            return text

        if node.find('w:tab'):
            text += '\t'
        node = node.find('w:t')
        if node:
            text += node.text

    # if we get here it means we did not reach the end
    # ..so go up one level
    paragraph_node = start.parent

    # go to the next paragraph
    paragraph_siblings = paragraph_node.next_siblings
    next_paragraph = next(paragraph_siblings)
    while next_paragraph.name != 'w:p':
        if next_paragraph == end:
            return text
        else:
            next_paragraph = next(paragraph_siblings)
    # this is a paragraph
    next_paragraph_rows = next_paragraph.children
    first_row = next(next_paragraph_rows)
    return transverse(first_row, end, text + '\n')


def process(in_filename, out_filename):
    print('processing', in_filename)
    with zipfile.ZipFile(in_filename, 'r') as archive:
        writer = csv.writer(open(out_filename, 'w', newline=''))
        # csv header
        writer.writerow(['file name', 'comment id', 'original sentence',
            'cleaned sentence', 'sentence embedding', 'codes'])

        doc_xml = archive.read('word/document.xml')
        #doc_soup = BeautifulSoup(doc_xml, 'xml')
        doc_soup = BeautifulSoup(doc_xml, 'lxml')
        open('tmp.xml', 'w').write(doc_soup.prettify())
        comments_xml = archive.read('word/comments.xml')
        comments_soup = BeautifulSoup(comments_xml, 'xml')

        #print 'comments:'#, comments_soup.find_all('w:comment')
        for comment in comments_soup.find_all('w:comment'):
            codes = comment.find_all('w:t')
            codes = ''.join([x.text for x in codes])
            codes = codes.strip().rstrip().lower()
            comment_id = comment['w:id']

            range_start = doc_soup.find('w:commentrangestart',
                attrs={'w:id': comment_id})
            range_end = doc_soup.find('w:commentrangeend',
                attrs={'w:id': comment_id})

            text = transverse(range_start, range_end, '')
            text = text.replace('\n', ' ')

            text = remove_interviewer(text)

            sentence_to_cleaned_dict = {}
            # split text into sentences
            for sentence in sent_tokenize(text):
                cleaned_sentence = clean_sentence(
                    remove_stop_words(clean_text(sentence)))
                sentence_to_cleaned_dict[sentence] = [cleaned_sentence,
                    model.get_vector(cleaned_sentence)]

            for sentence, tuple in sentence_to_cleaned_dict.items():
                row = [in_filename, comment_id, sentence, tuple[0],
                    tuple[1], codes]
                writer.writerow(row)

        os.remove('tmp.xml')

src = sys.argv[1]
dst = sys.argv[2]

if os.path.isdir(src) and os.path.isdir(dst):
    # process all .docx files in the src folder
    files = [f for f in os.listdir(src) if f.endswith('.docx')]
    for f in files:
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f.replace('.docx', '_train.csv'))
        process(src_file, dst_file)
else:
    # src and dst are files
    process(src, dst)
