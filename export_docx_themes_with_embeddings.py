import sys, os
import zipfile
import csv
import pandas as pd
from nltk import sent_tokenize
from bs4 import BeautifulSoup
from preprocess import (
    clean_sentence,
    # remove_interview_format,
    # remove_interviewer,
    remove_stop_words)
from lib.sentence2vec import Sentence2Vec

class Export():
    model = Sentence2Vec()

    doc_path = ''
    cat_df = pd.read_csv('text/reorder_categories.csv')


    def __init__(self, doc_path):
        self.doc_path = doc_path


    def transverse(self, start, end, text):
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
        return self.transverse(first_row, end, text + '\n')


    def process(self):
        print('processing', self.doc_path)
        with zipfile.ZipFile(self.doc_path, 'r') as archive:
            # write header
            header = ['file name', 'comment id', 'original sentence',
                'cleaned sentence', 'sentence embedding', 'codes', 'themes']
            themes_list = self.cat_df.category.unique()
            header.extend(themes_list)

            out_filename = self.doc_path.replace('.docx', '_train.csv')

            writer = csv.writer(open(out_filename, 'w', newline=''))
            writer.writerow(header)

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

                text = self.transverse(range_start, range_end, '')
                text = text.replace('"', "'")
                text = text.replace("’", "'")

                # text = remove_interviewer(text)

                sentence_to_cleaned_dict = {}
                # split text into sentences
                for sentence in sent_tokenize(text):
                    cleaned_sentence = remove_stop_words(
                        # remove_stop_words(remove_interview_format(sentence)))
                        clean_sentence(sentence))
                    sentence_to_cleaned_dict[sentence] = [cleaned_sentence,
                        self.model.get_vector(cleaned_sentence)]

                if ';' in codes:
                    themes = []
                    for code in codes.split('; '):
                        theme_df = self.cat_df.loc[self.cat_df['code'] == code, 'category']
                        if theme_df.shape[0] > 0:
                            theme = theme_df.item()
                        # else:
                        #     theme = 'none'
                        if theme not in themes:
                            themes.append(theme)
                    themes = sorted(themes, key=str.casefold)
                    themes = '; '.join(themes)
                else:
                    theme_df = self.cat_df.loc[self.cat_df['code'] == codes, 'category']
                    if theme_df.shape[0] > 0:
                        themes = theme_df.item()
                    # else:
                    #     themes = 'none'

                if len(themes) > 0:
                    for sentence, tuple in sentence_to_cleaned_dict.items():
                        themes_binary = []
                        for theme in themes_list:
                            if theme in themes:
                                themes_binary.append(1)
                            else:
                                themes_binary.append(0)
                        row = [self.doc_path, comment_id, sentence, tuple[0],
                            tuple[1], codes, themes]
                        row.extend(themes_binary)
                        writer.writerow(row)

            os.remove('tmp.xml')
