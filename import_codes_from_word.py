import os
import re
import zipfile
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from nltk import sent_tokenize
from bs4 import BeautifulSoup
from preprocess import (
    clean_sentence,
    remove_stop_words)


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


# doc_path and theme_code_table_path documents already copied in data folder
def import_codes(sentence2vec_model, doc_path, delimiter, theme_code_table_path, regexp):
    start = datetime.now()
    cat_df = pd.read_csv(theme_code_table_path, encoding='utf-8-sig').applymap(lambda x: x.lower() if type(x) == str else x)
    cat_df.columns = cat_df.columns.str.lower()

    with zipfile.ZipFile(doc_path, 'r') as archive:
        # write header
        header = ['file_name', 'comment_id', 'original_sentence',
            'cleaned_sentence', 'sentence_embedding', 'codes', 'themes']
        themes_list = list(cat_df)
        header.extend(themes_list)

        out_filename = doc_path.replace('.docx', '_train.csv')

        writer = csv.writer(open(out_filename, 'w', newline='',
            encoding='utf-8'))
        writer.writerow(header)

        doc_xml = archive.read('word/document.xml')
        doc_soup = BeautifulSoup(doc_xml, 'lxml')
        # open('tmp.xml', 'w').write(doc_soup.prettify())
        comments_xml = archive.read('word/comments.xml')
        comments_soup = BeautifulSoup(comments_xml, 'xml')

        missing_codes = []

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
            text = text.replace('"', "'")
            text = text.replace("’", "'")
            text = text.replace("´", "'")
            text = text.replace("…", "...")
            text = text.replace("\\", "\\\\")

            sentence_to_cleaned_dict = {}
            # split text into sentences
            for sentence in sent_tokenize(text):
                cleaned_sentence = remove_stop_words(
                    # remove_stop_words(remove_interview_format(sentence)))
                    clean_sentence(sentence, regexp))
                sentence_to_cleaned_dict[sentence] = [cleaned_sentence,
                    sentence2vec_model.get_vector(cleaned_sentence)]

            themes = []

            if delimiter != '' and delimiter in codes:
                for code in codes.split(delimiter):
                    code = code.strip()
                    find_code = (cat_df.values == code).any(axis=0)
                    try:
                        theme = cat_df.columns[np.where(find_code==True)[0]].item()
                        if theme not in themes:
                            themes.append(theme)
                    except ValueError:
                        missing_codes.append(code)
                themes = sorted(themes, key=str.casefold)
                themes = '; '.join(themes)
            else:
                find_code = (cat_df.values == codes).any(axis=0)
                try:
                    themes = cat_df.columns[np.where(find_code==True)[0]].item()
                except ValueError:
                    missing_codes.append(codes)

            if len(themes) > 0:
                for sentence, tuple in sentence_to_cleaned_dict.items():
                    themes_binary = []
                    for theme in themes_list:
                        if theme in themes:
                            themes_binary.append(1)
                        else:
                            themes_binary.append(0)
                    row = [re.search(r'([^\/]+).$', doc_path).group(0), 
                        comment_id, sentence, tuple[0], tuple[1], codes, themes]
                    row.extend(themes_binary)
                    writer.writerow(row)

        print(f'done extracting in {datetime.now() - start}')

        print(f'{len(set(missing_codes))} missing codes ({len(missing_codes)} sentences) in themes table (some counters in the codes table will be 0)')
        # print(set(missing_codes))

        # os.remove('tmp.xml')

        # returning ALL themes from cat_df, not just those found (to-do)
        return themes_list


def create_codes_csv_from_word(doc_path, delimiter):
    theme_code_table_path = os.path.join(Path(doc_path).parent.absolute(), doc_path.replace('.docx', '_codes.csv'))

    if not os.path.isfile(theme_code_table_path):
        print(f'extracting codes from {doc_path}...')
        all_codes = []

        with zipfile.ZipFile(doc_path, 'r') as archive:
            comments_xml = archive.read('word/comments.xml')
            comments_soup = BeautifulSoup(comments_xml, 'xml')

            for comment in comments_soup.find_all('w:comment'):
                codes = comment.find_all('w:t')
                codes = ''.join([x.text for x in codes])
                codes = codes.strip().rstrip().lower()

                if delimiter != '' and delimiter in codes:
                    for code in codes.split(delimiter):
                        code = code.strip()
                        if len(code) > 0:
                            all_codes.append(code)
                else:
                    if len(codes) > 0:
                        all_codes.append(codes)

        codes_df = pd.DataFrame({'Theme 1 (replace this)': sorted(list(set(all_codes))), 'Theme 2 (replace this)': np.nan, 'Theme 3 (replace this)': np.nan, '...': np.nan})
        codes_df.to_csv(theme_code_table_path, index=False, encoding='utf-8-sig')

        print(f'{doc_path.replace(".docx", "_codes.csv")} created in {Path(doc_path).parent.absolute()}')
    else:
        print(f'code table already exists in {theme_code_table_path}')

    return theme_code_table_path