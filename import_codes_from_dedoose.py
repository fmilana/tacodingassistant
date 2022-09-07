import os
import docx
import re
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from nltk import sent_tokenize
from preprocess import clean_sentence, remove_stop_words


def create_codes_csv_from_dedoose(doc_path, excerpts_txt_path):
    theme_code_table_path = doc_path.replace('.docx', '_codes.csv')

    if not os.path.isfile(theme_code_table_path):
        print(f'extracting codes from {excerpts_txt_path}...')
        all_codes = []

        with open(excerpts_txt_path, errors='ignore', encoding='utf-8') as f:
            for line in f.readlines():
                codes_search = re.search(r'Codes Applied:\s(.*)\(\d+ of \d+-\d+\)', line)
                if codes_search:
                    codes = codes_search.group(0)
                    codes = re.sub(r'^Codes Applied:\s+', '', codes)
                    codes = re.sub(r'\t', ' ', codes)
                    codes = re.split(r'\(\d+ of \d+-\d+\)\s+', codes)
                    for code in codes:
                        if len(code) > 0:
                            code = re.sub(r'\([^)]+\)$', '', code)
                            all_codes.append(code)
            f.close()
        
        codes_df = pd.DataFrame({'Theme 1 (replace this)': sorted(list(set(all_codes))), 'Theme 2 (replace this)': np.nan, 'Theme 3 (replace this)': np.nan, '...': np.nan})
        codes_df.to_csv(theme_code_table_path, index=False, encoding='utf-8-sig')

        print(f'{doc_path.replace(".docx", "_codes.csv")} created in {Path(doc_path).parent.absolute()}')
    else:
        print(f'code table already exists in {theme_code_table_path}')

    return theme_code_table_path


def import_codes(sentence2vec_model, doc_path, excerpts_txt_path, theme_code_table_path, regexp):
    print(f'extracting sentences...')
    start = datetime.now()
    cat_df = pd.read_csv(theme_code_table_path, encoding='utf-8-sig').applymap(lambda x: x.lower() if type(x) == str else x)
    cat_df.columns = cat_df.columns.str.lower()
    themes_found = []
    missing_codes = []

    train_columns = ['file_name', 'comment_id', 'original_sentence',
            'cleaned_sentence', 'sentence_embedding', 'codes', 'themes']
    themes_list = list(cat_df)
    train_columns.extend(themes_list)

    train_df = pd.DataFrame(columns=train_columns)

    with open(excerpts_txt_path, errors='ignore', encoding='utf-8') as f:
        lines = f.readlines()
        lines_length = len(lines)
        current_codes = []
        current_themes = []
        excerpt_found = False
        current_excerpt = ''

        for i, line in enumerate(lines):
            # looking for code names
            if not current_codes:
                codes_search = re.search(r'Codes Applied:\s(.*)\(\d+ of \d+-\d+\)', line)
                if codes_search:
                    codes = codes_search.group(0)
                    codes = re.sub(r'^Codes Applied:\s+', '', codes)
                    codes = re.sub(r'\t', ' ', codes)
                    codes = re.split(r'\(\d+ of \d+-\d+\)\s+', codes)
                    for code in codes:
                        if len(code) > 0:
                            code = re.sub(r'\([^)]+\)$', '', code)
                            code = code.lower()
                            current_codes.append(code)
                            find_code = (cat_df.values == code).any(axis=0)
                            try:
                                theme = cat_df.columns[np.where(find_code==True)[0]].item()
                                if theme not in current_themes:
                                    current_themes.append(theme)
                                if theme not in themes_found:
                                    themes_found.append(theme)
                            except ValueError:
                                missing_codes.append(code)
            # looking for excerpt start
            elif not excerpt_found:
                if re.search(r'^Excerpt Range: \d+-\d+', line):
                    excerpt_found = True
            # looking for excerpt text
            else:
                # continue reading if not at the end of the excerpt or file
                if i < lines_length-1 and (not re.search(r'^Title: ', line) or not re.search(r'^Doc Creator: ', lines[i+1])):
                    current_excerpt += line
                else:
                    current_excerpt = current_excerpt.replace('"', "'")
                    current_excerpt = current_excerpt.replace('’', "'")
                    current_excerpt = current_excerpt.replace("´", "'")
                    current_excerpt = current_excerpt.replace("…", "...")
                    current_excerpt = current_excerpt.replace("\\", "\\\\")

                    for sentence in sent_tokenize(current_excerpt):
                        cleaned_sentence = remove_stop_words(clean_sentence(sentence, regexp))
                        row = {
                            'file_name': re.search(r'([^\/]+).$', doc_path).group(0),
                            'comment_id': '0',
                            'original_sentence': sentence,
                            'cleaned_sentence': cleaned_sentence,
                            'sentence_embedding': sentence2vec_model.get_vector(cleaned_sentence),
                            'codes': '; '.join(current_codes),
                            'themes': '; '.join(current_themes)                            
                        }

                        themes_binary = {}

                        for theme in themes_list:
                            if theme in current_themes:
                                themes_binary[theme] = 1
                            else:
                                themes_binary[theme] = 0

                        row.update(themes_binary)

                        train_df = train_df.append(row, ignore_index=True)

                    current_codes = []
                    current_themes = []
                    excerpt_found = False
                    current_excerpt = ''  

        print(f'done extracting in {datetime.now() - start}')
        print(f'{len(set(missing_codes))} missing codes ({len(missing_codes)} sentences) in themes table')

        train_df.to_csv(doc_path.replace('.docx', '_train.csv'), index=False, encoding='utf-8-sig')

        # print(f'themes found = {themes_found}')

        f.close()

        return themes_found