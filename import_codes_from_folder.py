import os
import re
import zipfile
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from preprocess import clean_sentence, remove_stop_words


def import_themes(doc_path, codes_path):
    if os.path.exists(codes_path):
        cat_dict = {}

        for entry in os.scandir(codes_path):
            if entry.path.endswith('.docx'):
                code = entry.name[:-5]

                with zipfile.ZipFile(entry.path, 'r') as archive:
                    doc_xml = archive.read('word/document.xml')
                    doc_soup = BeautifulSoup(doc_xml, 'lxml')

                    first_line = doc_soup.find('w:p').find('w:t').get_text()
                    theme = re.match(r'Name: (.+?)\\', first_line).group(0)

                    if theme not in cat_dict:
                        cat_dict[theme] = [code]
                    else:
                        cat_dict[theme].append(code)
        
        cat_df = pd.DataFrame.from_dict(cat_dict)
        cat_df.to_csv(doc_path.replace('.docx', '_codes.csv'), index=False)    


def import_codes(model, doc_path, codes_path, theme_code_table_path, regexp):
    if theme_code_table_path == '':
        import_themes(doc_path, codes_path)
        theme_code_table_path = doc_path.replace('.docx', '_codes.csv')

    cat_df = pd.read_csv(theme_code_table_path, encoding='utf-8-sig')

    header = [
        'file_name', 
        'original_sentence', 
        'cleaned_sentence', 
        'sentence_embedding', 
        'codes', 
        'themes'
        ]
    themes_list = list(cat_df)
    header.extend(themes_list)

    train_df = pd.DataFrame(columns=header)

    for entry in os.scandir(codes_path):
        if entry.path.endswith('.docx'):
            code = entry.name[:-5]
            find_code = (cat_df.values == code).any(axis=0)
            theme = cat_df.columns[np.where(find_code==True)[0]].item()

            with zipfile.ZipFile(entry.path, 'r') as archive:
                doc_xml = archive.read('word/document.xml')
                doc_soup = BeautifulSoup(doc_xml, 'lxml')

                for paragraph in doc_soup.find_all('w:p'):
                    if (paragraph.find('w:shd') is None 
                    and paragraph.find('w:t')):
                        node = paragraph.find('w:t').get_text() 
                        for sentence in sent_tokenize(node):
                            cleaned_sentence = remove_stop_words(clean_sentence(sentence, regexp))
                        
                            if train_df['original_sentence'].eq(sentence).any():
                                matching_index = train_df.index[
                                    train_df['original_sentence'] == sentence].tolist()[0]

                                matching_row = train_df.iloc[matching_index]
                                
                                matching_row_codes = matching_row['codes'].split('; ')
                                matching_row_themes = matching_row['themes'].split('; ')

                                if code not in matching_row_codes:
                                    matching_row_codes.append(code)
                                
                                if theme not in matching_row_themes:
                                    matching_row_themes.append(theme)

                                new_codes = '; '.join(sorted(matching_row_codes, key=str.casefold))
                                new_themes = '; '.join(sorted(matching_row_themes, key=str.casefold))
                                
                                train_df.loc[matching_index, 'codes'] = new_codes                                    
                                train_df.loc[matching_index, 'themes'] = new_themes
                                train_df.loc[matching_index, theme] = 1
                                
                            else:
                                row = {
                                    'file_name': entry.path.replace('\\', '/'),
                                    'original_sentence': sentence,
                                    'cleaned_sentence': cleaned_sentence,
                                    'sentence_embedding': model.get_vector(cleaned_sentence),
                                    'codes': code,
                                    'themes': theme
                                }

                                theme_row = {}

                                for theme_name in themes_list:
                                    if theme_name == theme:
                                        theme_row[theme_name] = 1
                                    else:
                                        theme_row[theme_name] = 0

                                row.update(theme_row)

                                train_df = train_df.append(row, ignore_index=True)

    train_df.to_csv(doc_path.replace('.docx', '_train.csv'), index=False)