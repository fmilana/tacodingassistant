import os
import zipfile
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from lib.sentence2vec import Sentence2Vec
from nltk import sent_tokenize
from preprocess import clean_sentence, remove_stop_words


def import_codes(dir_path, regexp):
    model = Sentence2Vec()
    cat_df = pd.read_csv('text/reorder_exit_themes.csv', encoding='utf-8-sig')

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

    if os.path.exists(dir_path):
        for entry in os.scandir(dir_path):
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

        train_df.to_csv('text/reorder_exit_nvivo_train.csv', index=False)
    else:
        print('wrong folder path')