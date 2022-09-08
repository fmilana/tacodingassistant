import os
import zipfile
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from preprocess import clean_sentence, remove_stop_words
from datetime import datetime


def create_codes_csv_from_nvivo(doc_path, codes_folder_path):
    theme_code_table_path = os.path.join(codes_folder_path, doc_path.replace('.docx', '_codes.csv'))

    if not os.path.isfile(theme_code_table_path):
        print(f'extracting codes from {codes_folder_path}...')
        codes = []

        for entry in os.scandir(codes_folder_path):
            if entry.path.endswith('.docx'):
                code = entry.name[:-5].lower()
                codes.append(code)

        codes_df = pd.DataFrame({'Theme 1 (replace this)': sorted(list(set(codes))), 'Theme 2 (replace this)': np.nan, 'Theme 3 (replace this)': np.nan, '...': np.nan})
        codes_df.to_csv(theme_code_table_path, index=False, encoding='utf-8-sig', errors='replace')

        print(f'{doc_path.replace(".docx", "_codes.csv")} created in {codes_folder_path}')
    else:
        print(f'code table already exists in {theme_code_table_path}')

    return theme_code_table_path


# doc_path and theme_code_table_path documents already copied in data folder
def import_codes(sentence2vec_model, doc_path, codes_folder_path, theme_code_table_path, regexp):
    print(f'extracting codes from {codes_folder_path}...')
    start = datetime.now()

    print(f'theme_code_table_path = {theme_code_table_path}')
    print(f'len(theme_code_table_path) = {len(theme_code_table_path)}')

    # if theme_code_table_path == '':
    #     # create codes.csv in data folder from codes documents
    #     import_themes(doc_path, codes_folder_path)
    #     theme_code_table_path = doc_path.replace('.docx', '_codes.csv')

    cat_df = pd.read_csv(theme_code_table_path, encoding='utf-8-sig', encoding_errors='replace').applymap(lambda x: x.lower() if type(x) == str else x)
    cat_df.columns = cat_df.columns.str.lower()

    header = [
        'file_name', 
        'comment_id',
        'original_sentence', 
        'cleaned_sentence', 
        'sentence_embedding', 
        'codes', 
        'themes'
        ]
    themes_list = [theme_name.lower() for theme_name in list(cat_df)]
    header.extend(themes_list)

    train_df = pd.DataFrame(columns=header)

    themes_found = []

    missing_codes = []

    for entry in os.scandir(codes_folder_path):
        if entry.path.endswith('.docx'):
            code = entry.name[:-5].lower()
            find_code = (cat_df.values == code).any(axis=0)
            theme = None

            try:
                theme = cat_df.columns[np.where(find_code==True)[0]].item().lower()
                if theme not in themes_found:
                    themes_found.append(theme)

                with zipfile.ZipFile(entry.path, 'r') as archive:
                    doc_xml = archive.read('word/document.xml')
                    doc_soup = BeautifulSoup(doc_xml, 'lxml')

                    for paragraph in doc_soup.find_all('w:p'):
                        if (paragraph.find('w:shd') is None and paragraph.find('w:highlight') is None and paragraph.find('w:t')):
                            text = paragraph.find('w:t').get_text() 
                            text = text.replace('"', "'")
                            text = text.replace('’', "'")
                            text = text.replace("´", "'")
                            text = text.replace("…", "...")
                            text = text.replace("\\", "\\\\")
                            for sentence in sent_tokenize(text):
                                cleaned_sentence = remove_stop_words(clean_sentence(sentence, regexp))
                            
                                if train_df['original_sentence'].eq(sentence).any():
                                    matching_index = train_df.index[
                                        train_df['original_sentence'] == sentence].tolist()[0]

                                    matching_row = train_df.iloc[matching_index]
                                    
                                    matching_row_codes = [matching_code.lower() for matching_code in matching_row['codes'].split('; ')]
                                    matching_row_themes = [matching_theme.lower() for matching_theme in matching_row['themes'].split('; ')]

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
                                        'comment_id': '0', 
                                        'original_sentence': sentence,
                                        'cleaned_sentence': cleaned_sentence,
                                        'sentence_embedding': sentence2vec_model.get_vector(cleaned_sentence),
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
            except ValueError:
                missing_codes.append(code)            

    print(f'done extracting in {datetime.now() - start}')

    print(f'{len(set(missing_codes))} missing codes ({len(missing_codes)} sentences) in themes table (some counters in the codes table will be 0)')
    # print(set(missing_codes))

    train_df.to_csv(doc_path.replace('.docx', '_train.csv'), index=False, encoding='utf-8-sig', errors='replace')


    return themes_found

