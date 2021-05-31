import csv
import pandas as pd
from datetime import datetime
from nltk import word_tokenize
from collections import Counter


start = datetime.now()
print('analysing word frequencies...')

train_file_path = 'text/reorder_exit_train.csv'
predict_file_path = 'text/reorder_exit_predict.csv'
analyse_predict_file_path = 'text/reorder_exit_predict_analyse.csv'
analyse_train_file_path = 'text/reorder_exit_train_analyse.csv'

train_df = pd.read_csv(train_file_path, encoding='Windows-1252')
predict_df = pd.read_csv(predict_file_path, encoding='Windows-1252')

themes_list = [name for i, name in enumerate(predict_df.columns)
    if i >= 4 and i <= 9]

predict_word_freq_dict = {theme: [] for theme in themes_list}
train_word_freq_dict = {theme: [] for theme in themes_list}

for theme in themes_list:
    train_df[theme] = train_df[theme].astype(int)
    predict_df[theme] = predict_df[theme].astype(int)

more_stop_words = ['like', 'yes', 'actually', 'something', 'going', 'could',
    'would', 'oh', 'ah', 'things', 'think', 'know', 'really', 'well', 'kind',
    'always', 'mean', 'maybe', 'get', 'guess', 'bit', 'much', 'go', 'one',
    'thing', 'probably', 'iv', 'i', 'so', 'dont', 'but', 'and', 'how', 'why',
    'wouldnt', 'wasnt', 'didnt', 'thats', 'thatll', 'im', 'you', 'no', 'isnt',
    'what', 'do', 'did', 'got', 'ill', 'id', 'or', 'do', 'is', 'ive', 'youd',
    'cant', 'wont', 'youve', 'dooesnt', 'is', 'it', 'its', 'the', 'thenokay',
    'theres', 'ofyes', 'reasonsbecause', 'hadnt']

minimum_proba = 0.95
train_theme_counts = []
predict_theme_counts = []

for theme in themes_list:
    train_theme_df = train_df.loc[train_df[theme] == 1]
    predict_theme_df = predict_df.loc[predict_df[theme] == 1]
    train_theme_counts.append(train_theme_df.shape[0])
    predict_theme_counts.append(predict_theme_df.shape[0])

    for index, row in train_theme_df.iterrows():
        cleaned_sentence = row['cleaned_sentence']
        if isinstance(cleaned_sentence, str):
            words = set(word_tokenize(cleaned_sentence))
            for word in words:
                word = word.lower()
                if word not in more_stop_words:
                    train_word_freq_dict[theme].append(word)

    for index, row in predict_theme_df.iterrows():
        if row[theme + ' probability'] > minimum_proba:
            cleaned_sentence = row['cleaned_sentence']
            if isinstance(cleaned_sentence, str):
                words = set(word_tokenize(cleaned_sentence))
                for word in words:
                    word = word.lower()
                    if word not in more_stop_words:
                        predict_word_freq_dict[theme].append(word)

for theme in train_word_freq_dict:
    counter = Counter(train_word_freq_dict[theme])
    train_word_freq_dict[theme] = counter.most_common()

for theme in predict_word_freq_dict:
    counter = Counter(predict_word_freq_dict[theme])
    predict_word_freq_dict[theme] = counter.most_common()

with open(analyse_train_file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(themes_list)
    writer.writerow(train_theme_counts)

    biggest_list_length = 0
    for theme in train_word_freq_dict:
        if len(train_word_freq_dict[theme]) > biggest_list_length:
            biggest_list_length = len(train_word_freq_dict[theme])

    for i in range(biggest_list_length):
        row = []
        for theme in train_word_freq_dict:
            try:
                row.append(f'{train_word_freq_dict[theme][i][0]} ' +
                    f'({train_word_freq_dict[theme][i][1]})')
            except IndexError:
                row.append('')
        writer.writerow(row)

with open(analyse_predict_file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(themes_list)
    writer.writerow(predict_theme_counts)

    biggest_list_length = 0
    for theme in predict_word_freq_dict:
        if len(predict_word_freq_dict[theme]) > biggest_list_length:
            biggest_list_length = len(predict_word_freq_dict[theme])

    for i in range(biggest_list_length):
        row = []
        for theme in predict_word_freq_dict:
            try:
                row.append(f'{predict_word_freq_dict[theme][i][0]} ' +
                    f'({predict_word_freq_dict[theme][i][1]})')
            except IndexError:
                row.append('')
        writer.writerow(row)

# cm analysis

for theme in themes_list:
    theme = theme.replace(' ', '_')
    cm_df = pd.read_csv(f'text/cm/reorder_exit_{theme}_cm.csv',
        encoding='Windows-1252')

    col_names = cm_df.columns.values

    cm_word_freq_dict = {col_name: [] for col_name in col_names}

    for index, row in cm_df.iterrows():
        for col_name in col_names:
            sentence = row[col_name]

            if isinstance(sentence, str) and len(sentence) > 0:
                cleaned_sentence = train_df[train_df['original_sentence'] == sentence]['cleaned_sentence'].any()
                if isinstance(cleaned_sentence, str):
                    words = set(word_tokenize(cleaned_sentence))
                    for word in words:
                        word = word.lower()
                        if word not in more_stop_words:
                            cm_word_freq_dict[col_name].append(word)

    for col_name in cm_word_freq_dict:
        counter = Counter(cm_word_freq_dict[col_name])
        cm_word_freq_dict[col_name] = counter.most_common()

    with open(f'text/cm/reorder_exit_{theme}_cm_analyse.csv',
        'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(col_names)

        biggest_list_length = 0
        for col_name in cm_word_freq_dict:
            if len(cm_word_freq_dict[col_name]) > biggest_list_length:
                biggest_list_length = len(cm_word_freq_dict[col_name])

        for i in range(biggest_list_length):
            row = []
            for col_name in cm_word_freq_dict:
                try:
                    row.append(f'{cm_word_freq_dict[col_name][i][0]} ' +
                        f'({cm_word_freq_dict[col_name][i][1]})')
                except IndexError:
                    row.append('')
            writer.writerow(row)

print(f'done analysing in {datetime.now() - start}')
