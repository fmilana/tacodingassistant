import sys
import csv
import pandas as pd
from datetime import datetime
from functools import reduce
from nltk import word_tokenize
from collections import Counter


start = datetime.now()
print('analysing word frequencies...')

if len(sys.argv) == 2:
    train_file_path = sys.argv[1]
    predict_file_path = train_file_path.replace('train', 'predict')
    keywords_train_file_path = train_file_path.replace('1.csv', 'keywords_1.csv')
    keywords_predict_file_path = predict_file_path.replace('1.csv', 'keywords_1.csv')
    keywords_both_file_path = train_file_path.replace('train_1.csv', 'keywords_1.csv')
    analyse_predict_file_path = predict_file_path.replace('1.csv', 'analyse_1.csv')
    analyse_train_file_path = train_file_path.replace('1.csv', 'analyse_1.csv')
    analyse_both_file_path = train_file_path.replace('train_1.csv', 'analyse_1.csv')
else:
    train_file_path = 'text/reorder_exit_train.csv'
    predict_file_path = 'text/reorder_exit_predict.csv'
    keywords_train_file_path = train_file_path.replace('.csv', '_keywords.csv')
    keywords_predict_file_path = predict_file_path.replace('.csv', '_keywords.csv')
    keywords_both_file_path = train_file_path.replace('train.csv', '_keywords.csv')
    analyse_train_file_path = train_file_path.replace('.csv', '_analyse.csv')
    analyse_predict_file_path = predict_file_path.replace('.csv', '_analyse.csv')
    analyse_both_file_path = train_file_path.replace('train.csv', 'analyse.csv')


train_df = pd.read_csv(train_file_path, encoding='utf-8')
predict_df = pd.read_csv(predict_file_path, encoding='utf-8')

themes_list = [name for i, name in enumerate(predict_df.columns)
    if i >= 3 and i <= 8]

train_word_freq_dict = {theme: [] for theme in themes_list}
predict_word_freq_dict = {theme: [] for theme in themes_list}
both_word_freq_dict = {theme: [] for theme in themes_list}
train_keywords_dict = {}
predict_keywords_dict = {}

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
    'theres', 'ofyes', 'reasonsbecause', 'hadnt', 'youre', 'okay', 'if',
    'andyes', 'a']

minimum_proba = 0.95
train_theme_counts = []
predict_theme_counts = []
both_theme_counts = []

for theme in themes_list:
    train_theme_df = train_df.loc[train_df[theme] == 1].dropna() # <- drop moved
    predict_theme_df = predict_df.loc[predict_df[theme] == 1]    # predicted
    train_theme_counts.append(train_theme_df.shape[0])           # sentences
    predict_theme_counts.append(predict_theme_df.shape[0])       # moved in train

    for index, row in train_theme_df.iterrows():
        cleaned_sentence = row['cleaned_sentence']
        if isinstance(cleaned_sentence, str):
            words = set(word_tokenize(cleaned_sentence))
            for word in words:
                word = word.lower()
                if word not in more_stop_words:
                    train_word_freq_dict[theme].append(word)
                    both_word_freq_dict[theme].append(word)
                    if (word in train_keywords_dict and
                        index not in train_keywords_dict[word]):
                        train_keywords_dict[word].append(index)
                    elif word not in train_keywords_dict:
                        train_keywords_dict[word] = [index]

    for index, row in predict_theme_df.iterrows():
        if row[theme + ' probability'] > minimum_proba:
            cleaned_sentence = row['cleaned_sentence']
            if isinstance(cleaned_sentence, str):
                words = set(word_tokenize(cleaned_sentence))
                for word in words:
                    word = word.lower()
                    if word not in more_stop_words:
                        predict_word_freq_dict[theme].append(word)
                        both_word_freq_dict[theme].append(word)
                        if (word in predict_keywords_dict and
                            index not in predict_keywords_dict[word]):
                            predict_keywords_dict[word].append(index)
                        elif word not in predict_keywords_dict:
                            predict_keywords_dict[word] = [index]

both_theme_counts = [x + y for x, y in zip(predict_theme_counts,
    train_theme_counts)]

freq_dict_list = [
    train_word_freq_dict,
    predict_word_freq_dict,
    both_word_freq_dict
]

keywords_dict_list = [
    train_keywords_dict,
    predict_keywords_dict
]

freq_path_list = [
    analyse_train_file_path,
    analyse_predict_file_path,
    analyse_both_file_path
]

keywords_path_list = [
    keywords_train_file_path,
    keywords_predict_file_path
]

freq_counts_list = [
    train_theme_counts,
    predict_theme_counts,
    both_theme_counts
]

# create analyse csv's
for i, dict in enumerate(freq_dict_list):
    for theme in dict:
        counter = Counter(dict[theme])
        dict[theme] = counter.most_common()

    with open(freq_path_list[i], 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(themes_list)
        writer.writerow(freq_counts_list[i])

        biggest_list_length = 0
        for theme in dict:
            if len(dict[theme]) > biggest_list_length:
                biggest_list_length = len(dict[theme])

        for i in range(biggest_list_length):
            row = []
            for theme in dict:
                try:
                    row.append(f'{dict[theme][i][0]} ({dict[theme][i][1]})')
                except IndexError:
                    row.append('')
            writer.writerow(row)

# create keywords csv's
for i, dict in enumerate(keywords_dict_list):
    keywords_df = pd.DataFrame(dict.items(), columns=['word', 'sentences'])
    keywords_df.to_csv(keywords_path_list[i], index=False)


# cm analysis
for theme in themes_list:
    theme = theme.replace(' ', '_')
    cm_df = pd.read_csv(f'text/cm/reorder_exit_{theme}_cm.csv',
        encoding='utf-8')

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
