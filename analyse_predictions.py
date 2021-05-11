import csv
import pandas as pd
from nltk import word_tokenize
from collections import Counter


predict_file_path = 'text/reorder_exit_predict.csv'
analyse_file_path = 'text/reorder_exit_analyse.csv'

predict_df = pd.read_csv(predict_file_path, encoding='Windows-1252')

themes_list = [name for i, name in enumerate(predict_df.columns)
    if i >= 4 and i <= 9]

word_freq_dict = {theme: [] for theme in themes_list}

for theme in themes_list:
    predict_df[theme] = predict_df[theme].astype(int)

more_stop_words = ['like', 'yes', 'actually', 'something', 'going', 'could',
    'would', 'oh', 'things', 'think', 'know', 'really', 'well', 'kind',
    'always', 'mean', 'maybe', 'get', 'guess', 'bit', 'much', 'go', 'one',
    'thing', 'probably', 'iv', 'i', 'so', 'dont', 'but', 'and', 'how']

minimum_proba = 0.95

for theme in themes_list:
    for index, row in predict_df.loc[predict_df[theme] == 1].iterrows():
        if row[theme + ' probability'] > minimum_proba:
            cleaned_sentence = row['cleaned_sentence']
            if isinstance(cleaned_sentence, str):
                for word in word_tokenize(cleaned_sentence):
                    word = word.lower()
                    if word not in more_stop_words:
                        word_freq_dict[theme].append(word)

for theme in word_freq_dict:
    counter = Counter(word_freq_dict[theme])
    word_freq_dict[theme] = counter.most_common()

with open(analyse_file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(themes_list)

    biggest_list_length = 0
    for theme in word_freq_dict:
        if len(word_freq_dict[theme]) > biggest_list_length:
            biggest_list_length = len(word_freq_dict[theme])

    for i in range(biggest_list_length):
        row = []
        for theme in word_freq_dict:
            try:
                row.append(f'{word_freq_dict[theme][i][0]} ' +
                    f'({word_freq_dict[theme][i][1]})')
            except IndexError:
                row.append('')
        writer.writerow(row)
