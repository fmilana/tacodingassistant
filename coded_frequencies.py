import csv
import pandas as pd
from nltk import word_tokenize
from collections import Counter


cat_df = pd.read_csv('text/reorder_categories.csv', encoding='Windows-1252')
train_df = pd.read_csv('text/reorder_exit_train.csv', encoding='Windows-1252')

themes_list = pd.unique(cat_df['category']).tolist()

theme_freq_dict = {}

for theme in themes_list:
    theme_df = train_df[train_df['themes'].str.contains(theme)]
    cleaned_sentences = theme_df['cleaned sentence'].tolist()

    theme_word_list = []

    for sentence in cleaned_sentences:
        if isinstance(sentence, str):
            words = word_tokenize(sentence)
            stop_words = open('text/analysis_stopwords.txt', 'r').read().split(',')

            for word in words:
                if word not in stop_words:
                    theme_word_list.append(word)

    theme_freq_dict[theme] = theme_word_list

for theme in theme_freq_dict:
    theme_freq_dict[theme] = Counter(theme_freq_dict[theme])

longest_entry_length = 0

for theme in theme_freq_dict:
    entry_length = len(theme_freq_dict[theme])

    if entry_length > longest_entry_length:
        longest_entry_length = entry_length

print(f'longest_entry_length = {longest_entry_length}')

with open('text/reorder_exit_frequencies.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(themes_list)

    for i in range(longest_entry_length):
        row = []

        for theme in themes_list:
            word_freq_list = theme_freq_dict[theme].most_common()

            try:
                row.append(f'{word_freq_list[i][0]} ({word_freq_list[i][1]})')
            except IndexError:
                row.append('')

        writer.writerow(row)

    file.close()
