import csv
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


train_file_path = 'text/reorder_exit_train.csv'
predict_file_path = 'text/reorder_exit_predict.csv'

coded_df = pd.read_csv(train_file_path, encoding='Windows-1252')
cat_df = pd.read_csv('text/reorder_categories.csv')


def generate_training_and_testing_data(many_together):
    # convert embedding string to np array
    if not many_together:
        coded_df['sentence embedding'] = coded_df['sentence embedding'].apply(
            lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(']','')
                .replace('  ',' '), sep=' '))

    # split into training and testing
    by_themes = coded_df.groupby('themes')

    training_list = []
    testing_list = []
    # we now iterate by codes
    for name, group in by_themes:
        training = group.sample(frac=.8)
        testing = group.loc[~group.index.isin(training.index)]
        training_list.append(training)
        testing_list.append(testing)
    # create two new dataframes from the lists
    train_df = pd.concat(training_list)
    test_df = pd.concat(testing_list)

    # create matrices from embedding array columns
    train_embedding_matrix = np.array(train_df['sentence embedding'].tolist())
    test_embedding_matrix = np.array(test_df['sentence embedding'].tolist())
    # create matrices from theme binary columns
    train_themes_binary_matrix = train_df.iloc[:, 7:].to_numpy()
    test_themes_binary_matrix = test_df.iloc[:, 7:].to_numpy()

    return (train_embedding_matrix, test_embedding_matrix,
        train_themes_binary_matrix, test_themes_binary_matrix)


def add_classification_to_csv(prediction_array, predicted_proba):
    predict_df = pd.read_csv(predict_file_path, encoding='Windows-1252')

    themes_list = cat_df.category.unique()
    new_df = pd.DataFrame(data=prediction_array, columns=themes_list)

    predict_df = predict_df.merge(new_df, left_index=True, right_index=True)
    predict_df.to_csv(predict_file_path, index=False)


def classify(sentence_embedding_matrix, clf, many_together):
    (train_embedding_matrix, test_embedding_matrix, train_themes_binary_matrix,
        test_themes_binary_matrix) = generate_training_and_testing_data(
            many_together)

    # scale data to [0-1] to avoid negative data passed to MultinomialNB
    if isinstance(clf, MultinomialNB):
        scaler = MinMaxScaler()
        train_embedding_matrix = scaler.fit_transform(train_embedding_matrix)
        test_embedding_matrix = scaler.fit_transform(test_embedding_matrix)

    clf = OneVsRestClassifier(clf)
    clf.fit(train_embedding_matrix, train_themes_binary_matrix)

    test_score = clf.score(test_embedding_matrix, test_themes_binary_matrix)

    prediction_array = clf.predict(sentence_embedding_matrix)

    predicted_proba = clf.predict_proba(sentence_embedding_matrix)

    if not many_together:
        add_classification_to_csv(prediction_array, predicted_proba)

    return test_score


# move to app.py?
import docx
import pandas as pd
from nltk import sent_tokenize
from lib.sentence2vec import Sentence2Vec
from preprocess import (
    clean_sentence,
    remove_interview_format,
    remove_interviewer,
    remove_stop_words)


model = Sentence2Vec()

def get_text(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

docx_file_path = 'text/reorder_exit.docx'
predict_file_path = 'text/reorder_exit_predict.csv'

text = get_text(docx_file_path)
text = remove_interviewer(text)

writer = csv.writer(open(predict_file_path, 'w', newline=''))
writer.writerow(['file name', 'original sentence', 'cleaned_sentence',
    'sentence embedding'])

coded_original_sentences = coded_df['original sentence'].tolist()

all_original_sentences = sent_tokenize(text)

uncoded_original_sentences = [sentence for sentence in all_original_sentences
    if sentence not in coded_original_sentences]

sentence_embedding_list = []

for sentence in uncoded_original_sentences:
    cleaned_sentence = clean_sentence(remove_stop_words(
        remove_interview_format(sentence)))
    sentence_embedding = model.get_vector(cleaned_sentence)

    writer.writerow([docx_file_path, sentence, cleaned_sentence,
        sentence_embedding])

    sentence_embedding_list.append(sentence_embedding)

sentence_embedding_matrix = np.stack(sentence_embedding_list, axis=0)

# Classifiers:
clf = KNeighborsClassifier(n_neighbors=5)
# clf = MultinomialNB()
# clf = GaussianNB()
# clf = tree.DecisionTreeClassifier()
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = MLPClassifier(alpha=1, max_iter=1000)
# clf = AdaBoostClassifier()


# classify(sentence_embedding_matrix, clf, False)

coded_df['sentence embedding'] = coded_df['sentence embedding'].apply(
    lambda x: np.fromstring(
        x.replace('\n','')
        .replace('[','')
        .replace(']','')
        .replace('  ',' '), sep=' '))

print('---------------kNN(k=1)----------------')
scores = []
for i in range(20):
    clf = KNeighborsClassifier(n_neighbors=1)
    scores.append(classify(sentence_embedding_matrix, clf, True))
print(f'{sum(scores)/len(scores)}')
print('---------------kNN(k=5)----------------')
scores = []
for i in range(20):
    clf = KNeighborsClassifier(n_neighbors=5)
    scores.append(classify(sentence_embedding_matrix, clf, True))
print(f'{sum(scores)/len(scores)}')
print('---------------MultinomialNB----------------')
scores = []
for i in range(20):
    clf = MultinomialNB()
    scores.append(classify(sentence_embedding_matrix, clf, True))
print(f'{sum(scores)/len(scores)}')
print('---------------GaussianNB----------------')
scores = []
for i in range(20):
    clf = GaussianNB()
    scores.append(classify(sentence_embedding_matrix, clf, True))
print(f'{sum(scores)/len(scores)}')
print('---------------DecisionTree----------------')
scores = []
for i in range(20):
    clf = tree.DecisionTreeClassifier()
    scores.append(classify(sentence_embedding_matrix, clf, True))
print(f'{sum(scores)/len(scores)}')
print('---------------RandomForest----------------')
scores = []
for i in range(20):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    scores.append(classify(sentence_embedding_matrix, clf, True))
print(f'{sum(scores)/len(scores)}')
print('---------------MLP----------------')
scores = []
for i in range(20):
    clf = MLPClassifier(alpha=1, max_iter=1000)
    scores.append(classify(sentence_embedding_matrix, clf, True))
print(f'{sum(scores)/len(scores)}')
print('---------------AdaBoost----------------')
scores = []
for i in range(20):
    clf = AdaBoostClassifier()
    scores.append(classify(sentence_embedding_matrix, clf, True))
print(f'{sum(scores)/len(scores)}')
