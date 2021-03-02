import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


train_file_path = 'text/joint_groupbuy_jhim_train.csv'
predict_file_path = 'text/joint_groupbuy_jhim_predict.csv'

coded_df = pd.read_csv(train_file_path, encoding='Windows-1252')


def generate_training_and_testing_data():
    # convert embedding string to np array
    coded_df['sentence embedding'] = coded_df['sentence embedding'].apply(
        lambda x: np.fromstring(
            x.replace('\n','')
            .replace('[','')
            .replace(']','')
            .replace('  ',' '), sep=' '))


    # split into training and testing
    by_codes = coded_df.groupby('codes')

    training_list = []
    testing_list = []
    # we now iterate by codes
    for name, group in by_codes:
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
    # create arrays from codes column
    codes_array = coded_df['codes'].values
    # fit label encoder on all codes
    le = LabelEncoder()
    le.fit(codes_array)
    # create arrays from training and testing codes
    train_codes_array = train_df['codes'].values
    test_codes_array = test_df['codes'].values
    # encode them
    train_codes_encoded = le.transform(train_codes_array)
    test_codes_encoded = le.transform(test_codes_array)

    return (train_embedding_matrix, test_embedding_matrix,
        train_codes_encoded, test_codes_encoded, le)


def add_classification_to_csv(predicted_codes, predicted_proba):
    predict_df = pd.read_csv(predict_file_path, encoding='Windows-1252')
    new_columns = pd.DataFrame({'predicted code': predicted_codes.tolist(),
        'probability': list(map(np.amax, predicted_proba.tolist()))})
    predict_df = predict_df.merge(new_columns, left_index=True,
        right_index=True)
    predict_df.to_csv(predict_file_path, index=False)


def classify(sentence_embedding_matrix, clf):
    (train_embedding_matrix, test_embedding_matrix, train_codes_encoded,
        test_codes_encoded, le) = generate_training_and_testing_data()

    # scale data to [0-1] to avoid negative data passed to MultinomialNB
    if isinstance(clf, MultinomialNB):
        scaler = MinMaxScaler()
        train_embedding_matrix = scaler.fit_transform(train_embedding_matrix)
        test_embedding_matrix = scaler.fit_transform(test_embedding_matrix)

    clf.fit(train_embedding_matrix, train_codes_encoded)

    test_score = clf.score(test_embedding_matrix, test_codes_encoded)
    print(f'test score >>>>>>>>>> {test_score}')

    prediction_array = clf.predict(sentence_embedding_matrix)

    predicted_codes = le.inverse_transform(prediction_array)

    predicted_proba = clf.predict_proba(sentence_embedding_matrix)

    add_classification_to_csv(predicted_codes, predicted_proba)


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

docx_file_path = 'text/joint_groupbuy_jhim.docx'
predict_file_path = 'text/joint_groupbuy_jhim_predict.csv'

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
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = MultinomialNB()
# clf = GaussianNB()
clf = tree.DecisionTreeClassifier()
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = MLPClassifier(alpha=1, max_iter=1000)
# clf = AdaBoostClassifier()

classify(sentence_embedding_matrix, clf)
