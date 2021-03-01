import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


train_file_path = 'text/joint_groupbuy_jhim_train.csv'
predict_file_path = 'text/joint_groupbuy_jhim_predict.csv'

train_df = pd.read_csv(train_file_path, encoding='Windows-1252')


def generate_training_data():
    # convert embedding string to np array
    train_df['sentence embedding'] = train_df['sentence embedding'].apply(
        lambda x: np.fromstring(
            x.replace('\n','')
            .replace('[','')
            .replace(']','')
            .replace('  ',' '), sep=' '))
    # create matrix from embedding array column
    embedding_matrix = np.array(train_df['sentence embedding'].tolist())
    # create array from codes column
    codes_array = train_df['codes'].to_numpy()

    print(f'training_embedding_matrix shape = {embedding_matrix.shape}')
    print(f'codes_array shape = {codes_array.shape}')

    le = preprocessing.LabelEncoder()
    codes_encoded = le.fit_transform(codes_array)

    return embedding_matrix, codes_encoded, le


def add_classification_to_csv(predicted_codes, predicted_proba):
    # print(predicted_codes.tolist())

    predict_df = pd.read_csv(predict_file_path, encoding='Windows-1252')
    new_columns = pd.DataFrame({'predicted code': predicted_codes.tolist()})
        # 'probability': predicted_proba.tolist()})
    predict_df = predict_df.merge(new_columns, left_index=True,
        right_index=True)
    predict_df.to_csv(predict_file_path, index=False)


def knn_classify(sentence_embedding_matrix):
    training_embedding_matrix, codes_encoded, le = generate_training_data()

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(training_embedding_matrix, codes_encoded)
    prediction_array = clf.predict(sentence_embedding_matrix)
    predicted_codes = le.inverse_transform(prediction_array)

    predicted_proba = clf.predict_proba(sentence_embedding_matrix)
    print(f'predicted_codes.shape: {predicted_codes.shape}')
    print(f'predicted_proba.shape: {predicted_proba.shape}')

    add_classification_to_csv(predicted_codes, predicted_proba)


# move to app.py?
import docx
import pandas as pd
from nltk import sent_tokenize
from lib.sentence2vec import Sentence2Vec
from preprocess import (
    clean_sentence,
    clean_text,
    remove_interviewer,
    remove_stop_words)


model = Sentence2Vec('word2vec-google-news-300')

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

coded_original_sentences = train_df['original sentence'].tolist()

all_original_sentences = sent_tokenize(text)

uncoded_original_sentences = [sentence for sentence in all_original_sentences
    if sentence not in coded_original_sentences]

sentence_embedding_list = []

for sentence in uncoded_original_sentences:
    cleaned_sentence = clean_sentence(remove_stop_words(clean_text(sentence)))
    sentence_embedding = model.get_vector(cleaned_sentence)

    writer.writerow([docx_file_path, sentence, cleaned_sentence,
        sentence_embedding])

    sentence_embedding_list.append(sentence_embedding)

sentence_embedding_matrix = np.stack(sentence_embedding_list, axis=0)


knn_classify(sentence_embedding_matrix)
