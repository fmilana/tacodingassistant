import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM, Input
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold


coded_df = pd.read_csv('text/reorder_exit_train.csv', encoding='Windows-1252')


def get_embedding_matrix(word_index):
    glove_file = open('text/glove.twitter.27B.100d.txt', encoding='utf-8')

    embedding_dict = {}

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embedding_dict[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, index in word_index.items():
        try:
            embedding_vector = embedding_dict[word]
            embedding_matrix[index] = embedding_vector
        except KeyError:
            # word not in glove_file
            continue

    return embedding_matrix


def generate_data():
    filtered_df = coded_df[coded_df['cleaned sentence'] != '']
    filtered_df.dropna(inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    sentences = filtered_df['cleaned sentence'].to_list()

    max_sentence_length = len(max(sentences))

    themes = coded_df.iloc[:, 7:].values

    # tokenizer = Tokenizer()
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)

    vocab_size = len(tokenizer.word_index) + 1

    data = pad_sequences(sequences, padding='post', maxlen=max_sentence_length)

    by_themes_df = filtered_df.groupby('themes')

    train_index_list = []
    val_index_list = []
    test_index_list = []

    for _, group in by_themes_df:
        train = group.sample(frac=.7)
        val = group.loc[~group.index.isin(train.index)].sample(frac=.5)
        test = group.loc[~group.index.isin(train.index) &
            ~group.index.isin(val.index)]

        train_index_list += [index for index in train.index.values.tolist()]
        val_index_list += [index for index in val.index.values.tolist()]
        test_index_list += [index for index in test.index.values.tolist()]

    x_train = data[train_index_list]
    y_train = themes[train_index_list]
    x_val = data[val_index_list]
    y_val = themes[val_index_list]
    x_test = data[test_index_list]
    y_test = themes[test_index_list]

    embedding_matrix = get_embedding_matrix(tokenizer.word_index)

    return (x_train, y_train, x_val, y_val, x_test, y_test,
        vocab_size, max_sentence_length, embedding_matrix)


def get_model(vocab_size, max_sentence_length, embedding_matrix):
    model = Sequential()
    model.add(Input(shape=(max_sentence_length,)))
    model.add(Embedding(vocab_size, 100, weights=[embedding_matrix],
        trainable=False))
    model.add(LSTM(8))
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def plot_graphs(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


print('generating data and loading word embeddings...')
(x_train, y_train, x_val, y_val, x_test, y_test,
    vocab_size, max_sentence_length, embedding_matrix) = generate_data()
print('done!')

print(f'x_train.shape = {x_train.shape}')
print(f'y_train.shape = {y_train.shape}')
print(f'x_val.shape = {x_val.shape}')
print(f'y_val.shape = {y_val.shape}')
print(f'x_test.shape = {x_test.shape}')
print(f'y_test.shape = {y_test.shape}')

model = get_model(vocab_size, max_sentence_length, embedding_matrix)
print(model.summary())

callbacks = [
    # EarlyStopping(monitor='acc', patience=20)
    ModelCheckpoint(filepath='models/my_model.h5', monitor='val_loss',
        save_best_only=True)
]

history = model.fit(x_train, y_train, epochs=100, batch_size=32,
    validation_data=(x_val, y_val), callbacks=callbacks)

plot_graphs(history)


# model = load_model('models/my_model.h5')
# score = model.evaluate(x_test, y_test, verbose=1)
# print(f'Test Score: {score[0]}')
# print(f'Test Accuracy: {score[1]}')
