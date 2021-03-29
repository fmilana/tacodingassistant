import csv
import docx
import pickle
import keras
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.keras.backend as tfb
from collections import Counter
from nltk import sent_tokenize, word_tokenize
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import (
    Activation,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalMaxPool1D,
    GlobalMaxPooling1D,
    LSTM,
    MaxPooling1D,
    Input)
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.utils import class_weight
from preprocess import (
    clean_sentence,
    remove_interviewer,
    remove_interview_format,
    remove_stop_words)


docx_file_path = 'text/reorder_exit.docx'

coded_df = pd.read_csv('text/reorder_exit_train.csv',
    encoding='Windows-1252').sample(frac=1)
cat_df = pd.read_csv('text/reorder_categories.csv')


def get_embedding_matrix(word_index):
    glove_file = open('embeddings/glove.twitter.27B.100d.txt', encoding='utf-8')

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


def generate_data(pretrained):
    filtered_df = coded_df[coded_df['cleaned sentence'] != '']
    filtered_df.dropna(inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    coded_cleaned_sentences = filtered_df['cleaned sentence'].to_list()

    doc = docx.Document(docx_file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)

    text = '\n'.join(full_text)
    text = remove_interviewer(text)

    all_original_sentences = sent_tokenize(text)

    all_cleaned_sentences = []

    for sentence in all_original_sentences:
        cleaned_sentence = clean_sentence(remove_stop_words(
            remove_interview_format(sentence)))
        all_cleaned_sentences.append(cleaned_sentence)

    max_sentence_length = len(max(all_cleaned_sentences))

    themes = coded_df.iloc[:, 7:].values

    # tokenizer = Tokenizer(num_words=1000)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_cleaned_sentences)
    sequences = tokenizer.texts_to_sequences(coded_cleaned_sentences)

    # save tokenizer for future scoring
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    vocab_size = len(tokenizer.word_index) + 1
    print(f'vocab_size = {vocab_size}')

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

    test_cleaned_sentences = coded_df.iloc[test_index_list]['cleaned sentence'].tolist()

    x_train = data[train_index_list]
    y_train = themes[train_index_list]
    x_val = data[val_index_list]
    y_val = themes[val_index_list]
    x_test = data[test_index_list]
    y_test = themes[test_index_list]

    if pretrained:
        embedding_matrix = get_embedding_matrix(tokenizer.word_index)
    else:
        embedding_matrix = None

    return (x_train, y_train, x_val, y_val, x_test, y_test,
        vocab_size, max_sentence_length, embedding_matrix,
        test_cleaned_sentences)


POS_WEIGHT = 3  # multiplier for positive targets, needs to be tuned

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform output from proba to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # cast target to float32
    target = tf.cast(target, tf.float32)
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)


def get_model(version, vocab_size, max_sentence_length, embedding_matrix, y_train):
    if version == 'pretrained_lstm':
        model = Sequential()
        model.add(Input(shape=(max_sentence_length,)))
        model.add(Embedding(vocab_size, 100, weights=[embedding_matrix],
            mask_zero=True, trainable=False))
        model.add(LSTM(8))
        # model.add(Dropout(0.2))
        model.add(Dense(6, activation='sigmoid'))

    elif version == 'pretrained_conv1d':
        # 27.47% accuracy
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_sentence_length,
            weights=[embedding_matrix], mask_zero=True, trainable=False))
        model.add(Conv1D(32, 6, padding='valid', activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(32, 6, padding='valid', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.1))
        model.add(Dense(6, activation='sigmoid'))

    elif version == 'non_pretrained_conv1d':
        # 23.98% accuracy
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_sentence_length,
            mask_zero=True))
        model.add(Conv1D(32, 6, padding='valid', activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(32, 6, padding='valid', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.1))
        model.add(Dense(6, activation='sigmoid'))

    model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['acc'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

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



def get_keyword_labels(sentences_dict, themes_list):
    word_freq_dict = {}
    all_cms = []

    more_stop_words = ['like', 'yes', 'actually', 'something', 'going', 'could',
        'would', 'oh', 'things', 'think', 'know', 'really', 'well', 'kind',
        'always', 'mean', 'maybe', 'get', 'guess', 'bit', 'much', 'go', 'one',
        'thing', 'probably']

    for category in sentences_dict:
        sentence_list = sentences_dict[category]
        joined_sentences = ''
        joined_vocab = []

        for sentence in sentence_list:
            if isinstance(sentence, str):
                joined_sentences += (' ' + sentence)

                for word in set(word_tokenize(sentence)):
                    joined_vocab.append(word)

        if len(joined_sentences) > 0:
            counter_freq = Counter([word for word in
                word_tokenize(joined_sentences) if word not in more_stop_words])

            counter_vocab = Counter(joined_vocab)

            first_most_freq = counter_freq.most_common(3)[0]
            second_most_freq = counter_freq.most_common(3)[1]
            third_most_freq = counter_freq.most_common(3)[2]
            word_freq_dict[category] = (f'{first_most_freq[0]} ' +
                f'(f: {first_most_freq[1]}, s: {counter_vocab[first_most_freq[0]]})\n' +
                f'{second_most_freq[0]} (f: {second_most_freq[1]}, s: {counter_vocab[second_most_freq[0]]})\n' +
                f'{third_most_freq[0]} (f: {third_most_freq[1]}, s: {counter_vocab[third_most_freq[0]]})')
        else:
            word_freq_dict[category] = ''

    for theme in themes_list:
        true_negative_keyword = word_freq_dict[theme + ' true_negatives']
        false_positive_keyword = word_freq_dict[theme + ' false_positives']
        false_negative_keyword = word_freq_dict[theme + ' false_negatives']
        true_positive_keyword = word_freq_dict[theme + ' true_positives']
        # create 2x2 keyword confusion matrix array for each theme
        all_cms.append(np.array([[true_negative_keyword, false_positive_keyword],
                                [false_negative_keyword, true_positive_keyword]]))

    all_cms = np.dstack(all_cms)
    all_cms = np.transpose(all_cms, (2, 0, 1))

    return all_cms


def plot_multilabel_confusion_matrix(cm, labels, axes, theme, class_names, fontsize=14):
    annot = (np.asarray([f'{count}\n {keyword}'
        for keyword, count in zip(labels.flatten(), cm.flatten())])
        ).reshape(2, 2)

    heatmap = sns.heatmap(cm, annot=annot, fmt='', cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=axes)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
        ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
        ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(theme)



def plot_heatmaps(y_true, y_pred, sentences_dict, themes_list):
    all_cms = multilabel_confusion_matrix(y_true, y_pred)

    all_label_cms = get_keyword_labels(sentences_dict, themes_list)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for axes, labels, cm, theme in zip(ax.flatten(), all_label_cms, all_cms,
        themes_list):
        plot_multilabel_confusion_matrix(cm, labels, axes, theme, ['N', 'Y'])

    # fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    plt.show()



def plot_cm(model, x_test, y_test, test_cleaned_sentences):
    test_pred = np.rint(model.predict(x_test))

    themes_list = cat_df.category.unique()

    coded_df['sentence embedding'] = coded_df['sentence embedding'].apply(
        lambda x: np.fromstring(
            x.replace('\n','')
            .replace('[','')
            .replace(']','')
            .replace('  ',' '), sep=' '))

    sentences_dict = {}

    for col, class_name in enumerate(themes_list):
        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []

        for row in range(test_pred.shape[0]):
            if test_pred[row, col] == 1 and y_test[row, col] == 1:
                true_positives.append(test_cleaned_sentences[row])
            elif test_pred[row, col] == 0 and y_test[row, col] == 0:
                true_negatives.append(test_cleaned_sentences[row])
            elif test_pred[row, col] == 1 and y_test[row, col] == 0:
                false_positives.append(test_cleaned_sentences[row])
            elif test_pred[row, col] == 0 and y_test[row, col] == 1:
                false_negatives.append(test_cleaned_sentences[row])

        sentences_dict[class_name + ' true_positives'] = true_positives
        sentences_dict[class_name + ' true_negatives'] = true_negatives
        sentences_dict[class_name + ' false_positives'] = false_positives
        sentences_dict[class_name + ' false_negatives'] = false_negatives

    plot_heatmaps(y_test, test_pred, sentences_dict, themes_list)











print('generating data and loading word embeddings...')
(x_train, y_train, x_val, y_val, x_test, y_test,
    vocab_size, max_sentence_length, embedding_matrix,
    test_cleaned_sentences) = generate_data(pretrained=True)
print('done!')

print(f'x_train.shape = {x_train.shape}')
print(f'y_train.shape = {y_train.shape}')
print(f'x_val.shape = {x_val.shape}')
print(f'y_val.shape = {y_val.shape}')
print(f'x_test.shape = {x_test.shape}')
print(f'y_test.shape = {y_test.shape}')


# model = get_model('pretrained_lstm', vocab_size, max_sentence_length,
#     embedding_matrix, y_train)

model = get_model('pretrained_lstm', vocab_size, max_sentence_length,
    embedding_matrix, y_train)

# model = get_model('non_pretrained_conv1d', vocab_size, max_sentence_length,
#     None, y_train)

# print(model.summary())

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4),
    ModelCheckpoint(filepath='models/pretrained_lstm_model.h5',
        monitor='val_loss', save_best_only=True),
    # ModelCheckpoint(filepath='models/pretrained_conv1d_model.h5',
    #     monitor='val_loss', save_best_only=True),
    # ModelCheckpoint(filepath='models/non_pretrained_conv1d_model.h5',
    #     monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau()
]

############### WITH CLASS WEIGHTS ########################
# class_count_list = [sum(i) for i in zip(*y_train)]
# max_class_count = max(class_count_list)
# class_weights_list = [(max_class_count/i) * 10 for i in class_count_list]
#
# # num_samples = y_test.shape[0]
# # class_weights_list = [num_samples/count for count in class_count_list]
#
# class_weight = dict(enumerate(class_weights_list))
#
# print(f'CLASS WEIGHTS ===========> {class_weight}')
#
# history = model.fit(x_train, y_train, epochs=100, batch_size=16,
#     validation_data=(x_val, y_val), class_weight=class_weight,
#     callbacks=callbacks)
#
# plot_graphs(history)

################ WITH CUSTOM LOSS ########################
# history = model.fit(x_train, y_train, epochs=100, batch_size=16,
#     validation_data=(x_val, y_val), callbacks=callbacks)
#
# plot_graphs(history)


# model = load_model('models/pretrained_lstm_model.h5')
# model = load_model('models/pretrained_conv1d_model.h5')
# model = load_model('models/non_pretrained_conv1d_model.h5')
# score = model.evaluate(x_test, y_test, verbose=1)
# print(f'Test Score: {score[0]}')
# print(f'Test Accuracy: {score[1]}')







##### evaluate accuracy #####

# def evaluate_accuracy(model, iter):
#     accuracy_list = []
#
#     for i in range(iter):
#         _, _, _, _, x_test, y_test, _, _, _, _ = generate_data(pretrained=True)
#         accuracy_list.append(model.evaluate(x_test, y_test)[1])
#
#         print(f'{i + 1}/{iter} done.')
#
#     return sum(accuracy_list)/len(accuracy_list)
#
# model = load_model('models/pretrained_lstm_model.h5', compile=False)
# model = load_model('models/pretrained_conv1d_model.h5', compile=False)
# model = load_model('models/non_pretrained_conv1d_model.h5', compile=False)
# print(f'accuracy over 20 iterations = {evaluate_accuracy(model, 20)}')


# RUN *AFTER* classify_with_network.py (really?)
model = load_model('models/pretrained_lstm_model.h5', compile=False)
# model = load_model('models/pretrained_conv1d_model.h5', compile=False)
# model = load_model('models/non_pretrained_conv1d_model.h5', compile=False)
plot_cm(model, x_test, y_test, test_cleaned_sentences)
