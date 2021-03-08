import csv
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from skmultilearn.adapt import MLkNN, BRkNNaClassifier, MLARAM
from skmultilearn.problem_transform import (
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset
)
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


train_file_path = 'text/reorder_exit_train.csv'
predict_file_path = 'text/reorder_exit_predict.csv'
categories_file_path = 'text/reorder_categories.csv'

coded_df = pd.read_csv(train_file_path, encoding='Windows-1252')
cat_df = pd.read_csv(categories_file_path)


def generate_training_and_testing_data():
    # convert embedding string to np array
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

    train_themes_list = train_df['themes'].tolist()

    return (train_embedding_matrix, test_embedding_matrix,
        train_themes_binary_matrix, test_themes_binary_matrix, train_themes_list)


def add_classification_to_csv(clf, prediction_output):
    predict_df = pd.read_csv(predict_file_path, encoding='Windows-1252')

    themes_list = cat_df.category.unique()

    if isinstance(prediction_output, scipy.sparse.spmatrix):
        new_df = pd.DataFrame.sparse.from_spmatrix(data=prediction_output,
            columns=themes_list)
    else:
        new_df = pd.DataFrame(data=prediction_output, columns=themes_list)

    predict_df = predict_df.merge(new_df, left_index=True, right_index=True)
    predict_df.to_csv(predict_file_path, index=False)


def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80
    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)
    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades
    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)
    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T
    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]
    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)
    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)
    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half
    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)
    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier
    return ListedColormap(initial_cm)


def plot_training_data():
    train_embedding_matrix, _, _, _, train_themes_list = generate_training_and_testing_data()
    le = LabelEncoder()
    train_themes_encoded = le.fit_transform(train_themes_list)
    pca = PCA(2)
    projected = pca.fit_transform(train_embedding_matrix)
    plt.scatter(projected[:, 0], projected[:, 1],
        c=train_themes_encoded,
        edgecolors='none',
        cmap=generate_colormap(len(set(train_themes_list)))
    )
    plt.colorbar()
    plt.show()


def classify(sentence_embedding_matrix, clf):
    X_train, X_test, Y_train, Y_test, _ = generate_training_and_testing_data()

    # scale data to [0-1] to avoid negative data passed to MultinomialNB
    if isinstance(clf, MultinomialNB):
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    # elif isinstance(clf, BRkNNaClassifier):
    #     parameters = {'k': range(1, 5)}
    #     clf = GridSearchCV(clf, parameters, scoring='f1_macro')
    #     clf.fit(X_train, Y_train)
    #     print (clf.best_params_, clf.best_score_)
    #     return
    # elif isinstance(clf, MLARAM):
    #     parameters = {'vigilance': [0.8, 0.85, 0.9, 0.95, 0.99],
    #         'threshold': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]}
    #     clf = GridSearchCV(clf, parameters, scoring='f1_macro')
    #     clf.fit(X_train, Y_train)
    #     print(clf.best_params_, clf.best_score_)
    #     return

    clf.fit(X_train, Y_train)

    test_score = clf.score(X_test, Y_test)
    print(f'test score >>>>>>>>>> {test_score}')

    prediction_output = clf.predict(sentence_embedding_matrix)

    print(f'passing prediction_output of shape {prediction_output.shape} to add_classification_to_csv')

    add_classification_to_csv(clf, prediction_output)


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

# sklearn classifiers:
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = MultinomialNB()
# clf = GaussianNB()
# clf = tree.DecisionTreeClassifier()
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = MLPClassifier(alpha=1, max_iter=1000)
# clf = AdaBoostClassifier()

# scikit multilabel classifiers:
# clf = MLkNN(k=3, s=0.5)
# clf = BRkNNaClassifier(k=3)
# clf = MLARAM(threshold=0.04, vigilance=0.99)
# clf = BinaryRelevance(
#     classifier=KNeighborsClassifier(n_neighbors=1)
# )
# clf = BinaryRelevance(
#     classifier=tree.DecisionTreeClassifier()
# )
# clf = BinaryRelevance(
#     classifier=RandomForestClassifier(max_depth=2, random_state=0)
# )
# clf = BinaryRelevance(
#     classifier=MLPClassifier(alpha=1, max_iter=1000)
# )
# clf = ClassifierChain(
#     classifier=KNeighborsClassifier(n_neighbors=1)
# )
# clf = ClassifierChain(
#     classifier=tree.DecisionTreeClassifier()
# )
# clf = ClassifierChain(
#     classifier=RandomForestClassifier(max_depth=2, random_state=0)
# )
clf = ClassifierChain(
    classifier=MLPClassifier(alpha=1, max_iter=1000)
)

classify(sentence_embedding_matrix, clf)


# plot_training_data()
