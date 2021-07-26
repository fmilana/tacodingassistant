import sys
import re
import docx
import pandas as pd
from PySide2.QtCore import QDir, QObject, QThread, QUrl, Signal, Slot
from PySide2.QtWebChannel import QWebChannel
from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
from PySide2.QtWidgets import QApplication, QMainWindow
from classify_docx import run_classifier
from analyse_train_and_predict import analyse

import time


# hard-coded
doc_path = 'text/reorder_exit.docx'
themes = [
    'practices',
    'social',
    'study vs product',
    'system use',
    'system perception',
    'value judgements'
]

first_reclassify = True


def load_table_data(table_name, reclassified):
    table = []

    minimum_proba = 0.95

    if table_name == 'all-table':
        if reclassified:
            analyse_df = pd.read_csv(doc_path.replace('.docx', '_analyse_1.csv'))
            predict_df = pd.read_csv(doc_path.replace('.docx', '_predict_1.csv'))
            train_df = pd.read_csv(doc_path.replace('.docx', '_train_1.csv'))
            predict_keywords_df = pd.read_csv(doc_path.replace('.docx', '_predict_keywords_1.csv'))
            train_keywords_df = pd.read_csv(doc_path.replace('.docx', '_train_keywords_1.csv'))
        else:
            analyse_df = pd.read_csv(doc_path.replace('.docx', '_analyse.csv'))
            predict_df = pd.read_csv(doc_path.replace('.docx', '_predict.csv'))
            train_df = pd.read_csv(doc_path.replace('.docx', '_train.csv'))
            predict_keywords_df = pd.read_csv(doc_path.replace('.docx', '_predict_keywords.csv'))
            train_keywords_df = pd.read_csv(doc_path.replace('.docx', '_train_keywords.csv'))

        for index, row in analyse_df.iterrows():
            if index == 0:
                counter_row = []

                for theme in themes:
                    counter_row.append(row[theme])
                
                table.append(counter_row)
            else:
                table_row = {}

                for theme in themes:
                    text = row[theme]

                    if isinstance(text, str) and len(text) > 0:
                        word = re.sub(r' \(\d+\)', '', text).lower()
                        predict_sentences = []
                        train_sentences = []

                        predict_keywords_row = predict_keywords_df.loc[predict_keywords_df['word'] == word]

                        if not predict_keywords_row.empty:
                            predict_indeces = predict_keywords_row['sentences'].values[0].strip('[]').split(', ')
                        
                            for index in predict_indeces:
                                predict_row = predict_df.iloc[int(index)]
                                if predict_row[theme] == 1 and predict_row[f'{theme} probability'] > minimum_proba:
                                    predict_sentences.append(predict_row['original_sentence'])
                        
                        train_keywords_row = train_keywords_df.loc[train_keywords_df['word'] == word]

                        if not train_keywords_row.empty:
                            train_indeces = train_keywords_row['sentences'].values[0].strip('[]').split(', ')

                            for index in train_indeces:
                                train_row = train_df.iloc[int(index)]
                                if train_row[theme] == 1:
                                    train_sentences.append(train_row['original_sentence'])
                        
                        table_row[theme] = [text, predict_sentences, train_sentences]
                    else:
                        table_row[theme] = None

                table.append(table_row)

    elif table_name == 'predict-table':
        if reclassified:
            predict_analyse_df = pd.read_csv(doc_path.replace('.docx', '_predict_analyse_1.csv'))
            predict_df = pd.read_csv(doc_path.replace('.docx', '_predict_1.csv'))
            predict_keywords_df = pd.read_csv(doc_path.replace('.docx', '_predict_keywords_1.csv'))
        else:
            predict_analyse_df = pd.read_csv(doc_path.replace('.docx', '_predict_analyse.csv'))
            predict_df = pd.read_csv(doc_path.replace('.docx', '_predict.csv'))
            predict_keywords_df = pd.read_csv(doc_path.replace('.docx', '_predict_keywords.csv'))

        for index, row in predict_analyse_df.iterrows():
            if index == 0:
                counter_row = []

                for theme in themes:
                    counter_row.append(row[theme])
                
                table.append(counter_row)
            else:
                table_row = {}

                for theme in themes:
                    text = row[theme]

                    if isinstance(text, str) and len(text) > 0:
                        word = re.sub(r' \(\d+\)', '', text).lower()
                        predict_sentences = []

                        predict_keywords_row = predict_keywords_df.loc[predict_keywords_df['word'] == word]

                        if not predict_keywords_row.empty:
                            predict_indeces = predict_keywords_row['sentences'].values[0].strip('[]').split(', ')
                        
                            for index in predict_indeces:
                                predict_row = predict_df.iloc[int(index)]
                                if predict_row[theme] == 1 and predict_row[f'{theme} probability'] > minimum_proba:
                                    predict_sentences.append(predict_row['original_sentence'])
                        
                        table_row[theme] = [text, predict_sentences]
                    else:
                        table_row[theme] = None

                table.append(table_row)

    elif table_name == 'train-table':
        if reclassified:
            train_analyse_df = pd.read_csv(doc_path.replace('.docx', '_train_analyse_1.csv'))
            train_df = pd.read_csv(doc_path.replace('.docx', '_train_1.csv'))
            train_keywords_df = pd.read_csv(doc_path.replace('.docx', '_train_keywords_1.csv'))
        else:
            train_analyse_df = pd.read_csv(doc_path.replace('.docx', '_train_analyse.csv'))
            train_df = pd.read_csv(doc_path.replace('.docx', '_train.csv'))
            train_keywords_df = pd.read_csv(doc_path.replace('.docx', '_train_keywords.csv'))

        for index, row in train_analyse_df.iterrows():
            if index == 0:
                counter_row = []

                for theme in themes:
                    counter_row.append(row[theme])
                
                table.append(counter_row)
            else:
                table_row = {}

                for theme in themes:
                    text = row[theme]

                    if isinstance(text, str) and len(text) > 0:
                        word = re.sub(r' \(\d+\)', '', text).lower()
                        train_sentences = []
                        
                        train_keywords_row = train_keywords_df.loc[train_keywords_df['word'] == word]

                        if not train_keywords_row.empty:
                            train_indeces = train_keywords_row['sentences'].values[0].strip('[]').split(', ')

                            for index in train_indeces:
                                train_row = train_df.iloc[int(index)]
                                if train_row[theme] == 1:
                                    train_sentences.append(train_row['original_sentence'])
                        
                        table_row[theme] = [text, train_sentences]
                    else:
                        table_row[theme] = None

                table.append(table_row)

    return table


class WebEnginePage(QWebEnginePage):
    def __init__(self, parent=None):
        QWebEnginePage.__init__(self, parent)

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceId):
        print('js', level, message, lineNumber, sourceId)


class TextThread(QThread):
    thread_signal = Signal('QVariant')

    def run(self):
        print('===========================> TEXT THREAD STARTED!!')
        minimum_proba = 0.95
        # hard-coded
        doc_path = 'text/reorder_exit.docx'

        data = []

        document = docx.Document(doc_path)
        whole_text = []
        for paragraph in document.paragraphs:
            whole_text.append(paragraph.text)
        whole_text = '\n\n'.join(whole_text)

        train_df = pd.read_csv(doc_path.replace('.docx', '_train.csv'))
        predict_df = pd.read_csv(doc_path.replace('.docx', '_predict.csv'))

        train_data = []
        for _, row in train_df.iterrows():
          train_data.append([row['original_sentence'], re.sub(';', ',', row['themes'])])

        predict_data = []
        for index, row in predict_df.iterrows():
          themes = ''
          for i in range(3, len(predict_df.columns), 1):
            if (predict_df.iloc[index, i] == 1 and 
            row[f'{predict_df.columns[i]} probability'] > minimum_proba):
              if len(themes) == 0:
                themes = predict_df.columns[i]
              else:
                themes += f', {predict_df.columns[i]}'
          
          if len(themes) > 0:
            predict_data.append([row['original_sentence'], themes])

        data = [whole_text, train_data, predict_data]
    
        self.thread_signal.emit(data)


class AllTableThread(QThread):
    thread_signal = Signal('QVariant')
    reclassified = None

    def run(self):
        print('===========================> ALL TABLE THREAD STARTED!!')
        data = load_table_data('all-table', self.reclassified)
        self.thread_signal.emit(data)


class PredictTableThread(QThread):
    thread_signal = Signal('QVariant')
    reclassified = None

    def run(self):
        print('===========================> PREDICT TABLE THREAD STARTED!!')
        data = load_table_data('predict-table', self.reclassified)
        self.thread_signal.emit(data)


class TrainTableThread(QThread):
    thread_signal = Signal('QVariant')
    reclassified = None

    def run(self):
        print('===========================> TRAIN TABLE THREAD STARTED!!')
        data = load_table_data('train-table', self.reclassified)
        self.thread_signal.emit(data)


class ReclassifyThread(QThread):
    thread_signal = Signal('QVariant')
    table_changed_data = None
    first_reclassify = None

    def run(self):
        if self.first_reclassify:
            train_path = doc_path.replace('.docx', '_train.csv')
            predict_path = doc_path.replace('.docx', '_predict.csv')
        else:
            train_path = doc_path.replace('.docx', '_train_1.csv')
            predict_path = doc_path.replace('.docx', '_predict_1.csv')
        
        train_df = pd.read_csv(train_path)
        predict_df = pd.read_csv(predict_path)

        for data_row in self.table_changed_data:
            train_sentences = []
            predict_sentences = []

            if 'trainSentences' in data_row['movingSentences']:
                train_sentences = data_row['movingSentences']['trainSentences']
            if 'predictSentences' in data_row['movingSentences']:
                predict_sentences = data_row['movingSentences']['predictSentences']

            moving_column = data_row['movingColumn']
            target_column = data_row['targetColumn']
            
            if len(train_sentences) > 0:
                for index, row in train_df.iterrows():
                    if row['codes'] != '':
                        # unescape \t and \n
                        sentence = re.sub(r'\\t', '\t', row['original_sentence'])
                        sentence = re.sub(r'\\n', '\n', sentence)
                        if sentence in train_sentences:
                            train_df.loc[index, 'themes'] = row['themes'].replace(moving_column, target_column)
                            train_df.loc[index, moving_column] = '0'
                            train_df.loc[index, target_column] = '1'

            for sentence in predict_sentences:
                # unescape \t and \n
                sentence = re.sub(r'\\t', '\t', sentence)
                sentence = re.sub(r'\\n', '\n', sentence)
                matching_row = predict_df.loc[predict_df['original_sentence'] == sentence]
                cleaned_sentence = str(matching_row['cleaned_sentence'].any())
                sentence_embedding = str(matching_row['sentence_embedding'].any())
                all_themes = [target_column]
                for theme in themes: 
                    if (theme != moving_column and theme != target_column and 
                    str(matching_row[theme].any()) == '1'):
                        all_themes.append(theme)
                
                predict_as_train_row = {
                    'file name': doc_path,
                    'comment_id': '',
                    'original_sentence': sentence,
                    'cleaned_sentence': cleaned_sentence,
                    'sentence_embedding': sentence_embedding,
                    'codes': '',
                    'themes': '; '.join(sorted(all_themes))
                }

                for theme in themes:
                    if theme == target_column:
                        predict_as_train_row[theme] = '1'
                    else:
                        predict_as_train_row[theme] = '0'

                train_df = train_df.append(predict_as_train_row, ignore_index=True)

        new_train_path = doc_path.replace('.docx', '_train_1.csv')
        train_df.to_csv(new_train_path, index=False)

        print('running run_classifier...')
        run_classifier(doc_path, new_train_path)
        print('done!')

        print('running analyse...')
        analyse(new_train_path)
        print('done!')

        # print('loading table data...')
        # all_data = load_table_data('all-table', True)
        # predict_data = load_table_data('predict-table', True)
        # train_data = load_table_data('train-table', True)
        # print('done!')

        self.thread_signal.emit('done')


# partly based on https://stackoverflow.com/a/50610834/6872193
class TextBackend(QObject):
    signal = Signal('QVariant')
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = TextThread(self)
        self.thread.thread_signal.connect(self.send_text)

    @Slot()
    def get_text(self):
        self.start = time.time()
        self.thread.start()

    @Slot(str)
    def send_text(self, data):
        end = time.time()
        print(f'Text (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit(data)


class AllTableBackend(QObject):
    signal = Signal('QVariant')
    reclassified = None
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = AllTableThread(self)
        self.thread.thread_signal.connect(self.send_table)      

    # TODO: Differentiate different table types (in args?)
    @Slot(bool)
    def get_table(self, reclassified):
        self.reclassified = reclassified
        self.thread.reclassified = reclassified
        self.start = time.time()
        self.thread.start()

    @Slot(str)
    def send_table(self, data):
        end = time.time()
        print(f'All Table (Python) => {round(end-self.start, 2)} seconds')
        data_and_reclassified = [data, self.reclassified]
        self.signal.emit(data_and_reclassified)


class PredictTableBackend(QObject):
    signal = Signal('QVariant')
    reclassified = None
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = PredictTableThread(self)
        self.thread.thread_signal.connect(self.send_table)      

    # TODO: Differentiate different table types (in args?)
    @Slot(bool)
    def get_table(self, reclassified):
        self.reclassified = reclassified
        self.thread.reclassified = reclassified
        self.start = time.time()
        self.thread.start()

    @Slot(str)
    def send_table(self, data):
        end = time.time()
        print(f'Predict Table (Python) => {round(end-self.start, 2)} seconds')
        data_and_reclassified = [data, self.reclassified]
        self.signal.emit(data_and_reclassified)


class TrainTableBackend(QObject):
    signal = Signal('QVariant')
    reclassified = None
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = TrainTableThread(self)
        self.thread.thread_signal.connect(self.send_table)      

    # TODO: Differentiate different table types (in args?)
    @Slot(bool)
    def get_table(self, reclassified):
        self.reclassified = reclassified
        self.thread.reclassified = reclassified
        self.start = time.time()
        self.thread.start()

    @Slot(str)
    def send_table(self, data):
        end = time.time()
        print(f'Train Table (Python) => {round(end-self.start, 2)} seconds')
        data_and_reclassified = [data, self.reclassified]
        self.signal.emit(data_and_reclassified)


class ReclassifyBackend(QObject):
    signal = Signal('QVariant')
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = ReclassifyThread(self) # cannot be in get_data or send_data
        self.thread.thread_signal.connect(self.send_data) # does not emit data

    @Slot(list, bool)
    def get_data(self, table_changed_data, first_reclassify):
        self.thread.table_changed_data = table_changed_data
        self.thread.first_reclassify = first_reclassify
        self.start = time.time()
        self.thread.start()

    @Slot()
    def send_data(self):
        end = time.time()
        print(f'Reclassify (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit('done')

    
class WebView(QWebEngineView):
    def __init__(self, parent=None):
        QWebEngineView.__init__(self, parent)
        self.setPage(WebEnginePage(self))

    def contextMenuEvent(self, event):
        pass


class AppWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.resize(1500, 900)
        self.view = WebView(self)
        self.page = self.view.page()
        self.text_backend = TextBackend(self.view)
        self.all_table_backend = AllTableBackend(self.view)
        self.predict_table_backend = PredictTableBackend(self.view)
        self.train_table_backend = TrainTableBackend(self.view)
        self.reclassify_backend = ReclassifyBackend(self.view)
        channel = QWebChannel(self)
        self.page.setWebChannel(channel)
        channel.registerObject('textBackend', self.text_backend)
        channel.registerObject('allTableBackend', self.all_table_backend)
        channel.registerObject('predictTableBackend', self.predict_table_backend)
        channel.registerObject('trainTableBackend', self.train_table_backend)
        channel.registerObject('reclassifyBackend', self.reclassify_backend)
        self.view.load(QUrl.fromLocalFile(QDir.current().filePath('templates/main.html')))
        self.setCentralWidget(self.view)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())
