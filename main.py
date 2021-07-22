import sys
import re
import docx
import pandas as pd
from PySide2.QtCore import QDir, QObject, QThread, QUrl, Signal, Slot
from PySide2.QtWebChannel import QWebChannel
from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
from PySide2.QtWidgets import QApplication, QMainWindow

import time


class WebEnginePage(QWebEnginePage):
    def __init__(self, parent=None):
        QWebEnginePage.__init__(self, parent)

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceId):
        print('js', level, message, lineNumber, sourceId)


class TextThread(QThread):
    thread_signal = Signal('QVariant')

    def run(self):
        minimum_proba = 0.95
        # hard-coded
        doc_path = 'text/reorder_exit.docx'

        data = []

        document = docx.Document(doc_path)
        whole_text = []
        for paragraph in document.paragraphs:
            whole_text.append(paragraph.text)
        whole_text = '\n\n'.join(whole_text)

        # obj['whole_text'] = whole_text

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

        # obj['train_data'] = train_data
        # obj['predict_data'] = predict_data

        data = [whole_text, train_data, predict_data]
    
        self.thread_signal.emit(data)


class TableThread(QThread):
    thread_signal = Signal('QVariant')

    def run(self):
        data = []

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

        minimum_proba = 0.95
        
        analyse_df = pd.read_csv(doc_path.replace('.docx', '_analyse.csv'))
        train_df = pd.read_csv(doc_path.replace('.docx', '_train.csv'))
        predict_df = pd.read_csv(doc_path.replace('.docx', '_predict.csv'))
        train_keywords_df = pd.read_csv(doc_path.replace('.docx', '_train_keywords.csv'))
        predict_keywords_df = pd.read_csv(doc_path.replace('.docx', '_predict_keywords.csv'))

        for index, row in analyse_df.iterrows():
            if index == 0:
                counter_row = []

                for theme in themes:
                    counter_row.append(row[theme])
                
                data.append(counter_row)
            else:
                data_row = {}

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
                        
                        data_row[theme] = [text, predict_sentences, train_sentences]
                    else:
                        data_row[theme] = ''

                data.append(data_row)
        # print(obj)
        self.thread_signal.emit(data)


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
    def send_text(self, obj):
        end = time.time()
        print(f'Text (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit(obj)


class TableBackend(QObject):
    signal = Signal('QVariant')
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = TableThread(self)
        self.thread.thread_signal.connect(self.send_table)      

    # TODO: Differentiate different table types (in args?)
    @Slot()
    def get_table(self):
        self.start = time.time()
        self.thread.start()

    @Slot(str)
    def send_table(self, obj):
        end = time.time()
        print(f'Table (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit(obj)

    
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
        self.table_backend = TableBackend(self.view)
        channel = QWebChannel(self)
        self.page.setWebChannel(channel)
        channel.registerObject('textBackend', self.text_backend)
        channel.registerObject('tableBackend', self.table_backend)
        self.view.load(QUrl.fromLocalFile(QDir.current().filePath('templates/main.html')))
        self.setCentralWidget(self.view)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())
