# needed for pyinstaller for MacOS
import os
import sys

# # needed to build with pyinstaller on MacOS
# if getattr(sys, 'frozen', False) and sys.platform == 'darwin':
#     os.environ['QTWEBENGINEPROCESS_PATH'] = os.path.normpath(os.path.join(
#         sys._MEIPASS, 'PySide2', 'Qt', 'lib',
#         'QtWebEngineCore.framework', 'Helpers', 'QtWebEngineProcess.app',
#         'Contents', 'MacOS', 'QtWebEngineProcess'
#     ))

import re
import docx
import json
import time
import pandas as pd
import datetime
from shutil import copyfile
from PySide2.QtCore import QDir, QObject, QThread, QUrl, Signal, Slot
from PySide2.QtWebChannel import QWebChannel
from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog
from classify_docx import ClassifyDocx
from analyse_train_and_predict import analyse


def load_table_data(doc_path, themes, table_name, reclassified):
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


class SetupThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')
    classify_docx = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        print('===========================> SETUP THREAD STARTED')
        print(f'doc_path = {self.app_window.doc_path}')
        print(f'codes_folder_path = {self.app_window.codes_folder_path}')
        print(f'theme_code_table_path = {self.app_window.theme_code_table_path}')

        themes_found = self.classify_docx.run_classifier()
        self.app_window.themes = themes_found

        print('done with run_classifier in setup thread')
        print('running analyse now..')
        print(f'===========================> themes = {self.app_window.themes}')
        analyse(self.app_window.doc_path, self.app_window.themes)
        self.thread_signal.emit(self.app_window.themes)


class TextThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')
    reclassified = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        print('===========================> TEXT THREAD STARTED')
        minimum_proba = 0.95

        document = docx.Document(self.app_window.doc_path)
        whole_text = []
        for paragraph in document.paragraphs:
            whole_text.append(paragraph.text)
        whole_text = '\n\n'.join(whole_text)

        if self.reclassified:
            train_df = pd.read_csv(self.app_window.doc_path.replace('.docx', '_train_1.csv'))
            predict_df = pd.read_csv(self.app_window.doc_path.replace('.docx', '_predict_1.csv'))
        else:
            train_df = pd.read_csv(self.app_window.doc_path.replace('.docx', '_train.csv'))
            predict_df = pd.read_csv(self.app_window.doc_path.replace('.docx', '_predict.csv'))

        train_data = []
        for _, row in train_df.iterrows():
          train_data.append([row['original_sentence'], re.sub(';', ',', row['themes'])])

        predict_data = []
        for index, row in predict_df.iterrows():
          row_themes = ''
          for i in range(3, len(predict_df.columns) - len(self.app_window.themes), 1):
            if (predict_df.iloc[index, i] == 1 and 
            row[f'{predict_df.columns[i]} probability'] > minimum_proba):
              if len(row_themes) == 0:
                row_themes = predict_df.columns[i]
              else:
                row_themes += f', {predict_df.columns[i]}'
          
          if len(row_themes) > 0:
            predict_data.append([row['original_sentence'], row_themes])

        data = [whole_text, train_data, predict_data]
    
        self.thread_signal.emit(data)


class CodesTableThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        print('===========================> CODES TABLE THREAD STARTED')
        data = []
        counts = [0 for _ in self.app_window.themes]

        train_df = pd.read_csv(self.app_window.doc_path.replace('.docx', '_train.csv'))
        codes_df = pd.read_csv(self.app_window.doc_path.replace('.docx', '_codes.csv'))

        for _, row in train_df.iterrows():
            if row['codes'] != '':
                for i, theme in enumerate(self.app_window.themes):
                    counts[i] += int(row[theme])

        titles = []
        for i, theme in enumerate(self.app_window.themes):
            titles.append(f'{theme} ({counts[i]})')

        data.append(titles)
        
        for _, codes_row in codes_df.iterrows():
            table_row = {}
            for theme in self.app_window.themes:
                code = codes_row[theme]
                if isinstance(code, str):
                    sentences = []
                    for _, train_row in train_df.iterrows():
                        if code in train_row['codes']:
                            sentences.append(train_row['original_sentence'])
                        table_row[theme] = [f'{code} ({len(sentences)})', sentences]
            data.append(table_row)
        
        self.thread_signal.emit(data)    


class AllTableThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')
    reclassified = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        print('===========================> ALL TABLE THREAD STARTED')
        data = load_table_data(self.app_window.doc_path, self.app_window.themes, 'all-table', self.reclassified)
        # with open('all_table_data.json', 'w') as f:
        #     json.dump(data, f)
        self.thread_signal.emit(data)


class PredictTableThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')
    reclassified = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        print('===========================> PREDICT TABLE THREAD STARTED')
        data = load_table_data(self.app_window.doc_path, self.app_window.themes, 'predict-table', self.reclassified)
        self.thread_signal.emit(data)


class TrainTableThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')
    reclassified = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        print('===========================> TRAIN TABLE THREAD STARTED')
        data = load_table_data(self.app_window.doc_path, self.app_window.themes, 'train-table', self.reclassified)
        self.thread_signal.emit(data)


class ReclassifyThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')
    table_changed_data = None
    first_reclassify = None
    classify_docx = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        print('===========================> RECLASSIFY THREAD STARTED')
        if self.first_reclassify:
            train_path = self.app_window.doc_path.replace('.docx', '_train.csv')
            predict_path = self.app_window.doc_path.replace('.docx', '_predict.csv')
        else:
            train_path = self.app_window.doc_path.replace('.docx', '_train_1.csv')
            predict_path = self.app_window.doc_path.replace('.docx', '_predict_1.csv')
        
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
                        sentence = row['original_sentence'].replace(r'\\t', '\t').replace(r'\\n', '\n')
                        if sentence in train_sentences:
                            train_df.loc[index, moving_column] = '0'
                            if target_column is not None:
                                train_df.loc[index, 'themes'] = row['themes'].replace(moving_column, target_column)
                                train_df.loc[index, target_column] = '1'

            if len(predict_sentences) > 0:
                for index, row in predict_df.iterrows():
                    # unescape \t and \n
                    sentence = row['original_sentence'].replace(r'\\t', '\t').replace(r'\\n', '\n')
                    if sentence in predict_sentences:
                        cleaned_sentence = predict_df.loc[index, 'cleaned_sentence']
                        sentence_embedding = predict_df.loc[index, 'sentence_embedding']
                        if target_column is not None:
                            all_themes = [target_column]
                        else:
                            all_themes = []
                            for theme in self.app_window.themes: 
                                if (theme != moving_column and theme != target_column and 
                                str(predict_df.loc[index, theme].any()) == '1'):
                                    all_themes.append(theme)

                                predict_as_train_row = {
                                    'file_name': self.app_window.doc_path,
                                    'comment_id': '',
                                    'original_sentence': sentence,
                                    'cleaned_sentence': cleaned_sentence,
                                    'sentence_embedding': sentence_embedding,
                                    'codes': '',
                                    'themes': '; '.join(sorted(all_themes))
                                }

                                for theme in self.app_window.themes:
                                    if theme in all_themes:
                                        predict_as_train_row[theme] = '1'
                                    else:
                                        predict_as_train_row[theme] = '0'

                                train_df = train_df.append(predict_as_train_row, ignore_index=True)


        new_train_path = self.app_window.doc_path.replace('.docx', '_train_1.csv')
        train_df.to_csv(new_train_path, index=False)

        print('running run_classifier...')
        self.classify_docx.run_classifier(new_train_path)
        print('done!')

        print('running analyse...')
        analyse(self.app_window.doc_path, self.app_window.themes, new_train_path)
        print('done!')

        self.app_window.classify_counter += 1

        self.thread_signal.emit('done')


class ConfusionTablesThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        print('===========================> CONFUSION TABLES THREAD STARTED')
        data = []

        for theme in self.app_window.themes:
            table_data = []

            start_path = re.search(r'^(.*[\\\/])', self.app_window.doc_path).group(0)
            end_path = re.search(r'([^\/]+).$', self.app_window.doc_path).group(0)
            end_path = end_path.replace('.docx', f'_{theme.replace(" ", "_")}_cm.csv')

            cm_path = f'{start_path}cm/{end_path}'

            cm_df = pd.read_csv(cm_path)
            cm_analyse_df = pd.read_csv(cm_path.replace('.csv', '_analyse.csv'))
            cm_keywords_df = pd.read_csv(cm_path.replace('.csv', '_keywords.csv'))

            table_data.append(list(cm_analyse_df.columns)) # titles

            for _, analyse_row in cm_analyse_df.iterrows():
                data_row = {}
                for column_index, column in enumerate(cm_analyse_df.columns):
                    text = analyse_row[column]
                    if isinstance(text, str):
                        word = re.sub(r' \(\d+\)$', '', text)
                        sentences = []

                        matching_indices = cm_keywords_df.loc[cm_keywords_df['word'] == word, 'sentences'].item()

                        if isinstance(matching_indices, str):
                            indices_lists = json.loads(matching_indices)

                            for index in indices_lists[column_index]:
                                sentence = cm_df.iloc[index, column_index]
                                sentences.append(sentence)
                    
                        data_row[column] = [text, sentences]
                table_data.append(data_row)
            
            data.append(table_data)
            print(f'"{theme}" confusion table done')

        # with open('cm_data.json', 'w') as f:
        #     json.dump(data, f)
            
        self.thread_signal.emit(data)


class LogThread(QThread):
    app_window = None
    data = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()
    
    def run(self):
        with open('logs/app.log', 'a') as f:
            if 'setup finished' in self.data or 'reclassify' in self.data:
                self.data += f' (logs/models/{self.app_window.doc_file_name}_xgbmodel_{self.app_window.classify_counter}.pickle)'
            elif 'all table finished loading' in self.data:
                self.data += f' (logs/data/{self.app_window.doc_file_name}_analyse_{self.app_window.classify_counter}.csv)'
            elif 'predict table finished loading' in self.data:
                self.data += f' (logs/data/{self.app_window.doc_file_name}_predict_analyse_{self.app_window.classify_counter}.csv)'
            elif 'train table finished loading' in self.data:
                self.data += f' (logs/data/{self.app_window.doc_file_name}_train_analyse_{self.app_window.classify_counter}.csv)'

            f.write(f'{self.data}\n')
            f.close()


class KeywordsThread(QThread):
    app_window = None
    thread_signal = Signal('QVariant')
    input_regexp = None
    regular_expression = None
    case_insensitive = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.app_window = parent.parent().parent()

    def run(self):
        if self.regular_expression:
            if self.case_insensitive and not self.input_regexp.strip().startswith('(?i)'):
                self.app_window.regexp = '(?i)' + self.input_regexp
            else:
                self.app_window.regexp = self.input_regexp
        else:
            if self.case_insensitive:
                self.app_window.regexp = '(?i)(' + re.sub(r'; |;', '|', self.input_regexp).strip() + ')'
            else:
                self.app_window.regexp = '(' + re.sub(r'; |;', '|', self.input_regexp).strip() + ')'
        try: 
            re.compile(self.app_window.regexp)
            valid = True
        except re.error:
            valid = False

        print(f'saved regexp =================================> {self.app_window.regexp}')

        self.thread_signal.emit(['regexp', self.app_window.regexp, valid])


class SetupBackend(QObject):
    app_window = None
    signal = Signal('QVariant')
    classify_docx = None
    start = None

    def __init__(self, classify_docx, parent=None):
        QObject.__init__(self, parent)
        self.app_window = parent.parent()
        self.classify_docx = classify_docx
        self.thread = SetupThread(self)
        self.thread.thread_signal.connect(self.send_data)

    @Slot(str, str, str, str)
    def set_up(self, transcript_path, codes_dir_path, theme_code_lookup_path, interviewer_regexp):
        # copy transcript into text folder
        end_doc_path = re.search(r'([^\/]+).$', transcript_path).group(0)
        self.app_window.doc_path = f'text/{end_doc_path}'
        try:
            copyfile(transcript_path, self.app_window.doc_path)
        except:
            print('transcript already in text')

        if theme_code_lookup_path != '':
            end_theme_code_lookup_path = re.search(r'([^\/]+).$', theme_code_lookup_path).group(0)
            cat_path = f'text/{end_theme_code_lookup_path}'
            try:
                copyfile(theme_code_lookup_path, cat_path)
            except:
                print('theme-code lookup tabe already in text')

        self.app_window.doc_file_name = re.search(r'([^\/]+).$', self.app_window.doc_path).group(0).replace('.docx', '')
        self.app_window.codes_folder_path = codes_dir_path
        self.app_window.theme_code_table_path = theme_code_lookup_path

        if theme_code_lookup_path != '':
            cat_df = pd.read_csv(self.app_window.theme_code_table_path, encoding='utf-8-sig')
            self.app_window.themes = list(cat_df)     

        # set paths and regexp for classify_docx object
        self.classify_docx.set_up(transcript_path, self.app_window.codes_folder_path, self.app_window.theme_code_table_path, interviewer_regexp)
        # pass classify_docx object to thread
        self.thread.classify_docx = self.classify_docx
        self.start = time.time()
        self.thread.start()
        
    @Slot(str)
    def send_data(self, data):
        end = time.time()
        print(f'Set up (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit(data)
        

class TextBackend(QObject):
    signal = Signal('QVariant')
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = TextThread(self)
        self.thread.thread_signal.connect(self.send_text)

    @Slot(bool)
    def get_text(self, reclassified):
        self.thread.reclassified = reclassified
        self.start = time.time()
        self.thread.start()

    @Slot(str)
    def send_text(self, data):
        end = time.time()
        print(f'Text (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit(data)


class CodesTableBackend(QObject):
    signal = Signal('QVariant')
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = CodesTableThread(self)
        self.thread.thread_signal.connect(self.send_table)

    @Slot()
    def get_table(self):
        self.start = time.time()
        self.thread.start()

    @Slot(str)
    def send_table(self, data):
        end = time.time()
        print(f'Codes Table (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit(data)


class AllTableBackend(QObject):
    signal = Signal('QVariant')
    reclassified = None
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = AllTableThread(self)
        self.thread.thread_signal.connect(self.send_table)      

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
    classify_docx = None
    signal = Signal('QVariant')
    start = None

    def __init__(self, classify_docx, parent=None):
        QObject.__init__(self, parent)
        self.classify_docx = classify_docx
        self.thread = ReclassifyThread(self) # cannot be in get_data or send_data
        self.thread.thread_signal.connect(self.send_data) # does not emit data

    @Slot(list, bool)
    def get_data(self, table_changed_data, first_reclassify):
        self.thread.table_changed_data = table_changed_data
        self.thread.first_reclassify = first_reclassify

        # pass classify_docx object to thread
        self.thread.classify_docx = self.classify_docx
        self.start = time.time()
        self.thread.start()

    @Slot()
    def send_data(self):
        end = time.time()
        print(f'Reclassify (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit('done')


class ConfusionTablesBackend(QObject):
    signal = Signal('QVariant')
    start = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = ConfusionTablesThread(self)
        self.thread.thread_signal.connect(self.send_data)

    @Slot()
    def get_data(self):
        self.start = time.time()
        self.thread.start()

    @Slot(str)
    def send_data(self, data):
        end = time.time()
        print(f'Confusion Tables (Python) => {round(end-self.start, 2)} seconds')
        self.signal.emit(data)


class LogBackend(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = LogThread(self)

    @Slot(str)
    def log(self, data):
        self.thread.data = data
        self.thread.start()


class ImportBackend(QObject):    
    signal = Signal('QVariant')

    def __init__(self, parent=None, main_window=None):
        QObject.__init__(self, parent)
        self._main_window = main_window
        self.thread = KeywordsThread(self)
        self.thread.thread_signal.connect(self.send_keywords_data)
        
    @Slot()
    def open_transcript_chooser(self):                
        path_to_file, _ = QFileDialog.getOpenFileName(self._main_window, self.tr('Import Document'), self.tr('~/Desktop/'), self.tr('Document (*.docx)'))
        path_to_file = path_to_file.replace('\\', '/')
        self.signal.emit(['transcript', path_to_file])

    @Slot()
    def open_codes_chooser(self):
        path_to_folder = QFileDialog.getExistingDirectory(self._main_window, self.tr('Choose Directory'), self.tr('~/Desktop/'), QFileDialog.ShowDirsOnly)
        path_to_folder = path_to_folder.replace('\\', '/')
        self.signal.emit(['codes', path_to_folder])

    @Slot()
    def open_theme_code_table_chooser(self):
        path_to_file, _ = QFileDialog.getOpenFileName(self._main_window, self.tr('Import Table'), self.tr('~/Desktop/'), self.tr('Table (*.csv)'))
        path_to_file = path_to_file.replace('\\', '/')
        self.signal.emit(['codeThemeTable', path_to_file])

    @Slot(str, bool, bool)
    def save_regexp(self, input_regexp, regular_expression, case_insensitive):
        self.thread.input_regexp = input_regexp
        self.thread.regular_expression = regular_expression
        self.thread.case_insensitive = case_insensitive
        self.thread.start()

    @Slot(str)
    def send_keywords_data(self, data):
        self.signal.emit(data)

    
class WebView(QWebEngineView):
    def __init__(self, parent=None):
        QWebEngineView.__init__(self, parent)
        self.setPage(WebEnginePage(self))

    def contextMenuEvent(self, event):
        pass


class AppWindow(QMainWindow):
    doc_path = ''
    theme_code_table_path = ''
    codes_folder_path = ''
    doc_file_name = ''
    regexp = ''
    themes = []
    classify_counter = 0

    def __init__(self):
        QMainWindow.__init__(self)
        self.resize(1400, 800)
        self.view = WebView(self)
        self.page = self.view.page()
        # create classify_docx object to pass to setup_backend and reclassify_backend
        classify_docx = ClassifyDocx()
        self.setup_backend = SetupBackend(classify_docx, self.view)
        self.text_backend = TextBackend(self.view)
        self.codes_table_backend = CodesTableBackend(self.view)
        self.all_table_backend = AllTableBackend(self.view)
        self.predict_table_backend = PredictTableBackend(self.view)
        self.train_table_backend = TrainTableBackend(self.view)
        self.reclassify_backend = ReclassifyBackend(classify_docx, self.view)
        self.confusion_tables_backend = ConfusionTablesBackend(self.view)
        self.log_backend = LogBackend(self.view)
        self.import_backend = ImportBackend(self.view)
        channel = QWebChannel(self)
        self.page.setWebChannel(channel)
        channel.registerObject('setupBackend', self.setup_backend)
        channel.registerObject('textBackend', self.text_backend)
        channel.registerObject('codesTableBackend', self.codes_table_backend)
        channel.registerObject('allTableBackend', self.all_table_backend)
        channel.registerObject('predictTableBackend', self.predict_table_backend)
        channel.registerObject('trainTableBackend', self.train_table_backend)
        channel.registerObject('reclassifyBackend', self.reclassify_backend)
        channel.registerObject('confusionTablesBackend', self.confusion_tables_backend)
        channel.registerObject('logBackend', self.log_backend)
        channel.registerObject('importBackend', self.import_backend)
        self.view.load(QUrl.fromLocalFile(QDir.current().filePath('templates/main.html')))
        self.setCentralWidget(self.view)


def log_close():
    with open('logs/app.log', 'a') as f:
        f.write(f'[{datetime.date.today().strftime("%d/%m/%Y")}, {datetime.datetime.now().strftime("%H:%M:%S")} ({round(time.time() * 1000)})]: app closed\n')
        f.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    app.aboutToQuit.connect(log_close)
    sys.exit(app.exec_())
