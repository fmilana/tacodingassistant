import sys
from PySide2 import QtCore, QtWebChannel
from PySide2.QtCore import *
from PySide2.QtWebEngine import QtWebEngine
from PySide2.QtWidgets import QApplication
from PySide2.QtWebEngineWidgets import *


if __name__ == '__main__':
    QtWebEngine.initialize()
    app = QApplication(sys.argv)
    view = QWebEngineView()
    # view.load(QUrl.fromLocalFile(
    #           QDir.current().filePath('./templates/index.html')))
    view.load('http://localhost:5000/')
    view.resize(1300, 800)
    view.show()

sys.exit(app.exec_())
