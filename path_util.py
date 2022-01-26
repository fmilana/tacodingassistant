import sys
import os


def resource_path(relative_path):
    try:
        base_path = os.path.dirname(os.path.dirname(sys._MEIPASS))
    except AttributeError:
        base_path = os.path.abspath('.')

    return os.path.join(base_path, relative_path)