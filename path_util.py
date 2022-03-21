# # --onedir flag:
import sys
import os


def resource_path(relative_path):
    try:
        base_path = os.path.dirname(os.path.dirname(sys._MEIPASS))
    except AttributeError:
        base_path = os.path.abspath('.')

    return os.path.join(base_path, relative_path)


def logs_resource_path():
    try:
        base_path = os.path.dirname(os.path.dirname(sys._MEIPASS))
    except AttributeError:
        base_path = os.path.abspath('.')

    return os.path.join(base_path, 'logs/app.log')
    

# --onefile flag:
# import sys
# from os import path

# def resource_path(relative_path):
#     bundle_dir = path.abspath(path.dirname(__file__))
#     path_to_resource = path.join(bundle_dir, relative_path)

#     return path_to_resource


# def logs_resource_path():
#     if getattr(sys, 'frozen', False):
#         # we are running in a bundle
#         logs_path = path.join(sys.executable, 'app.log')
#     else:
#         # we are running in a normal Python environment
#         logs_path = 'logs/app.log'

#     return logs_path