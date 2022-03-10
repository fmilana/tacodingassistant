# # --onedir flag:
# import sys
# import os


# def resource_path(relative_path):
#     try:
#         base_path = os.path.dirname(os.path.dirname(sys._MEIPASS))
#     except AttributeError:
#         base_path = os.path.abspath('.')

#     return os.path.join(base_path, relative_path)

# --onefile flag:
from os import path

def resource_path(relative_path):
    bundle_dir = path.abspath(path.dirname(__file__))
    path_to_resource = path.join(bundle_dir, relative_path)

    return path_to_resource