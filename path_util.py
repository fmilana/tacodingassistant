import sys
from os import path


def resource_path(relative_path):
    if relative_path == 'logs/app.log' and sys.platform == 'win32' and getattr(sys, 'frozen', False):
        # on Windows, find logs in the same folder as the .exe
        base_path = path.abspath(path.dirname(sys.executable))
    else:
        base_path = path.abspath(path.dirname(__file__))
    
    path_to_resource = path.join(base_path, relative_path)

    return path_to_resource