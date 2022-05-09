from os import path


def resource_path(relative_path):
    base_path = path.abspath(path.dirname(__file__))
    path_to_resource = path.join(base_path, relative_path)

    return path_to_resource