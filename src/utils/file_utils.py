import shutil
import os


def clear_directory(folder):

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)