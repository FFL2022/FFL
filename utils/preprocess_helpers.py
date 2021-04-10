import os

def make_dir_if_not_exists(path):
    try:
        os.makedirs(path)
    except:
        pass

def write_to_file(x, filepath):
    with open(filepath, "w") as f:
        f.write(x)
