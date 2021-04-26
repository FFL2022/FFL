import os
import sqlite3
import numpy as np

def make_dir_if_not_exists(path):
    try:
        os.makedirs(path)
    except:
        pass

def write_to_file(x, filepath):
    with open(filepath, "w") as f:
        f.write(x)

def get_coverage(filename, nline_removed):
    
    def process_line(line):
        tag, line_no, code = line.strip().split(':', 2)
        return tag.strip(), int(line_no.strip()), code
    
    coverage = {}
    with open(filename, "r") as f:
        gcov_file = f.read()
        for idx, line in enumerate(gcov_file.split('\n')):
            if idx <= 4 or len(line.strip()) == 0:
                continue
            
            try:
                tag, line_no, code = process_line(line)
            except:
                print('idx:', idx, 'line:', line)
                print(line.strip().split(':', 2))
                raise
            assert idx!=5 or line_no==1, gcov_file
        
            if tag == '-':
                continue
            elif tag == '#####':
                coverage[line_no - nline_removed] = 0
            else:  
                tag = int(tag) 
                coverage[line_no - nline_removed] = 1
        return coverage

def remove_lib(filename):
    count = 0
    start = False
    with open(filename, "r") as f:
        with open("temp.c", "w") as t:
            for line in f:
                if line[0] in ("#") or len(line.strip()) == 0 and not start:
                    count += 1
                else:
                    t.write(line)
                    if not start:
                        start = True
        # with open("temp.c", "r") as x:
        #     print(x.read())
    return count
