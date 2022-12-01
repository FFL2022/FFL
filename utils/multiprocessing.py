import importlib
import os
import pickle as pkl
from multiprocessing import Process, Queue
from utils.nprocs import nprocs
from typing import Iterable
import glob

def test_func(k):
    return k+1

def process(qq_in, qq_out, func):
    while True:
        msg = qq_in.get()
        if msg[0] == 'DONE':
            qq_out.put(['DONE', None])
            break
        _, data = msg
        if not isinstance(data, Iterable):
            out = func(data)
        else:
            out = func(*data)
        qq_out.put(['DATA', out])


def start_procs(qq_in, qq_out, func):
    all_reader_procs = list()
    for ii in range(0, nprocs):
        reader_p = Process(target=process,
                args=(qq_in, qq_out, func))
        reader_p.daemon = True
        reader_p.start()  # Launch reader_p() as another proc

        all_reader_procs.append(reader_p)
    return all_reader_procs


def multi_process_data(qq_in, qq_out, func, all_datas):
    all_data_procs = start_procs(qq_in, qq_out, func)
    for data in all_datas:
        qq_in.put(('continue', data))
    for ii in range(nprocs):
        qq_in.put(("DONE", 0))

    rem_procs = len(all_data_procs)
    while rem_procs:
        msg, data_out = qq_out.get()
        print(msg, data_out)
        if msg == 'DONE':
            rem_procs -= 1
            continue
        yield data_out

    for idx, a_reader_proc in enumerate(all_data_procs):
        print(f"    Waiting for data processing {idx} to join")
        a_reader_proc.join()  # Wait for a_reader_proc() to finish
