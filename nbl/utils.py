from __future__ import print_function, unicode_literals
import os
import time
import numpy as np
from utils.utils import ConfigClass
import sqlite3
import pickle


def all_buggy_and_fixed(root=ConfigClass.raw_dir):
    dataset_db = os.path.join(root, 'dataset.db')
    all_data = {}
    eval_set = np.load(os.path.join(root, 'data/eval_set.py'),
                       allow_pickle=True).item()
    eval_dict = {}
    for problem_id in eval_set:
        for program_id, row in eval_set[problem_id].items():
            eval_dict[program_id] = row

    eval_set_program_ids = set(eval_dict.keys())

    query='''SELECT p.program_id, program, o.user_id, trs.tests_passed, o.time_stamp FROM
        programs p INNER JOIN orgsource o ON o.program_id = p.program_id
        INNER JOIN problems q ON o.problem_id = q.problem_id
        INNER JOIN test_run_summary trs ON trs.program_id = p.program_id
        WHERE trs.verdict<>"ALL_PASS" AND q.problem_id=?;'''

    with sqlite3.connect(dataset_db) as conn:
        c = conn.cursor()
    problem_ids = [str(row[0]) for
                   row in c.execute(
                       'SELECT DISTINCT problem_id FROM orgsource;')]
    for problem_id in problem_ids:
        all_data[problem_id] = {}
        os.makedirs(root + f"data/sources/{problem_id}", exist_ok=True)
        for row in c.execute(query, (problem_id,)):
            program_id = int(row[0])
            program = row[1]
            uid = row[2]
            verdict = int(row[3])
            time_stamp = float(row[4])
            if program_id not in eval_set_program_ids:
                if uid not in all_data[problem_id]:
                    all_data[problem_id][uid] = {}
                all_data[problem_id][uid][program_id] = [
                    verdict, time_stamp]

    c.close()
    conn.close()

    # intuition: same user, same problem, nearest timestamp with less errors
    # are buggy and fixed version

    for problem_id in all_data:
        for uid in all_data[problem_id]:
            ids = list(all_data[problem_id][uid].keys())
            values = all_data[problem_id][uid]
            # Sort by ascending timestamp
            ids = list(sorted(ids, key=lambda x: values[x][1]))
            for i, _id in enumerate(ids[::-1]):
                # Next time stamp that have improvements are buggy and fixed
                next_id = ids[i+1]
                if values[_id][0] < values[next_id][0]:
                    b_fp = root + "data/sources/{}/{}/{}.c".format(
                        problem_id, uid, _id)
                    f_fp = root + "data/sources/{}/{}/{}.c".format(
                        problem_id, uid, next_id)
                    # Get bug loc
                    yield {"buggy": _id, "fixed": next_id, "b_fp": b_fp,
                           "f_fp": f_fp, "problem_id": problem_id,
                           "uid": uid
                           }


cache_fp = os.path.join(ConfigClass.preprocess_dir_nbl, "all_keys.pkl")

if not os.path.exists(cache_fp):
    all_keys = list(all_buggy_and_fixed())
    pickle.dump(all_keys, open(cache_fp, 'wb'))
else:
    all_keys = pickle.load(open(cache_fp, 'rb'))
