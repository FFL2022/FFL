from __future__ import print_function, unicode_literals
import os
import time
import numpy as np
from utils.utils import ConfigClass
import sqlite3
import pickle

eval_set = pickle.load(open('data_nbl/eval_data.pkl', 'rb'))
eval_dict = np.load('data_nbl/data/eval_set.npy', allow_pickle=True).item()

mapping_eval_fp = os.path.join(ConfigClass.preprocess_dir_nbl,
                               "mapping_eval.pkl")
if not os.path.exists(mapping_eval_fp):
    mapping_eval = {}
    for problem_id in eval_dict:
        for program_id, row in eval_dict[problem_id].items():
            mapping_eval[program_id] = (problem_id, row[0])
    pickle.dump(mapping_eval, open(mapping_eval_fp, 'wb'))
else:
    mapping_eval = pickle.load(open(mapping_eval_fp, 'rb'))


def all_buggy_and_fixed(root=ConfigClass.nbl_raw_dir):
    dataset_db = os.path.join(root, 'dataset.db')
    all_data = {}

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
            if uid not in all_data[problem_id]:
                all_data[problem_id][uid] = {}
            all_data[problem_id][uid][program_id] = [
                verdict, time_stamp]

    c.close()
    conn.close()

    # intuition: same user, same problem, nearest timestamp with less errors
    # are buggy and fixed version

    for _id in mapping_eval:
        problem_id, next_id = mapping_eval[_id]
        yield {'buggy': _id, 'fixed': next_id,
               'b_fp': root + f'sources/{_id}.c',
               'f_fp': root + f'sources/{next_id}.c',
               'problem_id': problem_id, 'uid': uid
               }

    for problem_id in all_data:
        for uid in all_data[problem_id]:
            ids = list(all_data[problem_id][uid].keys())
            values = all_data[problem_id][uid]
            # Sort by ascending timestamp
            ids = list(sorted(ids, key=lambda x: values[x][1]))
            for i, _id in enumerate(ids[:-1]):
                jump = 1
                # Next time stamp that have improvements are buggy and fixed
                next_id = ids[i+jump]

                if _id in mapping_eval:
                    continue

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
