#!/usr/bin/python

from nbl.data_format import test_verdict
from nbl.utils import all_keys, eval_set
from utils.utils import ConfigClass
from utils.train_utils import AverageMeter
import tqdm, sqlite3, operator

db_path = './data_nbl/dataset.db'
with sqlite3.connect(db_path) as conn:
    cursor = c = conn.cursor()

def remove_all_white_space(line):
    return ''.join(line.split())

def insert_sorted(seq, elt):
    idx = 0
    if not seq or elt > seq[-1]:
        seq.append(elt)
    else:
        while elt > seq[idx] and idx < len(seq):
            idx += 1
        seq.insert(idx, elt)

def normalize_brackets(program):
    program = program.replace('\r', '\n')
    nline_removed = []
    lines = []
    idx = 0
    for line in program.split('\n'):
        idx += 1
        if len(line.strip()) > 0:
            lines.append(line)
        else:
            nline_removed.append(idx)

    if len(lines) == 1:
        raise ValueError()

    for i in range(len(lines)-1, -1, -1):
        line = lines[i]
        wsr_line = remove_all_white_space(line)
        if wsr_line in ['}', '}}', '}}}', '};', '}}}}', '}}}}}', '{', '{{']:
            if i > 0:
                idx = i + 1
                for line_num in nline_removed:
                    if idx >= line_num:
                        idx += 1
                insert_sorted(nline_removed, idx)
                lines[i-1] += ' ' + line.strip()
                lines[i] = ''
            else:
                raise ValueError()
                return ''

    # Remove empty lines
    for i in range(len(lines)-1, -1, -1):
        if lines[i] == '':
            del lines[i]

    for line in lines:
        assert(lines[i].strip() != '')

    return '\n'.join(lines), nline_removed

def get_nline_removed(program_id):
    query='''SELECT program, problem_id FROM orgsource WHERE program_id=?;'''
    global cursor

    for row in cursor.execute(query, (program_id, )):
        program = row[0]
        # print(f"program:\n{program}")
        program, nline_removed = normalize_brackets(program)
        return nline_removed

def get_coverage(cov_file_path, nline_removed=[]):
    def process_line(line):
        tag, line_no, code = line.strip().split(':', 2)
        return tag.strip(), int(line_no.strip()), code
    coverage = {}
    with open(cov_file_path, "r") as f:
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
                coverage[line_no-sum(line_no > k for k in nline_removed)] = 0
            else:
                tag = int(tag)
                coverage[line_no-sum(line_no > k for k in nline_removed)] = 1
    return coverage

def get_covs(key):
    COVs = []
    LINEs = []
    src_b = key['b_fp']
    src_f = key['f_fp']
    pid = key['problem_id']
    vid = key['buggy']
    tests_list = list(test_verdict[pid][vid].keys())
    for i, test in enumerate(tests_list):
        covfile = f"{ConfigClass.nbl_test_path}/{pid}/{test}-{vid}.gcov"
        nline_removed = get_nline_removed(vid)
        cov_map = get_coverage(covfile, nline_removed)
        line_in_cov = [k for k, v in cov_map.items() if v == 1]
        # line_not_in_cov = list(cov_map.keys() - line_in_cov)
        verdict = True if test_verdict[pid][vid][test] == 1 else False
        COVs.append((line_in_cov, verdict))
        LINEs.extend(list(cov_map.keys()))
    return COVs, LINEs

def dstar(COVs, LINEs, dstar_param):
    # init variable Ncf, Nuf, Ncs for dstar
    ins_vars = dict()
    for line in LINEs:
        v = {"Ncf" : 0, "Nuf" : 0, "Ncs" : 0, "Nus" : 0}
        ins_vars[line] = v

    for lines, verdict in COVs:
        # count coverage values
        for line in lines:
            if verdict:
                ins_vars[line]["Ncs"] += 1
            else:
                ins_vars[line]["Ncf"] += 1

        # count uncoverage values
        # print list(set(BBLs) - set(bbls))
        for line in list(set(LINEs) - set(lines)):
            if verdict:
                ins_vars[line]["Nus"] += 1
            else:
                ins_vars[line]["Nuf"] += 1

    # calculate score
    scores = {}
    for ins_var in ins_vars.items():
        ins_addr = ins_var[0]
        ins_v = ins_var[1]
        try:
            score = float(ins_v["Ncf"] * dstar_param) / (ins_v["Ncs"] + ins_v["Nuf"])
        except ZeroDivisionError:
            score = 99999999
        if score != 0:
            scores[ins_addr] = score

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_scores)
    return [x for x, y in sorted_scores]

def main():
    fixed_eval_set = {}
    for info, bug_lines in eval_set.items():
        program_id = info.split('-')[-1]
        fixed_eval_set[int(program_id)] = bug_lines
    eval_keys = []
    for i, key in enumerate(all_keys):
        if int(key['buggy']) in fixed_eval_set:
            key['bug_lines'] = fixed_eval_set[int(key['buggy'])]
            eval_keys.append(key)
    # print(f"len(eval_keys) = {len(eval_keys)}")
    dstar_param = 2
    bar = tqdm.tqdm(list(eval_keys))
    bar.set_description("Calculate d2 score (NBL)")
    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_3_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()
    dstar_err = 0
    for i, key in enumerate(bar):
        COVs, LINEs = get_covs(key)
        sorted_sus_lines = dstar(COVs, LINEs, dstar_param)
        try:
            top_10_meter.update(int(any([idx in key['bug_lines']
                                        for idx in sorted_sus_lines[:10]])), 1)
            top_5_meter.update(int(any([idx in key['bug_lines']
                                        for idx in sorted_sus_lines[:5]])), 1)
            top_3_meter.update(int(any([idx in key['bug_lines']
                                        for idx in sorted_sus_lines[:3]])), 1)
            top_2_meter.update(int(any([idx in key['bug_lines']
                                        for idx in sorted_sus_lines[:2]])), 1)
            top_1_meter.update(int(sorted_sus_lines[0] in key['bug_lines']), 1)
        except IndexError:
            dstar_err += 1
            pass
    result = {}
    result['top_1'] = top_1_meter.avg
    result['top_2'] = top_2_meter.avg
    result['top_3'] = top_3_meter.avg
    result['top_5'] = top_5_meter.avg
    result['top_10'] = top_10_meter.avg
    print(result)
    print(f"error_instance: {dstar_err}/{len(eval_keys)}")

main()
