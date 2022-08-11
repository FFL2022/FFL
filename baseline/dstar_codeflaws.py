#!/usr/bin/python

from codeflaws.data_format import key2test_verdict, key2bugfile, get_gcov_file
from codeflaws.data_utils import all_codeflaws_keys
from utils.utils import ConfigClass
from utils.preprocess_helpers import remove_lib, get_coverage
from utils.train_utils import AverageMeter
import tqdm, sqlite3, operator, pickle

def get_covs(key):
    COVs = []
    LINEs = []
    tests_list = key2test_verdict(key)
    for i, test in enumerate(tests_list):
        covfile = get_gcov_file(key, test)
        nline_removed = remove_lib(key2bugfile(key))
        cov_map = get_coverage(covfile, nline_removed)
        line_in_cov = [k for k, v in cov_map.items() if v == 1]
        # line_not_in_cov = list(cov_map.keys() - line_in_cov)
        verdict = True if tests_list[test] > 0 else False
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
    eval_data = pickle.load(open(ConfigClass.codeflaws_eval_cfgidx_map_pkl, 'rb'))
    eval_keys = []
    for key in all_codeflaws_keys:
        if key in eval_data:
            eval_keys.append((key, eval_data[key]))
    # print(f"len(eval_keys) = {len(eval_keys)}")
    dstar_param = 2
    bar = tqdm.tqdm(list(eval_keys))
    bar.set_description(f"Calculate d{dstar_param} score (Codeflaws)")
    top_1_meter = AverageMeter()
    top_2_meter = AverageMeter()
    top_3_meter = AverageMeter()
    top_5_meter = AverageMeter()
    top_10_meter = AverageMeter()
    dstar_err = 0
    for i, (key, bug_lines) in enumerate(bar):
        COVs, LINEs = get_covs(key)
        sorted_sus_lines = dstar(COVs, LINEs, dstar_param)
        try:
            top_10_meter.update(int(any([idx in bug_lines
                                        for idx in sorted_sus_lines[:10]])), 1)
            top_5_meter.update(int(any([idx in bug_lines
                                        for idx in sorted_sus_lines[:5]])), 1)
            top_3_meter.update(int(any([idx in bug_lines
                                        for idx in sorted_sus_lines[:3]])), 1)
            top_2_meter.update(int(any([idx in bug_lines
                                        for idx in sorted_sus_lines[:2]])), 1)
            top_1_meter.update(int(sorted_sus_lines[0] in bug_lines), 1)
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
