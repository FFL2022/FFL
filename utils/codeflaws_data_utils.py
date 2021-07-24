import os
from utils.utils import ConfigClass
from utils.codeflaws_data_format import key2bug, key2fix, key2bugfile,\
    key2fixfile
from utils.get_bug_localization import get_bug_localization
from utils.nx_graph_builder import get_coverage_graph_cfg_ast
import pickle as pkl

root = ConfigClass.codeflaws_data_path


def make_codeflaws_dict(key, test_verdict):
    info = key.split("-")
    codeflaws = {}
    codeflaws['container'] = key
    codeflaws['c_source'] = key2bug(key) + ".c"
    codeflaws['test_verdict'] = test_verdict["{}-{}".format(
        info[0], info[1])][info[3]]
    return codeflaws


def get_all_keys():
    data = {}
    ''' Example line:
    71-A-bug-18359456-18359477	DCCR	WRONG_ANSWER	DIFFOUT~~Loop~If~Printf
    Diname=key                      ..?
    '''

    with open(f"{root}/codeflaws-defect-detail-info.txt", "rb") as f:
        for line in f:
            info = line.split()
            try:
                data[info[1].decode("utf-8")].append(info[0].decode("utf-8"))
            except:
                data[info[1].decode("utf-8")] = []

    # diname -> graph
    all_keys = []
    graph_data_map = {}
    for _, keys in data.items():
        for key in keys:
            if not os.path.isdir("{}/{}".format(root, key)):
                continue
            all_keys.append(key)
    return all_keys


def get_cfg_ast_cov(key):
    nx_ast, nx_ast2, nx_cfg, nx_cfg2, nx_cfg_ast, nline_removed1 =\
        get_bug_localization(key2bugfile(key),
                             key2fixfile(key))
    nx_cfg_ast_cov = get_coverage_graph_cfg_ast(key, nx_cfg_ast,
                                                nline_removed1)
    return nx_ast, nx_ast2, nx_cfg, nx_cfg2, nx_cfg_ast, nx_cfg_ast_cov


if os.path.exists(ConfigClass.codeflaws_all_keys):
    all_codeflaws_keys = pkl.load(open(ConfigClass.codeflaws_all_keys, 'rb'))
else:
    all_codeflaws_keys = get_all_keys()
    pkl.dump(all_codeflaws_keys, open(ConfigClass.codeflaws_all_keys, 'wb'))
