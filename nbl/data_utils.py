from nbl.data_format import test_verdict
from utils.utils import ConfigClass from utils.preprocess_helpers import get_coverage, remove_lib
from utils.pyg_utils import get_coverage_graph_ast_pyg, \
        get_nx_ast_stmt_annt_pyg


def get_nx_ast_stmt_annt_cfl_nbl(key):
    src_b, src_f = key['b_fp'], key['f_fp']
    pid = key['problem_id']
    vid = key['buggy']
    test_list = list(test_verdict[pid][vid].keys())
    cov_maps, verdicts = [], []
    for test in test_list:
        verdicts.append(test_verdict[pid][vid][test])
        covfile = f"{ConfigClass.nbl_test_path}/{pid}/{test}-{vid}.gcov"
        cov_map = get_coverage(covfile, nline_removed1)
        cov_maps.append(cov_maps)

    return get_coverage_graph_ast_pyg(get_nx_ast_stmt_annt_pyg(src_b, src_f), cov_maps, verdicts)
