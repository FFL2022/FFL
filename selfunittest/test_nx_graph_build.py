from utils import draw_utils
from utils.utils import ConfigClass
from utils.nx_graph_builder import build_nx_graph_cfg_ast, build_nx_cfg,\
    build_nx_ast

from codeflaws.data_utils import build_nx_cfg_coverage_codeflaws, build_nx_cfg_ast_coverage_codeflaws
import pickle as pkl
import os
import time


if __name__ == '__main__':
    os.makedirs('visualize_nx_graphs', exist_ok=True)
    train_map = pkl.load(open(
        ConfigClass.codeflaws_train_cfgidx_map_pkl, 'rb'))
    test_verdict = pkl.load(open(
        ConfigClass.codeflaws_test_verdict_pickle, 'rb'))
    key = list(train_map.keys())[0]
    start_time = time.time()
    cfg, ast, cfg_ast, cfg_ast_cov = build_nx_cfg_ast_coverage_codeflaws(
        key)
    end_time = time.time()
    cfg, ast, cfg_ast, cfg_cov = build_nx_cfg_coverage_codeflaws(
        key)
    end_time2 = time.time()
    print("Build CFG AST COV took: {}".format(end_time - start_time))
    print("Build CFG COV took: {}".format(end_time2 - end_time))
    draw_utils.cfg_to_agraph(cfg, 'visualize_nx_graphs/cfg.png')
    draw_utils.ast_to_agraph(ast, 'visualize_nx_graphs/ast.png')
    draw_utils.cfg_ast_to_agraph(cfg_ast, 'visualize_nx_graphs/cfg_ast.png')
