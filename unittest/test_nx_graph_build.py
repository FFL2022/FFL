from utils import draw_utils
from utils.utils import ConfigClass
from utils.nx_graph_builder import build_nx_graph_cfg_ast, build_nx_cfg,\
    build_nx_ast, build_nx_cfg_ast_coverage_codeflaws
import pickle as pkl
import os


if __name__ == '__main__':
    os.makedirs('visualize_nx_graphs', exist_ok=True)
    train_map = pkl.load(open(
        ConfigClass.codeflaws_train_cfgidx_map_pkl, 'rb'))
    test_verdict = pkl.load(open(
        ConfigClass.codeflaws_test_verdict_pickle, 'rb'))
    key = list(train_map.keys())[0]
    info = key.split("-")
    codeflaws = {}
    codeflaws['container'] = key
    codeflaws['c_source'] = "{}-{}-{}".format(info[0], info[1], info[3])
    codeflaws['test_verdict'] = test_verdict["{}-{}".format(info[0],
                                                            info[1])][info[3]]

    cfg, ast, cfg_ast, cfg_ast_cov = build_nx_cfg_ast_coverage_codeflaws(codeflaws)
    draw_utils.cfg_ast_cov_to_agraph(cfg_ast_cov,
                                     'visualize_nx_graphs/cfg_ast_cov.png')
    draw_utils.cfg_ast_cov_to_agraph(cfg,
                                     'visualize_nx_graphs/cfg.png')
    draw_utils.cfg_ast_cov_to_agraph(ast,
                                     'visualize_nx_graphs/ast.png')
    draw_utils.cfg_ast_cov_to_agraph(cfg_ast,
                                     'visualize_nx_graphs/cfg_ast.png')
