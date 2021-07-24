from utils.codeflaws_data_utils import all_codeflaws_keys, get_cfg_ast_cov
from utils import draw_utils


if __name__ == '__main__':
    nx_ast, nx_ast_f, nx_cfg, nx_cfg_f, nx_cfg_ast, nx_cfg_ast_cov\
        = get_cfg_ast_cov(all_codeflaws_keys[0])
    draw_utils.ast_to_agraph(nx_ast, "visualize_nx_graphs/ast_diff.png")
    draw_utils.ast_to_agraph(nx_ast_f, "visualize_nx_graphs/ast_fixx.png")
    draw_utils.cfg_to_agraph(nx_cfg, "visualize_nx_graphs/cfg_diff.png")
    draw_utils.cfg_to_agraph(nx_cfg_f, "visualize_nx_graphs/cfg_fixx.png")
    draw_utils.cfg_ast_to_agraph(nx_cfg_ast, "visualize_nx_graphs/cfg_ast_diff.png")
