from utils.codeflaws_data_utils import all_codeflaws_keys, get_cfg_ast_cov
from utils import draw_utils


if __name__ == '__main__':
    # for i in list(range(10)) + [28] + [715]:
        nx_ast, nx_ast_f, nx_cfg, nx_cfg_f, nx_cfg_ast, nx_cfg_ast_cov\
            = get_cfg_ast_cov(all_codeflaws_keys[i])
        draw_utils.ast_to_agraph(nx_ast,
                                 f"visualize_nx_graphs/ast_diff_{i}.png")
        draw_utils.ast_to_agraph(nx_ast_f,
                                 f"visualize_nx_graphs/ast_fixx_{i}.png")
        '''
        draw_utils.cfg_ast_to_agraph(nx_cfg_ast,
                                     f"visualize_nx_graphs/cfg_ast_diff_{i}.png")
                                     '''
