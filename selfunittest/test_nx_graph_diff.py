from codeflaws.data_utils import all_codeflaws_keys, get_cfg_ast_cov, \
    get_nx_ast_stmt_annt_gumtree
from utils import draw_utils
import os

def test1():
    for i in list(range(10)) + [28] + [715]:
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
def test2():
    for i, key in enumerate(all_codeflaws_keys):
        try:
            nx_g = get_nx_ast_stmt_annt_gumtree(key)
            os.makedirs('visualize_nx_graphs', exist_ok=True)
            draw_utils.ast_to_agraph(nx_g.subgraph([n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'ast']),
                                    f"visualize_nx_graphs/ast_gumtree_diff_{i}.png")
        except:
            continue
if __name__ == '__main__':
    test2()
