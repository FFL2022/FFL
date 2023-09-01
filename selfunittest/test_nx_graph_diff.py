# from codeflaws.data_utils import all_codeflaws_keys, get_cfg_ast_cov, \
#     get_nx_ast_stmt_annt_gumtree, get_nx_ast_stmt_annt_cfl, \
#     get_nx_ast_node_annt_gumtree
from utils import draw_utils
import tqdm
import os
from pycparser.plyparser import ParseError
from nbl.utils import all_keys, get_nx_ast_stmt_annt_gumtree
from json import JSONDecodeError
from graph_algos.nx_shortcuts import nodes_where
from codeflaws.dataloader_key_only import get_cfg_ast_cov, all_codeflaws_keys
from codeflaws.data_utils import get_nx_ast_stmt_annt_cfl

def test1():
    bar = tqdm.tqdm(list(range(50)) + [715])
    for i in bar:
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
    bar = tqdm.tqdm(all_codeflaws_keys[:50])
    count = 0
    for i, key in enumerate(bar):
        if key == '114-A-bug-17914312-17914321':
            try:
                filename = f"visualize_nx_graphs/cfl/{key}.png"
                if os.path.exists(filename):
                    continue
                nx_g = get_nx_ast_stmt_annt_cfl(key)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                draw_utils.ast_to_agraph(nx_g.subgraph(nodes_where(nx_g, graph='ast')), filename)
            except ParseError:
                count += 1
                bar.set_postfix(parse_error_files=count)
                continue
            except OSError:
                print(key)
                continue
            except:
                print(key)
                raise
        # break

def test3():
    bar = tqdm.tqdm(all_keys[:5])
    for i, key in enumerate(bar):
        try:
            filename = "visualize_nx_graphs_new/nbl/{}.png".format(os.path.basename(key['b_fp']))
            print(filename)
            nx_g = get_nx_ast_stmt_annt_gumtree(key)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            draw_utils.ast_to_agraph(nx_g.subgraph(nodes_where(nx_g, graph='ast')),
                    filename)
        except JSONDecodeError:
            print('error:', key)
            raise

        break



if __name__ == '__main__':
    test1()
    test2()
    test3()
