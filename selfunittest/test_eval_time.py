from utils import draw_utils
from pycparser.plyparser import ParseError
from nbl.utils import all_keys, get_nx_ast_stmt_annt_gumtree
from json import JSONDecodeError
from utils.gumtree_utils import GumtreeASTUtils
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
from utils.nx_graph_builder import augment_with_reverse_edge_cat
import os
import random
import pickle as pkl
import torch
import tqdm
from model import GCN_A_L_T_1
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def construct_edge_metagraph(nx_ast_etypes):
    ast_etypes = nx_ast_etypes + ['a_self_loop'] + \
        [et + '_reverse' for et in nx_ast_etypes]
    a_t_etypes = ['a_pass_test', 'a_fail_test']
    t_a_etypes = ['t_pass_a', 't_fail_a']
    all_etypes = (
        ast_etypes +
        a_t_etypes +
        t_a_etypes)
    all_ntypes = (
        [('ast', 'ast') for et in ast_etypes] +
        [('ast', 'test') for et in a_t_etypes] +
        [('test', 'ast') for et in t_a_etypes]
    )
    return [(t[0], et, t[1]) for t, et in zip(all_ntypes,
                                              all_etypes)]
def eval():

    bar = tqdm.tqdm(all_keys)

    info_path = os.path.join(
            ConfigClass.preprocess_dir_nbl, 'nx_nbl_gumtree_stmt_dataset_info.pkl')
    info_dict = pkl.load(open(info_path, 'rb'))
    nx_ast_types = info_dict['ast_types']
    nx_ast_etypes = info_dict['ast_etypes']

    meta_graph = construct_edge_metagraph(nx_ast_etypes)
    model = GCN_A_L_T_1(
        128, meta_graph,
        device=device,
        num_ast_labels=len(nx_ast_types),
        num_classes_ast=2)
    model.eval()

    time_list = []

    for key in bar:
        try:
            tic = time.time()
            
            # nx
            nx_g = get_nx_ast_stmt_annt_gumtree(key)

            # dgl
            # Create a node mapping for ast
            n_asts = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'ast']
            ast2id = dict([n, i] for i, n in enumerate(n_asts))
            # Create a node mapping for test
            n_tests = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'test']
            t2id = dict([n, i] for i, n in enumerate(n_tests))
            # map2id = {'cfg': cfg2id, 'ast': ast2id, 'test': t2id}
            map2id = {'ast': ast2id, 'test': t2id}

            # Create dgl ast node
            ast_labels = torch.tensor([
                nx_ast_types.index(nx_g.nodes[node]['ntype'])
                for node in n_asts], dtype=torch.long
            )

            all_canon_etypes = {}
            for k in meta_graph:
                all_canon_etypes[k] = []
            nx_g = augment_with_reverse_edge_cat(nx_g, nx_ast_etypes, [])

            for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
                if nx_g.nodes[u]['graph'] == 'cfg' or nx_g.nodes[v]['graph'] == 'cfg':
                    continue
                map_u = map2id[nx_g.nodes[u]['graph']]
                map_v = map2id[nx_g.nodes[v]['graph']]
                all_canon_etypes[
                    (nx_g.nodes[u]['graph'], e['label'], nx_g.nodes[v]['graph'])
                ].append([map_u[u], map_v[v]])

            for k in all_canon_etypes:
                if len(all_canon_etypes[k]) > 0:
                    type_es = torch.tensor(all_canon_etypes[k], dtype=torch.int32)
                    all_canon_etypes[k] = (type_es[:, 0], type_es[:, 1])
                else:
                    all_canon_etypes[k] = (torch.tensor([], dtype=torch.int32),
                                           torch.tensor([], dtype=torch.int32))

            g = dgl.heterograph(all_canon_etypes)
            g.nodes['ast'].data['label'] = ast_labels
            g.nodes['ast'].data['label'] = ast_labels
            g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
            # g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
            # tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
            ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
            for node in ast2id:
                ast_tgts[ast2id[node]] = nx_g.nodes[node]['status']
            g.nodes['ast'].data['tgt'] = ast_tgts
            g = g.to(device)
            model(g)

            time_list.append(time.time() - tic)
        except JSONDecodeError:
            print('error:', key)
            continue
    print(time_list)
    time_list = np.array(time_list)
    print('mean:', np.mean(time_list))
    print('std:', np.std(time_list))
    print('min:', np.min(time_list))
    print('max:', np.max(time_list))


if __name__ == '__main__':
    eval()