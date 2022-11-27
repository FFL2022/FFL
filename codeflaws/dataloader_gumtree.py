from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
from codeflaws.data_utils import all_codeflaws_keys,\
    get_nx_ast_node_annt_gumtree,\
    get_nx_ast_stmt_annt_gumtree
from utils.gumtree_utils import GumtreeASTUtils
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from utils.data_utils import AstNxDataset
from graph_algos.nx_shortcuts import nodes_where, edges_where,\
        where_node, where_node_not
from collections import defaultdict
import os
import random
import pickle as pkl
import json
import torch
import tqdm

from json import JSONDecodeError
from utils.data_utils import NxDataset

class CodeflawsGumtreeNxStatementDataset(AstNxDataset):
    def __init__(self, save_dir=ConfigClass.preprocess_dir_codeflaws): super().__init__(
                all_entries=all_codeflaws_keys,
                process_func=get_nx_ast_stmt_annt_gumtree,
                save_dir=save_dir,
                name='gumtree_stmt',
                special_attrs=[('stmt_nodes', 
                    lambda nx_g: nodes_where(
            nx_g, lambda x: GumtreeASTUtils.check_is_stmt_cpp(nx_g.nodes[x]['ntype'], graph='ast')
            ))]


class CodeflawsGumtreeNxNodeDataset(AstNxDataset):
    def __init__(self,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        super().__init__(
                all_entries=all_codeflaws_keys,
                process_func=get_nx_ast_node_annt_gumtree,
                save_dir=save_dir,
                name='gumtree_node',
                special_attrs=[]


class CodeflawsGumtreeDGLStatementDataset(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.graph_save_path = os.path.join(
            save_dir, 'dgl_nx_new_gumtree_stmt.bin') self.info_path = os.path.join(
            save_dir, 'dgl_nx_new_gumtree_stmt_info.pkl')
        self.nx_dataset = CodeflawsGumtreeNxStatementDataset(raw_dir, save_dir)
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(CodeflawsGumtreeDGLStatementDataset, self).__init__(
            name='codeflaws_dgl',
            url=None,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=False,
            verbose=False)

    def train(self):
        self.active_idxs = self.train_idxs

    def val(self):
        self.active_idxs = self.val_idxs

    def test(self):
        self.active_idxs = self.test_idxs

    def has_cache(self):
        return os.path.exists(self.graph_save_path) and\
            os.path.exists(self.info_path)

    def load(self):
        self.gs = load_graphs(self.graph_save_path)[0]
        self.meta_graph = self.construct_edge_metagraph()
        info_dict = pkl.load(open(self.info_path, 'rb'))
        for k in info_dict:
            setattr(self, k, info_dict[k])
        self.train()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_graphs(self.graph_save_path, self.gs)
        self.master_idxs = list(range(len(self.gs)))
        random.shuffle(self.master_idxs)
        self.train_idxs = list(self.master_idxs[:int(len(self.gs)*0.8)])
        self.val_idxs = list(self.master_idxs[
            int(len(self.gs)*0.8):int(len(self.gs)*0.9)])
        self.test_idxs = list(self.master_idxs[
            int(len(self.gs)*0.9):int(len(self.gs))])
        pkl.dump({
            'master_idxs': self.master_idxs,
            'train_idxs': self.train_idxs,
            'val_idxs': self.val_idxs,
            'test_idxs': self.test_idxs,
            'stmt_idxs': self.stmt_idxs
            },
            open(self.info_path, 'wb')
        )

    def convert_from_nx_to_dgl(self, nx_g, stmt_nodes):
        n_asts = nodes_where(nx_g, graph='ast')
        ast2id = dict([n, i] for i, n in enumerate(n_asts))
        n_tests = nodes_where(nx_g, graph='test')
        t2id = dict([n, i] for i, n in enumerate(n_tests))
        map2id = {'ast': ast2id, 'test': t2id}

        # Create dgl ast node
        ast_labels = torch.tensor([
            self.nx_dataset.ast_types.index(nx_g.nodes[node]['ntype'])
            for node in n_asts], dtype=torch.long
        )

        # Create dgl test node
        # No need, will be added automatically when we update edges

        all_canon_etypes = defaultdict(list)
        nx_g = augment_with_reverse_edge_cat(nx_g, self.nx_dataset.ast_etypes,
                                             [])
        for u, v, k, e in edges_where(nx_g, where_node_not(graph='cfg'), where_node_not(graph='cfg')):
            map_u = map2id[nx_g.nodes[u]['graph']]
            map_v = map2id[nx_g.nodes[v]['graph']]
            all_canon_etypes[
                (nx_g.nodes[u]['graph'], e['label'], nx_g.nodes[v]['graph'])
            ].append([map_u[u], map_v[v]])

        for k in all_canon_etypes:
            type_es = torch.tensor(all_canon_etypes[k], dtype=torch.int32)
            all_canon_etypes[k] = (type_es[:, 0], type_es[:, 1])

        g = dgl.heterograph(all_canon_etypes)
        g.nodes['ast'].data['label'] = ast_labels
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        # g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        # tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        for node in ast2id:
            ast_tgts[ast2id[node]] = nx_g.nodes[node]['status']
        g.nodes['ast'].data['tgt'] = ast_tgts
        stmt_idxs = [ast2id[n] for n in stmt_nodes]
        return g, stmt_idxs

    def construct_edge_metagraph(self):
        self.ast_etypes = self.nx_dataset.ast_etypes + ['a_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.ast_etypes]
        self.a_t_etypes = ['a_pass_test', 'a_fail_test']
        self.t_a_etypes = ['t_pass_a', 't_fail_a']
        self.all_etypes = (
            self.ast_etypes +
            self.a_t_etypes +
            self.t_a_etypes)
        self.all_ntypes = (
            [('ast', 'ast') for et in self.ast_etypes] +
            [('ast', 'test') for et in self.a_t_etypes] +
            [('test', 'ast') for et in self.t_a_etypes]
        )
        return [(t[0], et, t[1]) for t, et in zip(self.all_ntypes,
                                                  self.all_etypes)]

    def process(self):
        self.meta_graph = self.construct_edge_metagraph()
        self.gs = []
        bar = tqdm.tqdm(self.nx_dataset)
        bar.set_description("Converting NX to DGL")
        self.stmt_idxs = []
        for i, (nx_g, stmt_nodes) in enumerate(bar):
            g, stmt_idx = self.convert_from_nx_to_dgl(nx_g, stmt_nodes)
            self.stmt_idxs.append(torch.tensor(stmt_idx).long())
            self.gs.append(g)

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]], \
            self.stmt_idxs[self.active_idxs[i]]
