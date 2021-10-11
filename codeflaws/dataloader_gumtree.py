from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
from codeflaws.data_utils import all_codeflaws_keys,\
    get_nx_ast_node_annt_gumtree,\
    get_nx_ast_stmt_annt_gumtree
from utils.gumtree_utils import GumtreeASTUtils
from utils.nx_graph_builder import augment_with_reverse_edge_cat
import os
import random
import pickle as pkl
import json
import torch
import tqdm

from json import JSONDecodeError


class CodeflawsGumtreeNxStatementDataset(object):
    def __init__(self, raw_dataset_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.save_dir = save_dir
        self.info_path = os.path.join(
            save_dir, 'nx_gumtree_stmt_dataset_info.pkl')
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()


    def len(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return pkl.load(open(
            f'{self.save_dir}/nx_gumtree_stmt_{self.active_idxs[i]}.pkl', 'rb')),\
            self.stmt_nodes[i]

    def process(self):
        self.ast_types = []
        self.ast_etypes = []
        self.stmt_nodes = []
        self.keys = []
        self.err_idxs = []
        self.active_idxs = []

        bar = tqdm.tqdm(list(enumerate(all_codeflaws_keys)))
        bar.set_description('Loading Nx Data with gumtree')
        for i, key in bar:

            try:
                if not os.path.exists(f'{self.save_dir}/nx_gumtree_stmt_{i}.pkl'):
                    nx_g = get_nx_ast_stmt_annt_gumtree(key)
                    pkl.dump(nx_g, open(
                        f'{self.save_dir}/nx_gumtree_stmt_{i}.pkl', 'wb')
                    )
                else:
                    nx_g = pkl.load(open(
                        f'{self.save_dir}/nx_gumtree_stmt_{i}.pkl', 'rb')
                    )
            except JSONDecodeError:
                self.err_idxs.append(i)
                count = len(self.err_idxs)
                print(f"Total syntax error files: {count}")
                continue
            self.active_idxs.append(i)
            self.keys.append(key)
            self.ast_types.extend(
                [nx_g.nodes[node]['ntype'] for node in nx_g.nodes()
                 if nx_g.nodes[node]['graph'] == 'ast'])
            self.ast_etypes.extend(
                [e['label'] for u, v, k, e in nx_g.edges(keys=True, data=True)
                 if nx_g.nodes[u]['graph'] == 'ast' and
                 nx_g.nodes[v]['graph'] == 'ast'])
            self.stmt_nodes.append(list(
                [n for n in nx_g.nodes() if
                 nx_g.nodes[n]['graph'] == 'ast'
                 and GumtreeASTUtils.check_is_stmt_cpp(nx_g.nodes[n]['ntype'])]
            ))

        self.ast_types = list(set(self.ast_types))
        self.ast_etypes = list(set(self.ast_etypes))

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # gs is saved somewhere else
        pkl.dump(
            {
                'ast_types': self.ast_types, 'ast_etypes': self.ast_etypes,
                'keys': self.keys, 'err_idxs': self.err_idxs,
                'active_idxs': self.active_idxs
            },
            open(self.info_path, 'wb'))

    def load(self):
        info_dict = pkl.load(open(self.info_dict, 'rb'))
        self.ast_types = info_dict['ast_types']
        self.ast_etypes = info_dict['ast_etypes']
        self.keys = info_dict['keys']
        self.err_idxs = info_dict['err_idxs']
        self.active_idxs = info_dict['active_idxs']

    def has_cache(self):
        return os.path.exists(self.info_path)


class CodeflawsGumtreeNxNodeDataset(object):
    def __init__(self, raw_dataset_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.save_dir = save_dir
        self.info_path = os.path.join(
            save_dir, 'nx_gumtree_node_dataset_info.pkl')
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

        self.active_idxs = list(range(len(self.ast_lbs)))

    def len(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return pkl.load(open(
            f'{self.save_dir}/nx_gumtree_node_{self.active_idxs[i]}', 'rb')),\
            self.stmt_nodes[self.active_idxs[i]]

    def process(self):
        self.ast_types = []
        self.ast_etypes = []
        self.stmt_nodes = []
        self.keys = []
        self.err_idxs = []
        bar = tqdm.tqdm(list(enumerate(all_codeflaws_keys)))
        bar.set_description('Loading Nx Data Node level with gumtree')
        for i, key in bar:
            try:
                nx_g = get_nx_ast_stmt_annt_gumtree(key)
                pkl.dump(nx_g, open(
                    f'{self.save_dir}/nx_gumtree_node_{i}.pkl', 'wb')
                )
            except JSONDecodeError:
                self.err_idxs.append(i)
                count = len(self.err_idxs)
                print(f"Total syntax error files: {count}")
                continue
            self.keys.append(key)
            self.ast_types.extend(
                [nx_g.nodes[node]['ntype'] for node in nx_g.nodes()
                 if nx_g.nodes[node]['graph'] == 'ast'])
            self.ast_etypes.extend(
                [e['label'] for u, v, k, e in nx_g.edges(keys=True, data=True)
                 if nx_g.nodes[u]['graph'] == 'ast' and
                 nx_g.nodes[v]['graph'] == 'ast'])
            self.stmt_nodes.append(list(
                [n for n in nx_g.nodes() if
                 nx_g.nodes[n]['graph'] == 'ast'
                 and GumtreeASTUtils.check_is_stmt_cpp(nx_g.nodes[n]['ntype'])]
            ))

        self.ast_types = list(set(self.ast_types))
        self.ast_etypes = list(set(self.ast_etypes))

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # gs is saved somewhere else
        pkl.dump(
            {
                'ast_types': self.ast_types, 'ast_etypes': self.ast_etypes,
                'keys': self.keys, 'err_idxs': self.err_idxs
            },
            open(self.info_path, 'wb'))

    def load(self):
        info_dict = pkl.load(open(self.info_dict, 'rb'))
        self.ast_types = info_dict['ast_types']
        self.ast_etypes = info_dict['ast_etypes']
        self.keys = info_dict['keys']
        self.err_idxs = info_dict['err_idxs']

    def has_cache(self):
        return os.path.exists(self.info_path)


class CodeflawsGumtreeDGLStatementDataset(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.graph_save_path = os.path.join(
            save_dir, 'dgl_nx_gumtree_stmt.bin')
        self.info_path = os.path.join(
            save_dir, 'dgl_nx_gumtree_stmt_info.pkl')
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
        self.ast_content_dim = info_dict['ast_content_dim']
        self.master_idxs = info_dict['master_idxs']
        self.train_idxs = info_dict['train_idxs']
        self.val_idxs = info_dict['val_idxs']
        self.test_idxs = info_dict['test_idxs']
        self.stmt_idxs = info_dict['stmt_idxs']
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
            'ast_content_dim': self.ast_content_dim,
            'master_idxs': self.master_idxs,
            'train_idxs': self.train_idxs,
            'val_idxs': self.val_idxs,
            'test_idxs': self.test_idxs,
            'stmt_idxs': self.stmt_idxs
            },
            open(self.info_path, 'wb')
        )

    def convert_from_nx_to_dgl(self, nx_g, stmt_nodes):
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
            self.nx_dataset.ast_types.index(nx_g.nodes[node]['ntype'])
            for node in n_asts], dtype=torch.long
        )

        # Create dgl test node
        # No need, will be added automatically when we update edges

        all_canon_etypes = {}
        for k in self.meta_graph:
            all_canon_etypes[k] = []
        nx_g = augment_with_reverse_edge_cat(nx_g, self.nx_dataset.ast_etypes,
                                             self.nx_dataset.cfg_etypes)

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
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        # g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        # tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        for node in ast2id:
            ast_tgts[ast2id[node]] = 1
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
        bar = tqdm.tqdm(enumerate(self.nx_dataset))
        bar.set_description("Converting NX to DGL")
        stmt_idxs = []
        for i, (nx_g, stmt_nodes) in bar:
            g, stmt_idx = self.convert_from_nx_to_dgl(nx_g, stmt_nodes)
            stmt_idxs.append(torch.tensor(stmt_idxs).long())
            self.gs.append(g)
            if i == 0:
                self.ast_content_dim = g.nodes['ast'].data['content'].shape[-1]

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]], \
            self.stmt_idxs[self.active_idxs[i]]
