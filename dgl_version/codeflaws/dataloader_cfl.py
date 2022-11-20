from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
import os
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset, \
    ASTMetadata
from utils.nx_graph_builder import augment_with_reverse_edge_cat
import pickle as pkl
import random
import torch
import tqdm


class CodeflawsCFLDGLStatementDataset(DGLDataset):
    def __init__(self, nx_dataset: CodeflawsCFLNxStatementDataset,
                 meta_data: ASTMetadata,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.graph_save_path = os.path.join(
            save_dir, 'dgl_nx_cfl_stmt.bin')
        self.info_path = os.path.join(
            save_dir, 'dgl_nx_cfl_stmt_info.pkl')
        self.nx_dataset = nx_dataset
        self.meta_data = meta_data
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(CodeflawsCFLDGLStatementDataset, self).__init__(
            name='codeflaws_dgl',
            url=None,
            raw_dir=".",
            save_dir=save_dir,
            force_reload=False,
            verbose=False)

        self.train()

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
        self.meta_graph = self.meta_data.meta_graph
        info_dict = pkl.load(open(self.info_path, 'rb'))
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
            'master_idxs': self.master_idxs, 'train_idxs': self.train_idxs,
            'val_idxs': self.val_idxs, 'test_idxs': self.test_idxs,
            'stmt_idxs': self.stmt_idxs
            },
            open(self.info_path, 'wb')
        )

    def convert_from_nx_to_dgl(self, nx_g, stmt_nodes):
        # Create a node mapping for ast
        n_asts = list(filter(lambda x: nx_g.nodes[x]['graph'] == 'ast',
                             nx_g.nodes()))
        ast2id = {n: i for i, n in enumerate(n_asts)}
        # Create a node mapping for test
        n_tests = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'test']
        t2id = {n: i for i, n in enumerate(n_tests)}
        # map2id = {'oncfg': cfg2id, 'ast': ast2id, 'test': t2id}
        n2id = {'ast': ast2id, 'test': t2id}

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
                                             [])

        for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
            if nx_g.nodes[u]['graph'] == 'cfg' or\
                    nx_g.nodes[v]['graph'] == 'cfg':
                continue
            map_u = n2id[nx_g.nodes[u]['graph']]
            map_v = n2id[nx_g.nodes[v]['graph']]
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

    def process(self):
        self.meta_graph = self.meta_data.meta_graph
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
