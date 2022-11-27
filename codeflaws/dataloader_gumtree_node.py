from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
from codeflaws.data_utils import get_cfg_ast_cov, all_codeflaws_keys
from codeflaws.data_utils import get_nx_ast_node_annt_gumtree
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from graph_algos.nx_shortcuts import nodes_where
import os
import random
import pickle as pkl
import json
import fasttext
import torch
import tqdm
from pycparser.plyparser import ParseError

embedding_model = fasttext.load_model(ConfigClass.pretrained_fastext)

class CodeflawsGumtreeNxNodeDataset(AstNxDataset):
    def __init__(self,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        super().__init__(
                all_entries=all_codeflaws_keys,
                process_func=get_nx_ast_node_annt_gumtree,
                save_dir=save_dir,
                name='gumtree_node',
                special_attrs=[]
        self.cfg_etypes = ['parent_child', 'next', 'ref', 'func_call']


class CodeflawsGumtreeDGLNodeDataset(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.graph_save_path = os.path.join(
            save_dir, 'dgl_nx_new_gumtree_node.bin')
        self.info_path = os.path.join(
            save_dir, 'dgl_nx_new_gumtree_node_info.pkl')
        self.nx_dataset = CodeflawsGumtreeNxNodeDataset(raw_dir, save_dir)
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(CodeflawsGumtreeDGLNodeDataset, self).__init__(
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
        self.cfg_content_dim = info_dict['cfg_content_dim']
        self.ast_content_dim = info_dict['ast_content_dim']
        self.master_idxs = info_dict['master_idxs']
        self.train_idxs = info_dict['train_idxs']
        self.val_idxs = info_dict['val_idxs']
        self.test_idxs = info_dict['test_idxs']
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
        pkl.dump({'cfg_content_dim': self.cfg_content_dim,
                  'ast_content_dim': self.ast_content_dim,
                  'master_idxs': self.master_idxs,
                  'train_idxs': self.train_idxs,
                  'val_idxs': self.val_idxs,
                  'test_idxs': self.test_idxs
                  },
                 open(self.info_path, 'wb'))

    def convert_from_nx_to_dgl(self, embedding_model, nx_g):
        '''
        # Create a node mapping for cfg
        n_cfgs = nodes_where(nx_g, graph='cfg')
        cfg2id = dict([n, i] for i, n in enumerate(n_cfgs))
        '''
        # Create a node mapping for ast
        n_asts = nodes_where(nx_g, graph='ast')
        ast2id = dict([n, i] for i, n in enumerate(n_asts))
        # Create a node mapping for test
        n_tests = nodes_where(nx_g, graph='test')
        t2id = dict([n, i] for i, n in enumerate(n_tests))
        # map2id = {'cfg': cfg2id, 'ast': ast2id, 'test': t2id}
        map2id = {'ast': ast2id, 'test': t2id}

        # Create dgl cfg node
        '''
        cfg_labels = torch.tensor(
            [ConfigClass.cfg_label_corpus.index(nx_g.nodes[n]['ntype'])
             for n in n_cfgs], dtype=torch.long)
        cfg_contents = torch.stack([
            torch.from_numpy(embedding_model.get_sentence_vector(
                nx_g.nodes[n]['text'].replace('\n', '')))
            for n in n_cfgs], dim=0)
        '''
        # Create dgl ast node
        ast_labels = torch.tensor([
            self.nx_dataset.ast_types.index(nx_g.nodes[node]['ntype'])
            for node in n_asts], dtype=torch.long
        )

        ast_contents = torch.stack([
            torch.from_numpy(embedding_model.get_sentence_vector(
                nx_g.nodes[n]['token'].replace('\n', '')))
            for n in n_asts], dim=0)
        # Create dgl test node
        # No need, will be added automatically when we update edges

        all_canon_etypes = {}
        for k in self.meta_graph:
            all_canon_etypes[k] = []
        '''
        line2cfg = {}
        for n in n_cfgs:
            if nx_g.nodes[n]['end_line'] - nx_g.nodes[n]['start_line'] > 0:
                continue
            if nx_g.nodes[n]['start_line'] not in line2cfg:
                line2cfg[nx_g.nodes[n]['start_line']] = [n]
            else:
                line2cfg[nx_g.nodes[n]['start_line']].append(n)
        '''
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
        '''
        g.nodes['cfg'].data['label'] = cfg_labels
        g.nodes['cfg'].data['content'] = cfg_contents
        '''
        g.nodes['ast'].data['label'] = ast_labels
        g.nodes['ast'].data['content'] = ast_contents
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        # g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        # tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        for node in nx_g.nodes():
            if nx_g.nodes[node]['graph'] != 'ast':
                continue
            ast_tgts[ast2id[node]] = nx_g.nodes[node]['status']
        '''
        for node in cfg_lb:
            tgts[cfg2id[node]] = 1

        g.nodes['cfg'].data['tgt'] = tgts
        '''
        g.nodes['ast'].data['tgt'] = ast_tgts
        return g

    def construct_edge_metagraph(self):
        self.ast_etypes = self.nx_dataset.ast_etypes + ['a_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.ast_etypes]
        # self.cfg_etypes = self.nx_dataset.cfg_etypes + ['c_self_loop'] + \
        #     [et + '_reverse' for et in self.nx_dataset.cfg_etypes]
        # self.c_a_etypes = ['corresponding_ast']
        # self.a_c_etypes = ['corresponding_cfg']
        # self.c_t_etypes = ['c_pass_test', 'c_fail_test']
        # self.t_c_etypes = ['t_pass_c', 't_fail_c']
        # self.a_t_etypes = ['a_pass_test', 'a_fail_test']
        self.t_a_etypes = ['t_pass_a', 't_fail_a']
        self.all_etypes = (
            self.ast_etypes +
            # self.cfg_etypes +
            # self.c_a_etypes + self.a_c_etypes +
            # self.c_t_etypes + self.t_c_etypes +
            # self.a_t_etypes +
            self.t_a_etypes)
        self.all_ntypes = (
            [('ast', 'ast') for et in self.ast_etypes] +
            # [('cfg', 'cfg') for et in self.cfg_etypes] +
            # [('cfg', 'ast') for et in self.c_a_etypes] +
            # [('ast', 'cfg') for et in self.a_c_etypes] +
            # [('cfg', 'test') for et in self.c_t_etypes] +
            # [('test', 'cfg') for et in self.t_c_etypes] +
            # [('ast', 'test') for et in self.a_t_etypes] +
            [('test', 'ast') for et in self.t_a_etypes]
        )

        return [(t[0], et, t[1]) for t, et in zip(self.all_ntypes,
                                                  self.all_etypes)]

    def process(self):
        self.meta_graph = self.construct_edge_metagraph()
        self.gs = []
        bar = tqdm.tqdm(self.nx_dataset)
        bar.set_description("Converting NX to DGL")
        for i, nx_g in enumerate(bar):
            g = self.convert_from_nx_to_dgl(embedding_model, nx_g)
            self.gs.append(g)
            if i == 0:
                # self.cfg_content_dim = g.nodes['cfg'].data['content'].shape[-1]
                self.cfg_content_dim = 1
                self.ast_content_dim = g.nodes['ast'].data['content'].shape[-1]

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]]
