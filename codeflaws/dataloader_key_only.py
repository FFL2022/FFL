from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
from codeflaws.data_utils import get_cfg_ast_cov, all_codeflaws_keys
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from graph_algos.nx_shortcuts import nodes_where, where_node_not, \
        edges_where, where_node
from utils.data_utils import AstNxDataset, del_all_status
import os
import random
import pickle as pkl
import json
import fasttext
import torch
import tqdm
from collections import defaultdict
from pycparser.plyparser import ParseError

embedding_model = fasttext.load_model(ConfigClass.pretrained_fastext)
errorneous_keys = json.load(open('error_instance.json', 'r'))
non_err_keys = [k for k in all_codeflaws_keys if k not in errorneous_keys]


class CodeflawsNxDataset(AstNxDataset):
    def __init__(self, save_dir=ConfigClass.preprocess_dir_codeflaws):
        super().__init__(
                all_entries=all_codeflaws_keys,
                process_func=lambda k: get_cfg_ast_cov(k)[-1],
                save_dir=save_dir, name='key_only',
                special_attrs=[
                    'ast_lb_d': lambda nx_g: nodes_where(nx_g, graph='ast', status=1),
                    'ast_lb_i': lambda nx_g: nodes_where(nx_g, graph='ast', status=2),
                    'cfg_lb': lambda nx_g: nodes_where(nx_g, graph='cfg', status=1)], post_process_func=del_all_status)
        self.cfg_etypes = ['parent_child', 'next', 'ref', 'func_call']

    
class CodeflawsFullDGLDataset(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.graph_save_path = os.path.join(
            save_dir, 'dgl_nx_graphs_only_keys.bin')
        self.info_path = os.path.join(
            save_dir, 'dgl_graphs_full_info_only_key.pkl')
        self.nx_dataset = CodeflawsNxDataset(raw_dir, save_dir)
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(CodeflawsFullDGLDataset, self).__init__(
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

    def convert_from_nx_to_dgl(self, embedding_model, nx_g, ast_lb_d,
                               ast_lb_i, cfg_lb):
        '''
        # Create a node mapping for cfg
        n_cfgs = nodes_where(nx_g, graph='cfg')
        cfg2id = dict([n, i] for i, n in enumerate(n_cfgs))
        '''
        # Create a node mapping for ast
        n_asts = nodes_where(nx_g, graph='ast')
        ast2id = dict([n, i] for i, n in enumerate(n_asts))
        # Create a node mapping for test
        n_ts = nodes_where(nx_g, graph='test')
        t2id = dict([n, i] for i, n in enumerate(n_ts))
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

        all_canon_etypes = defaultdict(list)
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

        for u, v, k, e in edges_where(nx_g, where_node_not(graph='cfg'),
                                      where_node_not(graph='cfg'):
            map_u = map2id[nx_g.nodes[u]['graph']]
            map_v = map2id[nx_g.nodes[v]['graph']]
            all_canon_etypes[
                (nx_g.nodes[u]['graph'], e['label'], nx_g.nodes[v]['graph'])
            ].append([map_u[u], map_v[v]])

        for k in all_canon_etypes:
            type_es = torch.tensor(all_canon_etypes[k], dtype=torch.int32)
            all_canon_etypes[k] = (type_es[:, 0], type_es[:, 1])
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
        ast_tgts[list(map(lambda n: ast2id[n], ast_lb_d))] = 1
        ast_tgts[list(map(lambda n: ast2id[n], ast_lb_i))] = 2
        g.nodes['ast'].data['tgt'] = ast_tgts
        '''
        tgts[list(map(lambda n: cfg2id[n], cfg_lb))] = 1
        g.nodes['cfg'].data['tgt'] = tgts
        '''
        return g

    def construct_edge_metagraph(self):
        self.ast_etypes = self.nx_dataset.ast_etypes + ['a_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.ast_etypes]
        # self.cfg_etypes = self.nx_dataset.cfg_etypes + ['c_self_loop'] + \
        #     [et + '_reverse' for et in self.nx_dataset.cfg_etypes]
        # self.c_a_etypes = ['corresponding_ast']
        # self.a_c_etypes = ['corresponding_cfg']
        # self.c_t_etypes = ['c_pass_t', 'c_fail_t']
        # self.t_c_etypes = ['t_pass_c', 't_fail_c']
        # self.a_t_etypes = ['a_pass_t', 'a_fail_t']
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
        bar = tqdm.tqdm(enumerate(self.nx_dataset))
        bar.set_description("Converting NX to DGL")
        for i, (nx_g, ast_lb_d, ast_lb_i, cfg_lb) in bar:
            g = self.convert_from_nx_to_dgl(embedding_model, nx_g, ast_lb_d,
                                            ast_lb_i, cfg_lb)
            self.gs.append(g)
            if i == 0:
                # self.cfg_content_dim = g.nodes['cfg'].data['content'].shape[-1]
                self.cfg_content_dim = 1
                self.ast_content_dim = g.nodes['ast'].data['content'].shape[-1]

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]]



class CodeflawsFullDGLDatasetCFG(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.graph_save_path = os.path.join(
            save_dir, 'dgl_nx_graphs_only_keys.bin')
        self.info_path = os.path.join(
            save_dir, 'dgl_graphs_full_info_only_key.pkl')
        self.nx_dataset = CodeflawsNxDataset(raw_dir, save_dir)
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(CodeflawsFullDGLDataset, self).__init__(
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
        self.test_idxs = list(self.master_idxs[
            int(len(self.gs)*0.8):int(len(self.gs))])
        self.train()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_graphs(self.graph_save_path, self.gs)
        self.master_idxs = list(range(len(self.gs)))
        random.shuffle(self.master_idxs)
        self.train_idxs = list(self.master_idxs[:int(len(self.gs)*0.6)])
        self.val_idxs = list(self.master_idxs[
            int(len(self.gs)*0.6):int(len(self.gs)*0.8)])
        self.test_idxs = list(self.master_idxs[
            int(len(self.gs)*0.8):int(len(self.gs))])
        pkl.dump({'cfg_content_dim': self.cfg_content_dim,
                  'ast_content_dim': self.ast_content_dim,
                  'master_idxs': self.master_idxs,
                  'train_idxs': self.train_idxs,
                  'val_idxs': self.val_idxs,
                  'test_idxs': self.test_idxs
                  },
                 open(self.info_path, 'wb'))

    def convert_from_nx_to_dgl(self, embedding_model, nx_g, ast_lb_d,
                               ast_lb_i, cfg_lb):
        # Create a node mapping for cfg
        n_cfgs = nodes_where(nx_g, graph='cfg')
        cfg2id = dict([n, i] for i, n in enumerate(n_cfgs))
        # Create a node mapping for ast
        n_asts = nodes_where(nx_g, graph='ast')
        ast2id = dict([n, i] for i, n in enumerate(n_asts))
        # Create a node mapping for test
        n_ts = nodes_where(nx_g, graph='test')
        t2id = dict([n, i] for i, n in enumerate(n_ts))
        map2id = {'cfg': cfg2id, 'ast': ast2id, 'test': t2id}
        # map2id = {'ast': ast2id, 'test': t2id}

        # Create dgl cfg node
        cfg_labels = torch.tensor(
            [ConfigClass.cfg_label_corpus.index(nx_g.nodes[n]['ntype'])
             for n in n_cfgs], dtype=torch.long)
        cfg_contents = torch.stack([
            torch.from_numpy(embedding_model.get_sentence_vector(
                nx_g.nodes[n]['text'].replace('\n', '')))
            for n in n_cfgs], dim=0)
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

        all_canon_etypes = defaultdict(list)
        line2cfg = {}
        for n in n_cfgs:
            if nx_g.nodes[n]['end_line'] - nx_g.nodes[n]['start_line'] > 0:
                continue
            if nx_g.nodes[n]['start_line'] not in line2cfg:
                line2cfg[nx_g.nodes[n]['start_line']] = [n]
            else:
                line2cfg[nx_g.nodes[n]['start_line']].append(n)
        nx_g = augment_with_reverse_edge_cat(
            nx_g, self.nx_dataset.ast_etypes,
            self.nx_dataset.cfg_etypes)

        for u, v, k, e in edges_where(nx_g, where_node_not(graph='cfg'),
                                      where_node_not(graph='cfg'):
            map_u = map2id[nx_g.nodes[u]['graph']]
            map_v = map2id[nx_g.nodes[v]['graph']]
            all_canon_etypes[ (nx_g.nodes[u]['graph'], e['label'], nx_g.nodes[v]['graph'])
            ].append([map_u[u], map_v[v]])

        for k in all_canon_etypes:
            type_es = torch.tensor(all_canon_etypes[k], dtype=torch.int32)
            all_canon_etypes[k] = (type_es[:, 0], type_es[:, 1])

        g = dgl.heterograph(all_canon_etypes)
        g.nodes['cfg'].data['label'] = cfg_labels
        g.nodes['cfg'].data['content'] = cfg_contents
        g.nodes['ast'].data['label'] = ast_labels
        g.nodes['ast'].data['content'] = ast_contents
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))

        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        ast_tgts[list(map(lambda n: ast2id[n], ast_lb_d))] = 1
        ast_tgts[list(map(lambda n: ast2id[n], ast_lb_i))] = 2
        g.nodes['ast'].data['tgt'] = ast_tgts

        tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        tgts[list(map(lambda n: cfg2id[n], cfg_lb))] = 1
        g.nodes['cfg'].data['tgt'] = tgts
        return g

    def construct_edge_metagraph(self):
        self.ast_etypes = self.nx_dataset.ast_etypes + ['a_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.ast_etypes]
        self.cfg_etypes = self.nx_dataset.cfg_etypes + ['c_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.cfg_etypes]
        self.c_a_etypes = ['corresponding_ast']
        self.a_c_etypes = ['corresponding_cfg']
        self.c_t_etypes = ['c_pass_t', 'c_fail_t']
        self.t_c_etypes = ['t_pass_c', 't_fail_c']
        self.a_t_etypes = ['a_pass_t', 'a_fail_t']
        self.t_a_etypes = ['t_pass_a', 't_fail_a']
        self.all_etypes = (
            self.ast_etypes +
            self.cfg_etypes +
            self.c_a_etypes + self.a_c_etypes +
            self.c_t_etypes + self.t_c_etypes +
            self.a_t_etypes +
            self.t_a_etypes)
        self.all_ntypes = (
            [('ast', 'ast') for et in self.ast_etypes] +
            [('cfg', 'cfg') for et in self.cfg_etypes] +
            [('cfg', 'ast') for et in self.c_a_etypes] +
            [('ast', 'cfg') for et in self.a_c_etypes] +
            [('cfg', 'test') for et in self.c_t_etypes] +
            [('test', 'cfg') for et in self.t_c_etypes] +
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
        for i, (nx_g, ast_lb_d, ast_lb_i, cfg_lb) in bar:
            g = self.convert_from_nx_to_dgl(embedding_model, nx_g, ast_lb_d,
                                            ast_lb_i, cfg_lb)
            self.gs.append(g)
            if i == 0:
                self.cfg_content_dim = g.nodes['cfg'].data['content'].shape[-1]
                # self.cfg_content_dim = 1
                self.ast_content_dim = g.nodes['ast'].data['content'].shape[-1]

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]]


class CodeflawsASTDGLDataset(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.graph_save_path = os.path.join(
            save_dir, 'dgl_nx_ast_graphs_only_keys.bin')
        self.info_path = os.path.join(
            save_dir, 'dgl_graphs_ast_info_only_key.pkl')
        self.nx_dataset = CodeflawsNxDataset(raw_dir, save_dir)
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(CodeflawsFullDGLDataset, self).__init__(
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
        self.test_idxs = list(self.master_idxs[
            int(len(self.gs)*0.8):int(len(self.gs))])
        self.train()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_graphs(self.graph_save_path, self.gs)
        self.master_idxs = list(range(len(self.gs)))
        random.shuffle(self.master_idxs)
        self.train_idxs = list(self.master_idxs[:int(len(self.gs)*0.6)])
        self.val_idxs = list(self.master_idxs[
            int(len(self.gs)*0.6):int(len(self.gs)*0.8)])
        self.test_idxs = list(self.master_idxs[
            int(len(self.gs)*0.8):int(len(self.gs))])
        pkl.dump({'cfg_content_dim': self.cfg_content_dim,
                  'ast_content_dim': self.ast_content_dim,
                  'master_idxs': self.master_idxs,
                  'train_idxs': self.train_idxs,
                  'val_idxs': self.val_idxs,
                  'test_idxs': self.test_idxs
                  },
                 open(self.info_path, 'wb'))

    def convert_from_nx_to_dgl(self, embedding_model, nx_g, ast_lb_d,
                               ast_lb_i, cfg_lb):
        '''
        # Create a node mapping for cfg
        n_cfgs = nodes_where(nx_g, graph='cfg')
        cfg2id = dict([n, i] for i, n in enumerate(n_cfgs))
        '''
        # Create a node mapping for ast
        n_asts = nodes_where(nx_g, graph='ast')
        ast2id = dict([n, i] for i, n in enumerate(n_asts))
        # Create a node mapping for test
        # n_ts = nodes_where(nx_g, graph='test')
        # t2id = dict([n, i] for i, n in enumerate(n_ts))
        # map2id = {'cfg': cfg2id, 'ast': ast2id, 'test': t2id}
        # map2id = {'ast': ast2id, 'test': t2id}
        map2id = {'ast': ast2id}

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

        all_canon_etypes = defaultdict(list)
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
        nx_g = augment_with_reverse_edge_cat(
            nx_g, self.nx_dataset.ast_etypes,
            self.nx_dataset.cfg_etypes)

        for u, v, k, e in edges_where(nx_g, where_node(graph='ast'), where_node(graph='ast')):
            all_canon_etypes[('ast', e['label'], 'ast')].append([ast2id[u], ast2id[v]])

        for k in all_canon_etypes:
            type_es = torch.tensor(all_canon_etypes[k], dtype=torch.int32)
            all_canon_etypes[k] = (type_es[:, 0], type_es[:, 1])
        g = dgl.heterograph(all_canon_etypes)
        '''
        g.nodes['cfg'].data['label'] = cfg_labels
        g.nodes['cfg'].data['content'] = cfg_contents
        '''
        g.nodes['ast'].data['label'] = ast_labels
        g.nodes['ast'].data['content'] = ast_contents
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        # g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        ast_tgts[list(map(lambda n: ast2id[n], ast_lb_d))] = 1
        ast_tgts[list(map(lambda n: ast2id[n], ast_lb_i))] = 2
        g.nodes['ast'].data['tgt'] = ast_tgts

        '''
        tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        tgts[list(map(lambda n: cfg2id[n], cfg_lb))] = 1
        g.nodes['cfg'].data['tgt'] = tgts
        '''
        return g

    def construct_edge_metagraph(self):
        self.ast_etypes = self.nx_dataset.ast_etypes + ['a_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.ast_etypes]
        # self.cfg_etypes = self.nx_dataset.cfg_etypes + ['c_self_loop'] + \
        #     [et + '_reverse' for et in self.nx_dataset.cfg_etypes]
        # self.c_a_etypes = ['corresponding_ast']
        # self.a_c_etypes = ['corresponding_cfg']
        # self.c_t_etypes = ['c_pass_t', 'c_fail_t']
        # self.t_c_etypes = ['t_pass_c', 't_fail_c']
        # self.a_t_etypes = ['a_pass_t', 'a_fail_t']
        # self.t_a_etypes = ['t_pass_a', 't_fail_a']
        self.all_etypes = self.ast_etypes
        self.all_ntypes = [('ast', 'ast') for et in self.ast_etypes]

        return [(t[0], et, t[1]) for t, et in zip(self.all_ntypes,
                                                  self.all_etypes)]

    def process(self):
        self.meta_graph = self.construct_edge_metagraph()
        self.gs = []
        bar = tqdm.tqdm(enumerate(self.nx_dataset))
        bar.set_description("Converting NX to DGL")
        for i, (nx_g, ast_lb_d, ast_lb_i, cfg_lb) in bar:
            g = self.convert_from_nx_to_dgl(embedding_model, nx_g, ast_lb_d,
                                            ast_lb_i, cfg_lb)
            self.gs.append(g)
            if i == 0:
                # self.cfg_content_dim = g.nodes['cfg'].data['content'].shape[-1]
                self.cfg_content_dim = 1
                self.ast_content_dim = g.nodes['ast'].data['content'].shape[-1]

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]]
