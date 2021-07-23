from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
from utils.nx_graph_builder import build_nx_cfg_ast_coverage_codeflaws,\
    build_nx_cfg_coverage_codeflaws, augment_with_reverse_edge
from utils.codeflaws_data_utils import make_codeflaws_dict
import os
import pickle as pkl
import json
import fasttext
import torch
import tqdm
from pycparser.plyparser import ParseError

embedding_model = fasttext.load_model(ConfigClass.pretrained_fastext)


class CodeflawsNxDataset(object):
    def __init__(self, raw_dataset_dir=ConfigClass.raw_dir,
                 save_dir=ConfigClass.preprocess_dir,
                 label_mapping_path=ConfigClass.codeflaws_train_cfgidx_map_pkl,
                 graph_opt=2):
        self.save_dir = save_dir
        self.mode = os.path.splitext(os.path.split(label_mapping_path)[-1])[0]
        self.graph_save_path = os.path.join(
            save_dir, f'nx_graphs_{graph_opt}_{self.mode}.bin')
        self.cfg_etypes = ['parent_child', 'next', 'ref']
        self.graph_opt = graph_opt
        self.label_mapping_path = label_mapping_path
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

        self.active_idxs = len(self.nx_gs)

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.nx_gs[i], self.lbs[i]

    def process(self):
        self.total_train = 0
        label_mapping = pkl.load(open(self.label_mapping_path, 'rb'))
        test_verdict = pkl.load(open(
            ConfigClass.codeflaws_test_verdict_pickle, 'rb'))

        self.temp_keys = list(label_mapping.keys())
        self.nx_gs = []
        self.ast_types = []
        self.ast_etypes = []
        self.lbs = []
        self.keys = []

        error_instance = []
        bar = tqdm.tqdm(enumerate(self.temp_keys))
        bar.set_description('Loading Nx Data')
        err_count = 0
        for i, key in bar:
            data_codeflaws = make_codeflaws_dict(key, test_verdict)
            try:
                if self.graph_opt == 2:
                    _, _, _, nx_g = build_nx_cfg_ast_coverage_codeflaws(
                        data_codeflaws)
                else:
                    _, _, _, nx_g = build_nx_cfg_coverage_codeflaws(
                        data_codeflaws)
            except ParseError:
                err_count += 1
                print(f"Total syntax error files: {err_count}")
                if key not in error_instance:
                    error_instance.append(key)
                json.dump(error_instance, open('error_instance.json', 'w'))
                continue
            self.keys.append(key)
            self.nx_gs.append(nx_g)
            self.ast_types.extend(
                [nx_g.nodes[node]['ntype'] for node in nx_g.nodes()
                 if nx_g.nodes[node]['graph'] == 'ast'])

            self.ast_etypes.extend(
                [e['label'] for u, v, k, e in nx_g.edges(keys=True, data=True)
                 if nx_g.nodes[u]['graph'] == 'ast' and
                 nx_g.nodes[v]['graph'] == 'ast'])

            # Process label
            # or not
            self.lbs.append(label_mapping[key])
        self.ast_types = list(set(self.ast_types))
        self.ast_etypes = list(set(self.ast_etypes))

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        pkl.dump(
            {'nx': self.nx_gs, 'lbs': self.lbs,
             'ast_types': self.ast_types, 'ast_etypes': self.ast_etypes},
            open(self.graph_save_path, 'wb'))

    def load(self):
        gs_label = pkl.load(open(self.graph_save_path, 'rb'))
        self.nx_gs = gs_label['nx']
        self.lbs = gs_label['lbs']
        self.ast_types = gs_label['ast_types']
        self.ast_etypes = gs_label['ast_etypes']

    def has_cache(self):
        return os.path.exists(self.graph_save_path)


class CodeflawsDGLDataset(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.raw_dir,
                 save_dir=ConfigClass.preprocess_dir,
                 label_mapping_path=ConfigClass.codeflaws_train_cfgidx_map_pkl,
                 graph_opt=1):
        self.mode = os.path.splitext(os.path.split(label_mapping_path)[-1])[0]
        self.graph_save_path = os.path.join(
            save_dir, f'dgl_nx_graphs_{graph_opt}_{self.mode}.bin')
        self.info_path = os.path.join(
            save_dir, 'dgl_graphs_info.pkl')
        self.graph_opt = graph_opt
        self.nx_dataset = CodeflawsNxDataset(raw_dir, save_dir,
                                             label_mapping_path, graph_opt)
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(CodeflawsDGLDataset, self).__init__(
            name='codeflaws_dgl',
            url=None,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=False,
            verbose=False)

        if 'train' in self.mode:    # Only split to train val when train
            self.train_idxs = range(int(len(self.gs)*0.8))
            self.val_idxs = range(int(len(self.gs)*0.8), len(self.gs))

            self.active_idxs = self.train_idxs
        else:
            self.active_idxs = range(len(self.gs))
            self.val_idxs = self.active_idxs

    def train(self):
        self.active_idxs = self.train_idxs

    def val(self):
        self.active_idxs = self.val_idxs

    def has_cache(self):
        return os.path.exists(self.graph_save_path) and\
            os.path.exists(self.info_path)

    def load(self):
        self.gs = load_graphs(self.graph_save_path)[0]
        self.meta_graph = self.construct_edge_metagraph()
        info_dict = pkl.load(open(self.info_path, 'rb'))
        self.cfg_content_dim = info_dict['cfg_content_dim']
        self.ast_content_dim = info_dict['ast_content_dim']

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_graphs(self.graph_save_path, self.gs)
        pkl.dump({'cfg_content_dim': self.cfg_content_dim,
                  'ast_content_dim': self.ast_content_dim},
                 open(self.info_path, 'wb'))

    def convert_from_nx_to_dgl(self, embedding_model, nx_g, lbl):
        # Create a node mapping for cfg
        n_cfgs = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'cfg']
        cfg2id = dict([n, i] for i, n in enumerate(n_cfgs))
        # Create a node mapping for ast
        n_asts = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'ast']
        ast2id = dict([n, i] for i, n in enumerate(n_asts))
        # Create a node mapping for test
        n_tests = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'test']
        t2id = dict([n, i] for i, n in enumerate(n_tests))
        map2id = {'cfg': cfg2id, 'ast': ast2id, 'test': t2id}

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

        all_canon_etypes = {}
        for k in self.meta_graph:
            all_canon_etypes[k] = []
        line2cfg = {}
        for n in n_cfgs:
            if nx_g.nodes[n]['end_line'] - nx_g.nodes[n]['start_line'] > 0:
                continue
            if nx_g.nodes[n]['start_line'] not in line2cfg:
                line2cfg[nx_g.nodes[n]['start_line']] = [n]
            else:
                line2cfg[nx_g.nodes[n]['start_line']].append(n)
        nx_g = augment_with_reverse_edge(nx_g, self.nx_dataset.ast_etypes,
                                         self.nx_dataset.cfg_etypes)

        for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
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
        g.nodes['cfg'].data['label'] = cfg_labels
        g.nodes['cfg'].data['content'] = cfg_contents
        g.nodes['ast'].data['label'] = ast_labels
        g.nodes['ast'].data['content'] = ast_contents
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        for line in lbl:
            while line not in line2cfg:
                line -= 1   # Note: Matching to the closest (presumably parent)
            for n in line2cfg[line]:
                tgts[cfg2id[n]] = 1
        g.nodes['cfg'].data['tgt'] = tgts
        return g

    def construct_edge_metagraph(self):
        self.ast_etypes = self.nx_dataset.ast_etypes + ['a_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.ast_etypes]
        self.cfg_etypes = self.nx_dataset.cfg_etypes + ['c_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.cfg_etypes]
        self.c_a_etypes = ['corresponding_ast']
        self.a_c_etypes = ['corresponding_cfg']
        self.c_t_etypes = ['c_pass_test', 'c_fail_test']
        self.t_c_etypes = ['t_pass_c', 't_fail_c']
        self.a_t_etypes = ['a_pass_test', 'a_fail_test']
        self.t_a_etypes = ['t_pass_a', 't_fail_a']
        self.all_etypes = self.ast_etypes + self.cfg_etypes +\
            self.c_a_etypes + self.a_c_etypes +\
            self.c_t_etypes + self.t_c_etypes + self.a_t_etypes +\
            self.t_a_etypes
        self.all_ntypes = [('ast', 'ast') for et in self.ast_etypes] +\
            [('cfg', 'cfg') for et in self.cfg_etypes] +\
            [('cfg', 'ast') for et in self.c_a_etypes] +\
            [('ast', 'cfg') for et in self.a_c_etypes] +\
            [('cfg', 'test') for et in self.c_t_etypes] +\
            [('test', 'cfg') for et in self.t_c_etypes] +\
            [('ast', 'test') for et in self.a_t_etypes] +\
            [('test', 'ast') for et in self.t_a_etypes]

        return [(t[0], et, t[1]) for t, et in zip(self.all_ntypes,
                                                  self.all_etypes)]

    def process(self):
        self.meta_graph = self.construct_edge_metagraph()
        self.gs = []
        bar = tqdm.tqdm(enumerate(self.nx_dataset))
        bar.set_description("Converting NX to DGL")
        for i, (nx_g, lbl) in bar:
            g = self.convert_from_nx_to_dgl(embedding_model, nx_g, lbl)
            self.gs.append(g)
            if i == 0:
                self.cfg_content_dim = g.nodes['cfg'].data['content'].shape[-1]
                self.ast_content_dim = g.nodes['ast'].data['content'].shape[-1]

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]]


class TestCodeflawsDGLDataset(DGLDataset):
    def __init__(self, train_dataset, raw_dir=ConfigClass.raw_dir,
                 save_dir=ConfigClass.preprocess_dir,
                 label_mapping_path=ConfigClass.codeflaws_train_cfgidx_map_pkl,
                 graph_opt=1):
        self.mode = os.path.splitext(os.path.split(label_mapping_path)[-1])[0]
        self.train_dataset = train_dataset
        self.graph_save_path = os.path.join(
            save_dir, f'dgl_nx_graphs_{graph_opt}_{self.mode}.bin')
        self.graph_opt = graph_opt
        self.nx_dataset = CodeflawsNxDataset(raw_dir, save_dir,
                                             label_mapping_path, graph_opt)
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))

        super(TestCodeflawsDGLDataset, self).__init__(
            name='codeflaws_dgl',
            url=None,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=False,
            verbose=False)

        self.active_idxs = range(len(self.gs))
        self.val_idxs = self.active_idxs

    def val(self):
        self.active_idxs = self.val_idxs

    def has_cache(self):
        return os.path.exists(self.graph_save_path)

    def load(self):
        self.gs = load_graphs(self.graph_save_path)[0]
        self.meta_graph = self.train_dataset.meta_graph
        self.cfg_content_dim = self.train_dataset.cfg_content_dim
        self.ast_content_dim = self.train_dataset.ast_content_dim

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_graphs(self.graph_save_path, self.gs)

    def convert_from_nx_to_dgl(self, embedding_model, nx_g, lbl):
        # Create a node mapping for cfg
        n_cfgs = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'cfg']
        cfg2id = dict([n, i] for i, n in enumerate(n_cfgs))
        # Create a node mapping for ast
        n_asts = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'ast']
        ast2id = dict([n, i] for i, n in enumerate(n_asts))
        # Create a node mapping for test
        n_tests = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'test']
        t2id = dict([n, i] for i, n in enumerate(n_tests))
        map2id = {'cfg': cfg2id, 'ast': ast2id, 'test': t2id}

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
            self.train_dataset.nx_dataset.ast_types.index(
                nx_g.nodes[node]['ntype'])
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
        line2cfg = {}
        for n in n_cfgs:
            if nx_g.nodes[n]['end_line'] - nx_g.nodes[n]['start_line'] > 0:
                continue
            if nx_g.nodes[n]['start_line'] not in line2cfg:
                line2cfg[nx_g.nodes[n]['start_line']] = [n]
            else:
                line2cfg[nx_g.nodes[n]['start_line']].append(n)
        nx_g = augment_with_reverse_edge(
            nx_g,
            self.train_dataset.nx_dataset.ast_etypes,
            self.train_dataset.nx_dataset.cfg_etypes)

        for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
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
        g.nodes['cfg'].data['label'] = cfg_labels
        g.nodes['cfg'].data['content'] = cfg_contents
        g.nodes['ast'].data['label'] = ast_labels
        g.nodes['ast'].data['content'] = ast_contents
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        for line in lbl:
            while line not in line2cfg:
                line -= 1   # Note: Matching to the closest (presumably parent)
            for n in line2cfg[line]:
                tgts[cfg2id[n]] = 1
        g.nodes['cfg'].data['tgt'] = tgts
        return g

    def process(self):
        self.meta_graph = self.construct_edge_metagraph()
        self.gs = []
        bar = tqdm.tqdm(enumerate(self.nx_dataset))
        bar.set_description("Converting NX to DGL")
        for i, (nx_g, lbl) in bar:
            g = self.convert_from_nx_to_dgl(embedding_model, nx_g, lbl)
            self.gs.append(g)
            if i == 0:
                self.cfg_content_dim = g.nodes['cfg'].data['content'].shape[-1]
                self.ast_content_dim = g.nodes['ast'].data['content'].shape[-1]

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]]


class CodeflawsFullDGLDataset(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.raw_dir,
                 save_dir=ConfigClass.preprocess_dir,
                 label_mapping_path=ConfigClass.codeflaws_full_cfgidx_map_pkl,
                 graph_opt=1):
        self.mode = os.path.splitext(os.path.split(label_mapping_path)[-1])[0]
        self.graph_save_path = os.path.join(
            save_dir, f'dgl_nx_graphs_{graph_opt}_{self.mode}.bin')
        self.info_path = os.path.join(
            save_dir, f'dgl_graphs_full_info.pkl')
        self.graph_opt = graph_opt
        self.nx_dataset = CodeflawsNxDataset(raw_dir, save_dir,
                                             label_mapping_path, graph_opt)
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
        self.train_idxs = self.master_idx[:int(len(self.gs)*0.6)]
        self.val_idxs = self.master_idx[
            int(len(self.gs)*0.6):int(len(self.gs)*0.8)]
        self.test_idxs = self.master_idx[
            int(len(self.gs)*0.9):int(len(self.gs))]
        pkl.dump({'cfg_content_dim': self.cfg_content_dim,
                  'ast_content_dim': self.ast_content_dim,
                  'master_idxs': self.master_idxs,
                  'train_idxs': self.train_idxs,
                  'val_idxs': self.val_idxs,
                  'test_idxs': self.test_idxs
                  },
                 open(self.info_path, 'wb'))

    def convert_from_nx_to_dgl(self, embedding_model, nx_g, lbl):
        # Create a node mapping for cfg
        n_cfgs = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'cfg']
        cfg2id = dict([n, i] for i, n in enumerate(n_cfgs))
        # Create a node mapping for ast
        n_asts = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'ast']
        ast2id = dict([n, i] for i, n in enumerate(n_asts))
        # Create a node mapping for test
        n_tests = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'test']
        t2id = dict([n, i] for i, n in enumerate(n_tests))
        map2id = {'cfg': cfg2id, 'ast': ast2id, 'test': t2id}

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

        all_canon_etypes = {}
        for k in self.meta_graph:
            all_canon_etypes[k] = []
        line2cfg = {}
        for n in n_cfgs:
            if nx_g.nodes[n]['end_line'] - nx_g.nodes[n]['start_line'] > 0:
                continue
            if nx_g.nodes[n]['start_line'] not in line2cfg:
                line2cfg[nx_g.nodes[n]['start_line']] = [n]
            else:
                line2cfg[nx_g.nodes[n]['start_line']].append(n)
        nx_g = augment_with_reverse_edge(nx_g, self.nx_dataset.ast_etypes,
                                         self.nx_dataset.cfg_etypes)

        for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
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
        g.nodes['cfg'].data['label'] = cfg_labels
        g.nodes['cfg'].data['content'] = cfg_contents
        g.nodes['ast'].data['label'] = ast_labels
        g.nodes['ast'].data['content'] = ast_contents
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        for line in lbl:
            while line not in line2cfg:
                line -= 1   # Note: Matching to the closest (presumably parent)
            for n in line2cfg[line]:
                tgts[cfg2id[n]] = 1
        g.nodes['cfg'].data['tgt'] = tgts
        return g

    def construct_edge_metagraph(self):
        self.ast_etypes = self.nx_dataset.ast_etypes + ['a_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.ast_etypes]
        self.cfg_etypes = self.nx_dataset.cfg_etypes + ['c_self_loop'] + \
            [et + '_reverse' for et in self.nx_dataset.cfg_etypes]
        self.c_a_etypes = ['corresponding_ast']
        self.a_c_etypes = ['corresponding_cfg']
        self.c_t_etypes = ['c_pass_test', 'c_fail_test']
        self.t_c_etypes = ['t_pass_c', 't_fail_c']
        self.a_t_etypes = ['a_pass_test', 'a_fail_test']
        self.t_a_etypes = ['t_pass_a', 't_fail_a']
        self.all_etypes = self.ast_etypes + self.cfg_etypes +\
            self.c_a_etypes + self.a_c_etypes +\
            self.c_t_etypes + self.t_c_etypes + self.a_t_etypes +\
            self.t_a_etypes
        self.all_ntypes = [('ast', 'ast') for et in self.ast_etypes] +\
            [('cfg', 'cfg') for et in self.cfg_etypes] +\
            [('cfg', 'ast') for et in self.c_a_etypes] +\
            [('ast', 'cfg') for et in self.a_c_etypes] +\
            [('cfg', 'test') for et in self.c_t_etypes] +\
            [('test', 'cfg') for et in self.t_c_etypes] +\
            [('ast', 'test') for et in self.a_t_etypes] +\
            [('test', 'ast') for et in self.t_a_etypes]

        return [(t[0], et, t[1]) for t, et in zip(self.all_ntypes,
                                                  self.all_etypes)]

    def process(self):
        self.meta_graph = self.construct_edge_metagraph()
        self.gs = []
        bar = tqdm.tqdm(enumerate(self.nx_dataset))
        bar.set_description("Converting NX to DGL")
        for i, (nx_g, lbl) in bar:
            g = self.convert_from_nx_to_dgl(embedding_model, nx_g, lbl)
            self.gs.append(g)
            if i == 0:
                self.cfg_content_dim = g.nodes['cfg'].data['content'].shape[-1]
                self.ast_content_dim = g.nodes['ast'].data['content'].shape[-1]

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]]
