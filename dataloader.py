from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from utils.utils import ConfigClass
import os
import pickle as pkl
import json
import fasttext
import torch


class BugLocalizeGraphDataset(DGLDataset):
    def __init__(self, raw_dir=ConfigClass.raw_dir,
                 save_dir=ConfigClass.preprocess_dir,
                 dataset_opt='nbl', graph_opt=1):
        self.graph_save_path = os.path.join(save_dir, 'dgl_graphs.bin')
        self.info_path = os.path.join(save_dir, 'info.pkl')
        self.mode = 'train'
        self.dataset_opt = dataset_opt
        self.graph_opt = graph_opt

        super(BugLocalizeGraphDataset, self).__init__(
            name='key_value_dataset',
            url=None,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=False,
            verbose=False)

        self.train_idxs = range(int(self.total_train))
        self.val_idxs = range(self.total_train, len(self.gs))
        
        self.active_idxs = self.train_idxs

    def train(self):
        self.mode = 'train'
        self.active_idxs = self.train_idxs

    def val(self):
        self.mode = 'val'
        self.active_idxs = self.val_idxs

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        i = self.active_idxs[i]
        g = self.gs[i]
        lbs = torch.zeros([g.num_nodes('cfg')])
        for j in self.lbs[i]:
            if j in self.cfg_id2idx[i].keys():
                lbs[self.cfg_id2idx[i][j]] = 1

        return g, lbs.long()

    def process(self):
        from dataset import build_dgl_graph
        self.total_train = 0 
        if self.dataset_opt == 'nbl':
            train_map = pkl.load(open(
                ConfigClass.train_cfgidx_map_pkl, 'rb'))
            eval_map = pkl.load(open(
                ConfigClass.eval_cfgidx_map_pkl, 'rb'))
            test_verdict = pkl.load(open(
                os.path.join(self.raw_dir, 'test_verdict.pkl'), 'rb'))
        if self.dataset_opt == 'codeflaws':
            train_map = pkl.load(open(
                ConfigClass.codeflaws_train_cfgidx_map_pkl, 'rb'))
            eval_map = pkl.load(open(
                ConfigClass.codeflaws_eval_cfgidx_map_pkl, 'rb'))
            test_verdict = pkl.load(open(
                ConfigClass.codeflaws_test_verdict_pickle, 'rb'))
        self.temp_keys = list(train_map.keys())
        self.gs = []
        self.lbs = []
        self.ast_id2idx = []
        self.cfg_id2idx = []
        self.test_id2idx = []
        self.keys = []

        model = fasttext.load_model(ConfigClass.pretrained_fastext)
        error_instance = []
        for i, key in enumerate(self.temp_keys):
            nbl, codeflaws = None, None
            if self.dataset_opt == 'nbl':
                problem_id, uid, program_id = key.split("-")
                nbl = {}
                nbl['problem_id'] = problem_id
                nbl['uid'] = uid
                nbl['program_id'] = program_id
                nbl['test_verdict'] = test_verdict[problem_id][int(program_id)]
            if self.dataset_opt == 'codeflaws':
                info = key.split("-")
                codeflaws = {}
                codeflaws['container'] = key
                codeflaws['c_source'] = "{}-{}-{}".format(info[0], info[1], info[3])
                codeflaws['test_verdict'] = test_verdict["{}-{}".format(info[0], info[1])][info[3]]
            try:
                G, ast_id2idx, cfg_id2idx, test_id2idx = build_dgl_graph(nbl=nbl, codeflaws=codeflaws, model=model, graph_opt=self.graph_opt)
            except:
                if key not in error_instance:
                    error_instance.append(key)
                json.dump(error_instance, open('error_instance.json', 'w'))
                continue
            self.keys.append(key)
            self.gs.append(G)
            self.ast_id2idx.append(ast_id2idx)
            self.cfg_id2idx.append(cfg_id2idx)
            self.test_id2idx.append(test_id2idx)
            if i == 0:
                self.cfg_label_feats = G.nodes['cfg'].data['label'].shape[-1]
                self.cfg_content_feats = G.nodes['cfg'].data['content'].shape[-1]
                if (self.graph_opt == 2):
                    self.ast_label_feats = G.nodes['ast'].data['label'].shape[-1]
                    self.ast_content_feats = G.nodes['ast'].data['content'].shape[-1]
            # Process label
            # or not
            self.lbs.append(train_map[key])
        self.total_train = len(self.gs)

        self.temp_keys = list(eval_map.keys())
        for i, key in enumerate(self.temp_keys):
            nbl, codeflaws = None, None
            if self.dataset_opt == 'nbl':
                problem_id, uid, program_id = key.split("-")
                nbl = {}
                nbl['problem_id'] = problem_id
                nbl['uid'] = uid
                nbl['program_id'] = program_id
                nbl['test_verdict'] = test_verdict[problem_id][int(program_id)]
            if self.dataset_opt == 'codeflaws':
                info = key.split("-")
                codeflaws = {}
                codeflaws['container'] = key
                codeflaws['c_source'] = "{}-{}-{}".format(info[0], info[1], info[3])
                codeflaws['test_verdict'] = test_verdict["{}-{}".format(info[0], info[1])][info[3]]
            try:
                G, ast_id2idx, cfg_id2idx, test_id2idx = build_dgl_graph(nbl=nbl, codeflaws=codeflaws, model=model, graph_opt=self.graph_opt)
            except:
                if key not in error_instance:
                    error_instance.append(key)
                json.dump(error_instance, open('error_instance.json', 'w'))
                continue
            self.keys.append(key)
            self.gs.append(G)
            self.ast_id2idx.append(ast_id2idx)
            self.cfg_id2idx.append(cfg_id2idx)
            self.test_id2idx.append(test_id2idx)
            if i == 0:
                self.cfg_label_feats = G.nodes['cfg'].data['label'].shape[-1]
                self.cfg_content_feats = G.nodes['cfg'].data['content'].shape[-1]
                if (self.graph_opt == 2):
                    self.ast_label_feats = G.nodes['ast'].data['label'].shape[-1]
                    self.ast_content_feats = G.nodes['ast'].data['content'].shape[-1]
            # Process label
            # or not
            self.lbs.append(eval_map[key])

    def download(self):
        pass

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # print(len(self.gs))
        save_graphs(self.graph_save_path, self.gs)
        if (self.graph_opt == 1):
            save_info(self.info_path, {
                                    # 'ast_id2idx': self.ast_id2idx,
                                    'labels': self.lbs,
                                    'keys': self.keys,
                                    'cfg_id2idx': self.cfg_id2idx,
                                    'test_id2idx': self.test_id2idx,
                                    'cfg_content_feats': self.cfg_content_feats,
                                    'cfg_label_feats': self.cfg_label_feats,
                                    'total_train': self.total_train
                                    }
                    )
        if (self.graph_opt == 2):
            save_info(self.info_path, {'ast_id2idx': self.ast_id2idx,
                                    'labels': self.lbs,
                                    'keys': self.keys,
                                    'cfg_id2idx': self.cfg_id2idx,
                                    'test_id2idx': self.test_id2idx,
                                    'cfg_content_feats': self.cfg_content_feats,
                                    'cfg_label_feats': self.cfg_label_feats,
                                    'ast_content_feats': self.ast_content_feats,
                                    'ast_label_feats': self.ast_label_feats,
                                    'total_train': self.total_train
                                    }
                    )

    def load(self):
        self.gs, _ = load_graphs(self.graph_save_path)
        info_dict = load_info(self.info_path)
        self.lbs = info_dict['labels']
        self.keys = info_dict['keys']
        self.cfg_id2idx = info_dict['cfg_id2idx']
        self.cfg_content_feats = info_dict['cfg_content_feats']
        self.cfg_label_feats = info_dict['cfg_label_feats']
        self.total_train = info_dict['total_train']
        if (self.graph_opt == 2):
            self.ast_id2idx = info_dict['ast_id2idx']
            self.ast_content_feats = info_dict['ast_content_feats']
            self.ast_label_feats = info_dict['ast_label_feats']

    def has_cache(self):
        if os.path.exists(self.graph_save_path) and\
                os.path.exists(self.info_path):
            return True
        return False


# default_dataset = BugLocalizeGraphDataset()
