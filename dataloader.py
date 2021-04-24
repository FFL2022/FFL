from dataset import build_dgl_graph
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
                 save_dir=ConfigClass.preprocess_dir):
        self.graph_save_path = os.path.join(save_dir, 'dgl_graphs.bin')
        self.info_path = os.path.join(save_dir, 'info.pkl')
        self.mode = 'train'

        super(BugLocalizeGraphDataset, self).__init__(
            name='key_value_dataset',
            url=None,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=False,
            verbose=False)
        self.train_idxs = range(int(len(self.gs)*0.8))
        self.val_idxs = range(int(len(self.gs)*0.8), len(self.gs))
        self.active_idxs = self.train_idxs

    def train(self):
        self.mode = 'train'
        self.active_idxs = self.train_idxs

    def val(self):
        self.mode = 'val'
        self.active_idxs = self.val_idxs

    def __getitem__(self, i):
        i = self.active_idxs[i]
        g = self.gs[i]
        lbs = torch.zeros([g.num_nodes('cfg')])
        lbs[self.cfg_id2idx[i][self.lbs[i]]] = 1
        return g, lbs

    def process(self):
        train_map = json.load(open(
            os.path.join(self.raw_dir, 'training_dat.json'), 'r'))
        test_verdict = pkl.load(open(
            os.path.join(self.raw_dir, 'test_verdict.pkl'), 'rb'))
        self.keys = list(train_map.keys())
        self.gs = []
        self.lbs = []
        self.ast_id2idx = []
        self.cfg_id2idx = []
        self.test_id2idx = []

        model = fasttext.load_model(ConfigClass.pretrained_fastext)
        error_instance = {}
        for i, key in enumerate(self.keys):
            # Get the mapping
            # Get the train index
            problem_id, uid, program_id = key.split("-")
            instance_verdict = test_verdict[problem_id][int(program_id)]
            print("Program id {}, problem id {}".format(program_id, problem_id))
            # Temporary only:
            try:
                G, ast_id2idx, cfg_id2idx, test_id2idx = build_dgl_graph(problem_id, program_id, instance_verdict, model)
            except:
                if problem_id not in error_instance.keys():
                    error_instance[problem_id] = []
                error_instance[problem_id].append(program_id)
                json.dump(error_instance, open('error_instance.json', 'w'))
            self.gs.append(G)
            self.ast_id2idx.append(ast_id2idx)
            self.cfg_id2idx.append(cfg_id2idx)
            self.test_id2idx.append(test_id2idx)
            if i == 0:
                self.cfg_label_feats = G.nodes['cfg'].data['label'].shape[-1]
                self.cfg_content_feats = G.nodes['cfg'].data['content'].shape[-1]
            # Process label
            # or not
            self.lbs.append(train_map[key])

    def download(self):
        pass

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_graphs(self.graph_save_path, self.gs)
        save_info(self.info_path, {'ast_id2idx': self.ast_id2idx,
                                   'labels': self.lbs,
                                   'keys': self.keys,
                                   'cfg_id2idx': self.cfg_id2idx,
                                   'test_id2idx': self.test_id2idx,
                                   'cfg_content_feats': self.cfg_content_feats,
                                   'cfg_label_feats': self.cfg_label_feats
                                   }
                  )

    def load(self):
        self.gs = load_graphs(self.graph_save_path)
        info_dict = load_info(self.info_path)
        self.lbs = info_dict['labels']
        self.keys = info_dict['keys']
        self.cfg_id2idx = info_dict['cfg_id2idx']
        self.cfg_content_feats = info_dict['cfg_content_feats']
        self.cfg_label_feats = info_dict['cfg_content_feats']

    def has_cache(self):
        if os.path.exists(self.graph_save_path) and\
                os.path.exists(self.info_path):
            return True
        return False


default_dataset = BugLocalizeGraphDataset()
