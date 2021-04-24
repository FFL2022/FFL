from dataset import build_dgl_graph
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, save_info, load_info
from utils import ConfigClass
import os
import pickle as pkl
import json
import fasttext


class BugLocalizeGraphDataset(DGLDataset):
    def __init__(self, raw_dir, save_dir='preprocessed'):
        self.graph_save_path = os.path.join(save_dir, 'dgl_graphs.pkl')
        self.label_save_path = os.path.join(save_dir, 'label_processed.pkl')

        super(BugLocalizeGraphDataset, self).__init__(
            name='key_value_dataset',
            url=None,
            raw_dir=None,
            save_dir=save_dir,
            force_reload=False,
            verbose=False)

    def __getitem__(self, i):
        pass

    def process(self):
        train_map = json.load(open(
            os.path.join(self.raw_dir, 'training_dat.json'), 'r'))
        test_verdict = pkl.load(open(
            os.path.join(self.raw_dir, 'test_verdict.pkl'), 'rb'))
        self.keys = train_map.keys()
        self.gs = []
        self.lbls = []
        self.ast_id2idx = []
        self.cfg_id2idx = []
        self.test_id2idx = []
        for key in self.keys:
            # Get the mapping
            # Get the train index
            problem_id, uid, program_id = key.split("-")
            instance_verdict = test_verdict[problem_id][int(program_id)]
            G, ast_id2idx, cfg_id2idx, test_id2idx = build_dgl_graph(problem_id, program_id, instance_verdict)
            self.gs.append(G)
            self.ast_id2idx.append(ast_id2idx)
            self.cfg_id2idx.append(cfg_id2idx)
            self.test_id2idx.append(test_id2idx)
            # Process label

    def has_cache(self):
        if os.path.exists(self.graph_save_path) and\
                os.path.exists(self.label_save_path):
            return True
        return False


model = fasttext.load_model(ConfigClass.pretrained_fastext)
