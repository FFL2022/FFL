from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
from codeflaws.data_utils import all_codeflaws_keys,\
    get_nx_ast_node_annt_gumtree,\
    get_nx_ast_stmt_annt_gumtree
from utils.nx_graph_builder import augment_with_reverse_edge_cat
import os
import random
import pickle as pkl
import json
import fasttext
import torch
import tqdm

from json import JSONDecodeError

class CodeflawsGumtreeNxStmtDataset(object):
    def __init__(self, raw_dataset_dir=ConfigClass.raw_dir,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.save_dir = save_dir
        self.info_path = os.path.join(
            save_dir, 'nx_gumtree_dataset_info.pkl')
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
            f'{self.save_dir}/nx_gumtree_{self.active_idxs[i]}', 'rb')),\
            self.stmt_nodes[self.active_idxs[i]]

    def process(self):
        self.ast_types = []
        self.ast_etypes = []
        self.stmt_nodes = []
        self.keys = []
        self.err_idxs = []
        error_instance = []
        bar = tqdm.tqdm(list(enumerate(all_codeflaws_keys)))
        bar.set_description('Loading Nx Data with gumtree')
        for i, key in bar:
            try:
                nx_g = get_nx_ast_stmt_annt_gumtree(key)
                pkl.dump(nx_g, open(
                    f'{self.save_dir}/nx_gumtree_{i}.pkl', 'wb')
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
