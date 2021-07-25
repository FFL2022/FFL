from utils.utils import ConfigClass
import os
import tqdm
from pycparser.plyparser import ParseError
from utils.codeflaws_data_utils import get_cfg_ast_cov, all_codeflaws_keys
import json
import pickle as pkl


class CodeflawsNxDataset(object):
    def __init__(self, raw_dataset_dir=ConfigClass.raw_dir,
                 save_dir=ConfigClass.preprocess_dir):
        self.save_dir = save_dir
        self.graph_save_path = os.path.join(
            save_dir, 'nx_graphs_keyonly.bin')
        self.cfg_etypes = ['parent_child', 'next', 'ref', 'func_call']
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

        self.active_idxs = len(self.nx_gs)

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.nx_gs[i]

    def process(self):
        self.nx_gs = []
        self.ast_types = []
        self.ast_etypes = []
        self.max_ast_arity = 0
        self.keys = []
        error_instance = []
        bar = tqdm.tqdm(all_codeflaws_keys)
        bar.set_description('Loading Nx Data')
        err_count = 0
        for key in bar:
            try:
                _, _, _, _, _, nx_g = get_cfg_ast_cov(key)
            except ParseError:
                err_count += 1
                print(f"Total syntax error files: {err_count}")
                if key not in error_instance:
                    error_instance.append(key)
                json.dump(error_instance, open('error_instance.json', 'w'))
                continue
            for n in nx_g.nodes():
                if nx_g.nodes[n]['graph'] == 'ast':
                    self.max_ast_arity = max(
                        self.max_ast_arity, nx_g.nodes[n]['n_order'])
            self.keys.append(key)
            self.nx_gs.append(nx_g)
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
        pkl.dump(
            {'nx': self.nx_gs, 'max_arity': self.max_ast_arity,
             'ast_types': self.ast_types, 'ast_etypes': self.ast_etypes},
            open(self.graph_save_path, 'wb'))

    def load(self):
        gs_label = pkl.load(open(self.graph_save_path, 'rb'))
        self.nx_gs = gs_label['nx']
        self.ast_types = gs_label['ast_types']
        self.ast_etypes = gs_label['ast_etypes']
        self.max_ast_arity = gs_label['max_ast_arity']

    def has_cache(self):
        return os.path.exists(self.graph_save_path)

if __name__ == '__main__':
    dataset = CodeflawsNxDataset()
