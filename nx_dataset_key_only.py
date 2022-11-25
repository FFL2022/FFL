from utils.utils import ConfigClass
import os
import tqdm
from pycparser.plyparser import ParseError
from utils.codeflaws_data_utils import get_cfg_ast_cov, all_codeflaws_keys
from graph_algos.nx_shortcuts import nodes_where
import json
import pickle as pkl


class CodeflawsNxDataset(object):
    def __init__(self, raw_dataset_dir=ConfigClass.raw_dir,
                 save_dir=ConfigClass.preprocess_dir):
        self.save_dir = save_dir
        self.info_path = os.path.join(
            save_dir, 'nx_key_only_info.pkl')
        self.cfg_etypes = ['parent_child', 'next', 'ref', 'func_call']
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

        self.active_idxs = len(self.ast_lbs_d)

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return pkl.load(open(
            f'{self.save_dir}/nx_{self.active_idxs[i]}', 'rb')),\
            self.ast_lbs_d[self.active_idxs[i]], \
            self.ast_lbs_i[self.active_idxs[i]], \
            self.cfg_lbs[self.active_idxs[i]]

    def process(self):
        self.ast_types = []
        self.ast_etypes = []
        self.ast_lbs_i = []
        self.ast_lbs_d = []
        self.cfg_lbs = []
        self.keys = []
        error_instance = []
        bar = tqdm.tqdm(enumerate(all_codeflaws_keys))
        bar.set_description('Loading Nx Data')
        err_count = 0
        for i, key in bar:
            try:
                _, _, _, _, _, nx_g = get_cfg_ast_cov(key)
                ast_lb_d = nodes_where(nx_g, graph='ast', status=1)
                ast_lb_i = nodes_where(nx_g, graph='ast', status=2)
                cfg_lb = nodes_where(nx_g, graph='cfg', status=1)
                for n in (nodes_where(nx_g, graph='cfg') +
                          nodes_where(nx_g, graph='ast')):
                    del nx_g.nodes[n]['status']
                pkl.dump(nx_g,
                         open(f"{self.save_dir}/nx_{i}", 'wb'))
            except ParseError:
                err_count += 1
                print(f"Total syntax error files: {err_count}")
                if key not in error_instance:
                    error_instance.append(key)
                json.dump(error_instance, open('error_instance.json', 'w'))
                continue
            self.keys.append(key)
            self.ast_types.extend(
                [nx_g.nodes[node]['ntype'] for node in nx_g.nodes()
                 if nx_g.nodes[node]['graph'] == 'ast'])
            self.ast_lbs_i.append(ast_lb_i)
            self.ast_lbs_d.append(ast_lb_d)
            self.cfg_lbs.append(cfg_lb)

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
                'ast_lb_d': self.ast_lbs_d,
                'ast_lb_i': self.ast_lbs_i,
                'cfg_lb': self.cfg_lbs,
                'ast_types': self.ast_types, 'ast_etypes': self.ast_etypes},
            open(self.info_path, 'wb'))

    def load(self):
        gs_label = pkl.load(open(self.info_path, 'rb'))
        self.ast_types = gs_label['ast_types']
        self.ast_etypes = gs_label['ast_etypes']
        self.ast_lbs_d = gs_label['ast_lb_d']
        self.ast_lbs_i = gs_label['ast_lb_i']
        self.cfg_lbs = gs_label['cfg_lb']

    def has_cache(self):
        return os.path.exists(self.info_path)


if __name__ == '__main__':
    dataset = CodeflawsNxDataset()
