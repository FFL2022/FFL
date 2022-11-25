from utils.utils import ConfigClass
from codeflaws.data_utils import all_codeflaws_keys,\
    get_nx_ast_stmt_annt_cfl, \
    cfl_check_is_stmt_cpp
from graph_algos.nx_shortcuts import nodes_where, edges_where,\
        where_node
import os
import pickle as pkl
import tqdm
from pycparser.plyparser import ParseError
from utils.data_utils import NxDataset


class CodeflawsCFLNxStatementDataset(NxDataset):
    def __init__(self, raw_dataset_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.save_dir = save_dir
        self.info_path = f"{save_dir}/nx_cfl_stmt_dataset_info.pkl"
        super().__init__()

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        try:
            nx_g = pkl.load(open(
                f'{self.save_dir}/nx_cfl_stmt_{self.active_idxs[i]}.pkl', 'rb')
            )
        except UnicodeDecodeError:
            nx_g = get_nx_ast_stmt_annt_cfl(
                all_codeflaws_keys[self.active_idxs[i]])
            pkl.dump(
                nx_g,
                open(f'{self.save_dir}/nx_cfl_stmt_{self.active_idxs[i]}.pkl',
                     'wb'))
        return nx_g, self.stmt_nodes[i]

    def process(self):
        self.ast_types = set()
        self.ast_etypes = set()
        self.stmt_nodes = []
        self.keys = []
        self.err_idxs = []
        self.active_idxs = []

        bar = tqdm.tqdm(list(all_codeflaws_keys))
        bar.set_description('Loading Nx Data with CFL')
        # bar = list(enumerate(all_codeflaws_keys))
        for i, key in enumerate(bar):
            try:
                if not os.path.exists(f'{self.save_dir}/nx_cfl_stmt_{i}.pkl'):
                    nx_g = get_nx_ast_stmt_annt_cfl(key)
                    pkl.dump(
                        nx_g,
                        open(f'{self.save_dir}/nx_cfl_stmt_{i}.pkl', 'wb')
                    )
                else:
                    nx_g = pkl.load(open(
                        f'{self.save_dir}/nx_cfl_stmt_{i}.pkl', 'rb')
                    )
            except ParseError:
                self.err_idxs.append(i)
                count = len(self.err_idxs)
                bar.set_postfix(syntax_error_files=count)
                continue
            self.active_idxs.append(i)
            self.keys.append(key)
            self.ast_types = self.ast_types.union(
                [nx_g.nodes[n]['ntype'] for n in nodes_where(nx_g, graph='ast')])
            self.ast_etypes = self.ast_etypes.union(
                x[-1]['label'] for x in edges_where(nx_g, where_node(graph='ast'), where_node(graph='ast')))
            self.stmt_nodes.append(
                nodes_where(
                    nx_g,
                    lambda x: cfl_check_is_stmt_cpp(nx_g.nodes[x]),
                    graph='ast'))

        self.ast_types = list(self.ast_types)
        self.ast_etypes = list(self.ast_etypes)

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # gs is saved somewhere else
        pkl.dump(
            {
                'ast_types': self.ast_types, 'ast_etypes': self.ast_etypes,
                'keys': self.keys, 'err_idxs': self.err_idxs,
                'stmt_nodes': self.stmt_nodes,
                'active_idxs': self.active_idxs
            },
            open(self.info_path, 'wb'))

    def load(self):
        info_dict = pkl.load(open(self.info_path, 'rb'))
        self.ast_types = info_dict['ast_types']
        self.ast_etypes = info_dict['ast_etypes']
        self.keys = info_dict['keys']
        self.err_idxs = info_dict['err_idxs']
        self.stmt_nodes = info_dict['stmt_nodes']
        self.active_idxs = info_dict['active_idxs']

    def has_cache(self):
        return os.path.exists(self.info_path)


class CodeflawsCFLStatementGraphMetadata(object):
    def __init__(self, nx_g_dataset):
        self.t_asts = nx_g_dataset.ast_types
        self.ntype2id = {n: i for i, n in enumerate(self.t_asts)}
        self.t_e_asts = nx_g_dataset.ast_etypes
        self.meta_graph = self.construct_edge_metagraph()

    def construct_edge_metagraph(self):
        self.t_e_a_a = self.t_e_asts + \
            list(map(lambda x: f'{x}_reverse', self.t_e_asts))
        self.t_e_a_t = ['a_pass_t', 'a_fail_t']
        self.t_e_t_a = ['t_pass_a', 't_fail_a']
        self.t_all = (self.t_e_a_a + self.t_e_a_t + self.t_e_t_a)
        self.srcs = ['ast'] * len(self.t_e_a_a) +\
            ['ast'] * len(self.t_e_a_t) + ['test'] * len(self.t_e_t_a)
        self.dsts = ['ast'] * len(self.t_e_a_a) +\
            ['test'] * len(self.t_e_a_t) + ['ast'] * len(self.t_e_t_a)
        return list(zip(self.srcs, self.t_all, self.dsts))
