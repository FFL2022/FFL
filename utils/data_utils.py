from typing import Tuple, List, Union
import random
from collections import Sequence
from itertools import accumulate, Callable
import networkx as nx


class NxDataset(Sequence):
    def __init__(self):
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def has_cache(self):
        raise NotImplementedError


class AstNxDataset(NxDataset):
    def __init__(self, all_entries, process_func, save_dir, name,
            special_attrs: List[Tuple[str, Callable[[nx.Graph], None]]]):
        self.save_dir = save_dir
        self.info_path = f"{save_dir}/nx_{name}_info.pkl"
        self.name = name
        self.process_func = process_func
        self.all_entries = all_entries
        self.special_attrs = special_attrs
        for k, _ in self.special_attrs:
            setattr(self, k, [])
        super().__init__()

    def has_cache(self):
        return os.path.exists(self.info_path)

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        try:
            nx_g = pkl.load(open(
                f'{self.save_dir}/nx_{self.name}_{self.active_idxs[i]}.pkl', 'rb'))
        except UnicodeDecodeError:
            nx_g = self.process_func(self.all_entries[self.active_idxs[i])
            pkl.dump(
                nx_g,
                open(f'{self.save_dir}/nx_{self.name}_{self.active_idxs[i]}.pkl',
                     'wb'))
        return nx_g, *[self.__dict__[k][i] for k, _ in self.special_attrs]

    def process(self):
        self.ast_types = set()
        self.ast_etypes = set()
        self.keys = []
        self.err_idxs = []
        self.active_idxs = []

        bar = tqdm.tqdm(list(all_codeflaws_keys))
        bar.set_description(f'Loading Nx Data {self.name}')
        for i, key in enumerate(bar):
            try:
                if not os.path.exists(f'{self.save_dir}/nx_{self.name}_{i}.pkl'):
                    nx_g = self.process_func(key)
                    pkl.dump(
                        nx_g,
                        open(f'{self.save_dir}/nx_{self.name}_{i}.pkl', 'wb')
                    )
                else:
                    nx_g = pkl.load(open(
                        f'{self.save_dir}/nx_{self.name}_{i}.pkl', 'rb')
                    )
            except:
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
            for k, f in self.special_attrs:
                self.__dict__[k].append(f(nx_g))

    def load(self):
        info_dict = pkl.load(open(self.info_path, 'rb'))
        self.ast_types = info_dict['ast_types']
        self.ast_etypes = info_dict['ast_etypes']
        self.keys = info_dict['keys']
        self.err_idxs = info_dict['err_idxs']
        self.active_idxs = info_dict['active_idxs']
        for k, _ in self.special_attrs:
            setattr(self, k, info_dict[k])

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # gs is saved somewhere else
        pkl.dump(
            {
                'ast_types': self.ast_types, 'ast_etypes': self.ast_etypes,
                'keys': self.keys, 'err_idxs': self.err_idxs,
                'active_idxs': self.active_idxs
                **{k: self.__dict__[k] for k, _ in self.special_attrs}
            },
            open(self.info_path, 'wb'))


class NxDataloader(Sequence):
    def __init__(self, nx_dataset: NxDataset, idxs):
        self.idxs = idxs
        self.nx_dataset = nx_dataset

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.nx_dataset[self.idxs[i]]

    def get_dataset(self) -> NxDataset:
        return self.nx_dataset


def split_nx_dataset(nx_dataset: NxDataset,
                     ratio: Union[int, List[int]]=0.8,
                     shuffle=True) -> Tuple[NxDataset, NxDataset]:
    N = len(nx_dataset)
    idxs = list(range(N))
    if shuffle:
        random.shuffle(idxs)
    if isinstance(ratio, list):
        assert len(ratio), "Ratio must not be empty List"
        assert sum(ratio) == 1, "Sum of ratio must be 1"
        cumsum = list(accumulate(ratio))
        out = []
        for i, c in enumerate(cumsum):
            if i:
                out.append(NxDataloader(
                    nx_dataset, idxs[int(cumsum[i-1]*N):int(c*N)]))
            else:
                out.append(NxDataloader(nx_dataset, idxs[:int(c*N)]))
        return out
    return NxDataloader(nx_dataset, idxs[:int(ratio*N)]),\
        NxDataloader(nx_dataset, idxs[int(N*ratio):])
