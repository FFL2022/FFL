from graph_algos.nx_shortcuts import where_node, edges_where, nodes_where
import dgl
from dgl.data import DGLDataset
from codeflaws.dataloader_cfl import CodeflawsAstGraphMetadata, \
        NxDataloader


def numerize_graph(nx_g, graphs=['ast', 'cfg', 'test']):
    n2id = {}
    for graph in graphs:
        ns = nodes_where(nx_g, graph=graph)
        n2id[graph] = dict([n, i] for i, n in enumerate(ns))

    # Create dgl test node
    # No need, will be added automatically when we update edges
    all_canon_etypes = defaultdict(list)

    for u, v, k, e in edges_where(nx_g, where_node(graph=graphs),
                                  where_node(graph=graphs)):
        map_u = n2id[nx_g.nodes[u]['graph']]
        map_v = n2id[nx_g.nodes[v]['graph']]
        all_canon_etypes[
            (nx_g.nodes[u]['graph'], e['label'], nx_g.nodes[v]['graph'])
        ].append([map_u[u], map_v[v]])

    for k in all_canon_etypes:
        type_es = torch.tensor(all_canon_etypes[k], dtype=torch.int32)
        all_canon_etypes[k] = (type_es[:, 0], type_es[:, 1])

    g = dgl.heterograph(all_canon_etypes)
    return g, n2id


class CodeDGLDataset(DGLDataset):
    def __init__(self, dataloader: NxDataloader,
                 meta_data: CodeflawsAstGraphMetadata,
                 name: str,
                 save_dir,
                 convert_arg_func=lambda x: x):
        self.name = name
        self.graph_save_path = os.path.join(
            save_dir, f'dgl_{self.name}.bin')
        self.info_path = os.path.join(
            save_dir, f'dgl_{self.name}_info.pkl')
        self.dataloader = dataloader
        self.meta_data = meta_data
        self.convert_arg_func = convert_arg_func

        super().__init__(
            name='codeflaws_dgl', url=None,
            raw_dir=".", save_dir=save_dir,
            force_reload=False, verbose=False)

        self.train()
        super().__init__(
            name=name, url=None, raw_dir=raw_dir,
            save_dir=save_dir, force_reload=False,
            verbose=False)
    
    def has_cache(self):
        return os.path.exists(self.graph_save_path) and\
            os.path.exists(self.info_path)

    def __len__(self):
        return len(self.gs)


    def load(self):
        self.gs = load_graphs(self.graph_save_path)[0]
        self.meta_graph = self.construct_edge_metagraph()
        for k, v in pkl.load(open(self.info_path, 'rb')):
            setattr(self, k, v)
        self.train()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_graphs(self.graph_save_path, self.gs)
        pkl.dump({'additionals': self.additionals},
                 open(self.info_path, 'wb'))

    def convert_from_nx_to_dgl(*args):
        raise NotImplementedError

    def process(self):
        self.meta_graph = self.meta_data.meta_graph()
        self.gs = []
        self.additionals = []
        bar = tqdm.tqdm(enumerate(self.dataloader))
        bar.set_description(f"Converting NX to DGL for {self.name}")
        for i, x in bar:
            if isinstance(x, tuple):
                g = self.convert_from_nx_to_dgl(
                        *convert_arg_func(x))
            else:
                g = self.convert_from_nx_to_dgl(
                        *convert_arg_func(x))
            if isinstance(g, tuple):
                self.gs.append(g[0])
                if i == 0:
                    self.additionals = [[] for _ in range(len(g) - 1)]
                for a in g[1:]:
                    self.additionals.append(a)
            else:
                self.gs.append(g)

    def __getitem__(self, i):
        if self.additionals:
            return self.gs[i], *self.additionals
        else:
            return self.gs[i]
