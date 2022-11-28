from dgl_version.data_utils import CodeDGLDataset, numerize_graph
from utils.nx_graph_builder import augment_with_reverse_edge_cat

class CodeflawsGumtreeDGLStatementDataset(CodeDGLDataset):
    def __init__(self, dataloader, meta_data, mode, save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.name = f'{mode}_cf_gumtree_dgl_ast_stmt'
        super().__init__(dataloader, meta_data, self.name, save_dir)

    def convert_from_nx_to_dgl(self, nx_g, stmt_nodes):
        nx_g = augment_with_reverse_edge_cat(nx_g, self.meta_graph.t_e_asts, [])
        g, n2id = numerize_graph(nx_g, ['ast', 'test'])
        ast2id = n2id['ast']
        # Create dgl ast node
        g.nodes['ast'].data['label'] = torch.tensor([
            self.nx_dataset.ast_types.index(nx_g.nodes[node]['ntype'])
            for node in ast2id], dtype=torch.long
        )
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        ast_tgts[list(map(lambda x: ast2id[x], ast2id))] = list(
                nx_g.nodes[x]['status'] for x in ast2id)
        g.nodes['ast'].data['tgt'] = ast_tgts
        stmt_idxs = [ast2id[n] for n in stmt_nodes]
        return g, stmt_idxs
