import dgl
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from utils.data_utils import NxDataloader, AstGraphMetadata
import torch
from dgl_version.data_utils import numerize_graph, CodeDGLDataset


def convert_from_nx_to_dgl(graph, nx_g, stmt_nodes, meta_data):
    nx_g = augment_with_reverse_edge_cat(nx_g, meta_data.t_e_asts, [])
    g, n2id = numerize_graph(nx_g, ['ast', 'test'])
    ast2id = n2id['ast']
    # Create dgl ast node
    ast_labels = torch.tensor([
        meta_data.t_e_asts.index(nx_g.nodes[node]['ntype']) for node in ast2id
    ], dtype=torch.long)

    g.nodes['ast'].data['label'] = ast_labels
    g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
    ast_tgts = torch.zeros(len(ast2id), dtype=torch.long)
    ast_tgts[list(map(lambda x: ast2id[x], ast2id))] = torch.tensor(
        [nx_g.nodes[x]['status'] for x in ast2id])
    g.nodes['ast'].data['tgt'] = ast_tgts
    stmt_idxs = [ast2id[n] for n in stmt_nodes]
    return g, stmt_idxs


class CFLDGLStatementDataset(CodeDGLDataset):

    def __init__(self,
                 dataloader: NxDataloader,
                 meta_data: AstGraphMetadata,
                 name: str,
                 save_dir="preprocessed"):
        self.name = f"{name}_dgl_statement"
        super().__init__(dataloader, meta_data, self.name, save_dir)

    def convert_from_nx_to_dgl(self, nx_g, stmt_nodes):
        return convert_from_nx_to_dgl(self, nx_g, stmt_nodes, self.meta_data)
