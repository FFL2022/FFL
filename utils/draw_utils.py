import networkx as nx


def cfg_to_agraph(cfg: nx.MultiDiGraph, path: str):
    ''' Cfg to agraph
    AGraph: used by pygraphviz for visualization
    Parameters
    ----------
    cfg:  nx.MultiDiGraph
          cfg graph build from "build_nx_cfg"
    path:  str
    '''
    for node in cfg.nodes():
        cfg.nodes[node]['label'] = cfg.nodes[node]['ntype']
        if cfg.nodes[node]['ntype'] == 'entry_node':
            cfg.nodes[node]['label'] = cfg.nodes[node]['ntype'] + \
                ' ' + cfg.nodes[node]['funcname']
    nx.drawing.nx_agraph.to_agraph(cfg).draw(path, prog='dot')


def ast_to_agraph(ast: nx.MultiDiGraph, path: str):
    for node in ast.nodes():
        ast.nodes[node]['label'] = ast.nodes[node]['ntype'] + \
            ' ' + ast.nodes[node]['token']
    nx.drawing.nx_agraph.to_agraph(ast).draw(path, prog='dot')


def cfg_ast_to_agraph(cfg_ast: nx.MultiDiGraph, path: str):
    for node in cfg_ast.nodes():
        if cfg_ast.nodes[node]['graph'] == 'cfg':
            cfg_ast.nodes[node]['label'] = cfg_ast.nodes[node]['ntype']
            if cfg_ast.nodes[node]['ntype'] == 'entry_node':
                cfg_ast.nodes[node]['label'] = cfg_ast.nodes[node]['ntype'] + \
                    ' ' + cfg_ast.nodes[node]['funcname']
        elif cfg_ast.nodes[node]['graph'] == 'ast':
            cfg_ast.nodes[node]['label'] = cfg_ast.nodes[node]['ntype'] + \
                ' ' + cfg_ast.nodes[node]['token']
    nx.drawing.nx_agraph.to_agraph(cfg_ast).draw(path, prog='dot')


def cfg_ast_cov_to_agraph(cfg_ast_cov: nx.MultiDiGraph, path):
    g = cfg_ast_cov
    for node in g.nodes():
        if g.nodes[node]['graph'] == 'cfg':
            g.nodes[node]['label'] = g.nodes[node]['ntype']
            if g.nodes[node]['ntype'] == 'entry_node':
                g.nodes[node]['label'] = g.nodes[node]['ntype'] + \
                    ' ' + g.nodes[node]['funcname']
        elif g.nodes[node]['graph'] == 'ast':
            g.nodes[node]['label'] = g.nodes[node]['ntype'] + \
                ' ' + g.nodes[node]['token']
        else:
            g.nodes[node]['label'] = g.nodes[node]['name']

    nx.drawing.nx_agraph.to_agraph(g).draw(path, prog='dot')
