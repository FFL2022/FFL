import networkx as nx


def set_label_ast(ast, node):
    ast.nodes[node]['label'] = ast.nodes[node]['ntype'] + \
        ' ' + ast.nodes[node]['token']
    if 'status' in ast.nodes[node]:
        if ast.nodes[node]['status'] == 'i':
            ast.nodes[node]['fillcolor'] = "#CCFFCC"
            ast.nodes[node]['style'] = "filled"
        elif ast.nodes[node]['status'] == 'd':
            ast.nodes[node]['fillcolor'] = "#FFCCCC"
            ast.nodes[node]['style'] = "filled"


def set_label_cfg(cfg, node):
    cfg.nodes[node]['label'] = cfg.nodes[node]['ntype'] + \
        cfg.nodes[node]['text']
    if cfg.nodes[node]['ntype'] == 'entry_node':
        cfg.nodes[node]['label'] = cfg.nodes[node]['ntype'] + \
            ' ' + cfg.nodes[node]['funcname'] + ' ' +\
            cfg.nodes[node]['text']
    if 'status' in cfg.nodes[node]:
        if cfg.nodes[node]['status'] == 'm':
            cfg.nodes[node]['fillcolor'] = "#FFFFCC"
            cfg.nodes[node]['style'] = "filled"


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
        set_label_cfg(cfg, node)
    nx.drawing.nx_agraph.to_agraph(cfg).draw(path, prog='dot')


def ast_to_agraph(ast: nx.MultiDiGraph, path: str):
    ''' Nx Ast to agraph
    Parameters
    ----------
    ast:  nx.MultiDiGraph
    path:  str
    '''
    for node in ast.nodes():
        set_label_ast(ast, node)
    nx.drawing.nx_agraph.to_agraph(ast).draw(path, prog='dot')


def cfg_ast_to_agraph(cfg_ast: nx.MultiDiGraph, path: str):
    ''' Cfg ast to agraph
    Parameters
    ----------
    cfg_ast:  nx.MultiDiGraph
    path:  str
    '''
    for node in cfg_ast.nodes():
        if cfg_ast.nodes[node]['graph'] == 'cfg':
            set_label_cfg(cfg_ast, node)
        elif cfg_ast.nodes[node]['graph'] == 'ast':
            set_label_ast(cfg_ast, node)
    nx.drawing.nx_agraph.to_agraph(cfg_ast).draw(path, prog='dot')


def cfg_ast_cov_to_agraph(cfg_ast_cov: nx.MultiDiGraph, path):
    g = cfg_ast_cov
    for node in g.nodes():
        if g.nodes[node]['graph'] == 'cfg':
            set_label_cfg(cfg_ast_cov, node)
        elif g.nodes[node]['graph'] == 'ast':
            set_label_ast(cfg_ast_cov, node)
        else:
            g.nodes[node]['label'] = g.nodes[node]['name']

    nx.drawing.nx_agraph.to_agraph(g).draw(path, prog='dot')
