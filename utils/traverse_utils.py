from cfg.cfg_nodes import CFGNode, CFGEntryNode
import networkx as nx


def is_leaf(astnode):
    if isinstance(astnode, str):
        return True
    return len(astnode.children()) == 0


def get_token(astnode, lower=True):
    if isinstance(astnode, str):
        return astnode.node
    name = astnode.__class__.__name__
    token = name
    is_name = False
    if is_leaf(astnode):
        attr_names = astnode.attr_names
        if attr_names:
            if 'names' in attr_names:
                token = astnode.names[0]
            elif 'name' in attr_names:
                token = astnode.name
                is_name = True
            else:
                token = astnode.value
        else:
            token = name
    else:
        if name == 'TypeDecl':
            token = astnode.declname
        if astnode.attr_names:
            attr_names = astnode.attr_names
            if 'op' in attr_names:
                if astnode.op[0] == 'p':
                    token = astnode.op[1:]
                else:
                    token = astnode.op
    if token is None:
        token = name
    if lower and is_name:
        token = token.lower()
    return token


def traverse_ast(node, index, parent, parent_index):
    tmp_n = {}
    tmp_e = {}
    if parent_index != 0:
        tmp_e[(parent_index, index+1)] = 1
    index += 1
    curr_index = index
    node_token = get_token(node)
    if node_token == "TypeDecl":
        coord_line = node.type.coord.line
    else:
        try:
            coord_line = node.coord.line
        except AttributeError:
            coord_line = parent.coord.line

    tmp_n[index] = [node_token, coord_line]

    for edgetype, child in node.children():
        if child is not None:
            index, n, e = traverse_ast(child, index, node, curr_index)
            tmp_e.update(e)
            tmp_n.update(n)
    return index, tmp_n, tmp_e


def traverse_cfg(graph):
    list_cfg_nodes = {}
    list_cfg_edges = {}
    list_callfunction = [node._func_name for node in graph._entry_nodes]
    list_callfuncline = {}
    parent = {}
    is_traversed = []
    for i in range(len(graph._entry_nodes)):
        # Loop through entry nodes
        entry_node = graph._entry_nodes[i]
        # Node type on each line
        list_cfg_nodes[entry_node.line] = "entry_node"
        # Entry node is often function
        list_callfuncline[entry_node._func_name] = entry_node.line
        # Visited node
        is_traversed.append(entry_node)
        if isinstance(entry_node._func_first_node, CFGNode):
            queue = []
            node = entry_node._func_first_node
            queue.append(node)
            parent[node] = entry_node.line
            while len(queue) > 0:
                # print(queue)
                node = queue.pop(0)
                # print(node.get_start_line())
                if node not in is_traversed:
                    # print(node.get_start_line())
                    # print(node._type)
                    # print(node.get_children())
                    parent_id = parent[node]
                    is_traversed.append(node)
                    start_line = node.get_start_line()
                    last_line = node.get_last_line()
                    for child in node.get_children():
                        # print(child)
                        parent[child] = start_line
                        queue.append(child)
                    if node._type == "END":
                        pass
                    else:
                        if node._type == "CALL":
                            x = node.get_ast_elem_list()
                            for func in x:
                                try:
                                    call_index = list_callfuncline[func.name.name]
                                    list_cfg_edges[(last_line, call_index)] = 1
                                except KeyError:
                                    pass

                        list_cfg_edges[(parent_id, start_line)] = 1
                        for i in range(start_line, last_line + 1, 1):
                            if i != last_line:
                                list_cfg_edges[(i, i+1)] = 1
                            list_cfg_nodes[i] = node._type
    return list_cfg_nodes, list_cfg_edges


def build_nx_cfg(graph, break_to_line=True):
    '''Build networkx version of cfg'''
    g = nx.MultiDiGraph()
    cfg2nx = {}
    ''' There exists 2 types of CFGNode:
        1. CFGEntryNode:
            Function definition
        2. CFGNode
        In the CFGNode, there will also be different node types: "COMMON",
        "IF", "ELSE", "ELSE_IF", "FOR", "WHILE", "DO_WHILE", "PSEUDO",
        "CALL", "END"
    '''
    for i in range(len(graph._entry_nodes)):
        # Loop through entry nodes
        entry_node = graph._entry_nodes[i]
        if entry_node not in cfg2nx:
            cfg2nx[entry_node] = g.number_of_nodes()
            g.add_node(cfg2nx[entry_node], ntype="entry_node",
                       funcname=entry_node._func_name,
                       start_line=entry_node.line,
                       end_line=entry_node.line,
                       )
        if isinstance(entry_node._func_first_node, CFGNode):
            # Entry to the function
            queue = []
            node = entry_node._func_first_node
            queue.append(node)
            cfg2nx[node] = g.number_of_nodes()
            g.add_node(cfg2nx[node], ntype=node._type,
                       start_line=node.get_start_line(),
                       end_line=node.get_last_line()
                       )
            g.add_edge(cfg2nx[entry_node],
                       cfg2nx[node],
                       label='parent_child')
            while len(queue) > 0:
                # print(queue)
                node = queue.pop(0)
                # print(node.get_children())
                min_start_line_child = node.get_last_line()
                if len(node.get_children()) > 0:
                    for child in node.get_children():
                        if child not in cfg2nx:
                            cfg2nx[child] = g.number_of_nodes()
                            min_start_line_child = min(min_start_line_child,
                                                       child.get_start_line())
                            g.add_node(cfg2nx[child], ntype=child._type,
                                       start_line=child.get_start_line(),
                                       end_line=child.get_last_line()
                                       )
                            g.add_edge(cfg2nx[node],
                                       cfg2nx[child],
                                       label='parent_child')
                            # print(child)
                            queue.append(child)
                    # Break node down to smaller components of same type
                if node._type == 'COMMON' and break_to_line:
                    start_line = g.nodes[cfg2nx[node]]['start_line']
                    end_line = min_start_line_child
                    if end_line - start_line > 1:
                        for line in range(start_line, end_line + 1):
                            line_idx = g.number_of_nodes()
                            cfg2nx[g.number_of_nodes()] = line_idx
                            g.add_node(line_idx, ntype='COMMON',
                                       start_line=line,
                                       end_line=line)
                            g.add_edge(cfg2nx[node], line_idx,
                                       label='parent_child')
                            if line >= start_line + 1:
                                g.add_edge(line_idx-1,
                                           line_idx,
                                           label='next')

                if node._type == "END":
                    pass
                else:
                    if node._type == "CALL":
                        x = node.get_ast_elem_list()
                        for func in x:
                            # Find the node which have that funcname
                            # So that we can connect them together
                            mapping = dict([
                                (g.nodes[n]['funcname'], n)
                                for n in g.nodes() if
                                g.nodes[n]['ntype'] == 'entry_node'
                            ])
                            try:
                                dst_node = mapping[func.name.name]
                                g.add_edge(node, dst_node,
                                           label='func_call')
                            except KeyError:
                                pass
                    if node._type == "PSEUDO" or node._type == "CALL":
                        refnode = node.get_refnode()
                        if refnode is not None:
                            if refnode not in cfg2nx:
                                cfg2nx[refnode] = g.number_of_nodes()
                                if isinstance(refnode, CFGEntryNode):
                                    g.add_node(cfg2nx[refnode],
                                               ntype="entry_node",
                                               funcname=refnode._func_name,
                                               start_line=refnode.line,
                                               end_line=refnode.line)
                                    new_node = refnode._func_first_node
                                    cfg2nx[new_node] = g.number_of_nodes()
                                    g.add_node(cfg2nx[new_node], ntype=node._type,
                                               start_line=node.get_start_line(),
                                               end_line=node.get_last_line()
                                               )
                                    g.add_edge(cfg2nx[refnode],
                                               cfg2nx[new_node],
                                               label='parent_child')
                                    queue.append(new_node)
                                else:
                                    g.add_node(cfg2nx[refnode], ntype=node._type,
                                               start_line=refnode.get_start_line(),
                                               end_line=refnode.get_last_line())
                                    queue.append(refnode)
                            g.add_edge(cfg2nx[node], cfg2nx[refnode],
                                       label='ref')

    # Connect every consecutive lines between the node's
    # range
    startline2node = dict([(g.nodes[node]['start_line'], [])
                          for node in g.nodes()])
    for node in g.nodes():
        startline2node[g.nodes[node]['start_line']].append(node)
    largest_startline = max(list(startline2node.keys()))
    for node in g.nodes():
        if g.nodes[node]['ntype'] == 'entry_node':
            continue
        next_line = g.nodes[node]['end_line'] + 1
        while next_line not in startline2node and\
                next_line <= largest_startline:
            next_line += 1
        if next_line > largest_startline:
            continue
        # Find the largest node among them
        candidate = max(startline2node[next_line],
                        key=lambda node: g.nodes[node]['end_line']
                        )
        if not g.has_edge(node, candidate):
            g.add_edge(node, candidate, label='next')
    return g, cfg2nx


def get_unique_base(children: list):
    unique_child_map = {}
    for child_name, child in children:
        if "[" in child_name:
            base = child_name.split("[")[0]
            index = int(child_name.split("[")[1].split("]")[0])
            if base not in unique_child_map:
                unique_child_map[base] = {index: child}
            else:
                unique_child_map[base][index] = child
        else:
            unique_child_map[child_name] = {0: child}
    return unique_child_map


def build_nx_ast(ast):
    g = nx.MultiDiGraph()
    ast2nx = {ast: 0}

    g.add_node(0, ntype=ast.__class__.__name__,
               token=get_token(ast), coord_line=-1)
    queue = [ast]

    while len(queue) > 0:
        node = queue.pop()
        # Child name can also be used as edge etype
        # First, check childname
        if len(node.children()) > 0:
            unique_child_map = get_unique_base(list(node.children()))
            for etype in unique_child_map:
                # Sort all nodes
                sorted_cidxs = sorted(list(unique_child_map[etype].keys()))
                for i in sorted_cidxs:
                    child = unique_child_map[etype][i]
                    child_token = get_token(child)
                    if child_token == "TypeDecl":
                        coord_line = node.type.coord.line
                    else:
                        try:
                            coord_line = node.coord.line
                        except AttributeError:
                            coord_line = g.nodes[ast2nx[node]]['coord_line']
                    ast2nx[child] = g.number_of_nodes()
                    g.add_node(g.number_of_nodes(),
                               ntype=child.__class__.__name__,
                               token=get_token(child),
                               coord_line=coord_line)
                    g.add_edge(ast2nx[node], ast2nx[child], label=etype)
                    if i > 0:
                        g.add_edge(ast2nx[child]-1, ast2nx[child],
                                   label='next_sibling')
                    queue.insert(0, child)
    return g, ast2nx
