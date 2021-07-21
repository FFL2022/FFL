from cfg.cfg_nodes import CFGNode


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
        if child != None:
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
       entry_node = graph._entry_nodes[i]
       list_cfg_nodes[entry_node.line] = "entry_node"
       list_callfuncline[entry_node._func_name] = entry_node.line
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
