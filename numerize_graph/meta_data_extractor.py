import networkx as nx
from typing import List
from collections import defaultdict
import inspect
import sys
from typing import List, Tuple, Dict


def get_all_attr_names(nx_gs: List[nx.Graph]) -> Tuple[List[str], List[str]]:
    # Get all nodes and edges attribute names
    node_attr_names = set()
    edge_attr_names = set()
    for nx_g in nx_gs:
        for node in nx_g.nodes:
            node_attr_names.update(nx_g.nodes[node].keys())
        for edge in nx_g.edges:
            edge_attr_names.update(nx_g.edges[edge].keys())
    return sorted(node_attr_names), sorted(edge_attr_names)


def get_node_type_signature(nx_g, node, node_attr_names, attrs=[]):
    # Get node type signature
    attrs = set(attrs)
    node_attr_types = []
    for attr_name in node_attr_names:
        if attr_name in nx_g.nodes[node]:
            if attr_name in attrs:
                node_attr_types.append(
                    (attr_name, nx_g.nodes[node][attr_name], True))
            else:
                node_attr_types.append((attr_name, None, False))
        else:
            node_attr_types.append(('', None, False))
    return tuple(sorted(node_attr_types))


def get_edge_type_mapping_signature(nx_g,
                                    e,
                                    node_attr_names,
                                    edge_attr_names,
                                    node_attrs=[],
                                    edge_attrs=[]):
    # Get edge type mapping signature
    edge_attr_types = []
    node_attrs = set(node_attrs)
    edge_attrs = set(edge_attrs)
    src_sig = get_node_type_signature(nx_g, e[0], node_attr_names, node_attrs)
    tgt_sig = get_node_type_signature(nx_g, e[1], node_attr_names, node_attrs)
    for attr_name in edge_attr_names:
        if attr_name in nx_g.edges[e]:
            if attr_name in edge_attrs:
                edge_attr_types.append(
                    (attr_name, nx_g.edges[e][attr_name], True))
            else:
                edge_attr_types.append((attr_name, None, False))
        else:
            edge_attr_types.append(('', None, False))
    edge_type = tuple([tuple(edge_attr_types), src_sig, tgt_sig])
    return edge_type


def infer_node_types(nx_gs, node_attr_names, attrs=[]):
    # Get all possible node types mapping based on the availability
    # of attributes as well as node attributes in attrs
    node_types = set()
    attrs = set(attrs)
    for nx_g in nx_gs:
        for node in nx_g.nodes:
            node_attr_types = get_node_type_signature(nx_g, node,
                                                      node_attr_names, attrs)
            node_types.add(tuple(node_attr_types))
    return sorted(list(node_types))


def infer_edge_types(nx_gs,
                     node_attr_names,
                     edge_attr_names,
                     node_attrs=[],
                     edge_attrs=[]):
    # Get all possible edge types mapping based on the availability
    # of attributes and the nodes' types
    edge_types = set()
    for nx_g in nx_gs:
        for edge in nx_g.edges:
            edge_type = get_edge_type_mapping_signature(
                nx_g, edge, node_attr_names, edge_attr_names, node_attrs,
                edge_attrs)
            edge_types.add(edge_type)
    return sorted(list(edge_types))


def get_node_type_mapping(nx_g, node_attr_names, node_attrs):
    # Get node type mapping
    node_type_mapping = {}
    for node in nx_g.nodes:
        sig = get_node_type_signature(nx_g, node, node_attr_names, node_attrs)
        node_type_mapping[node] = sig
    return node_type_mapping


def get_edge_type_mapping(nx_g,
                          node_attr_names,
                          edge_attr_names,
                          node_attrs=[],
                          edge_attrs=[]):
    # Get edge type mapping
    edge_type_mapping = {}
    for edge in nx_g.edges:
        sig = get_edge_type_mapping_signature(nx_g, edge, node_attr_names,
                                              edge_attr_names, node_attrs,
                                              edge_attrs)
        edge_type_mapping[edge] = sig
    return edge_type_mapping


def get_type_node_mapping(nx_g: nx.Graph, node_attr_names, node_attrs=[]):
    # Get type node mapping
    type_node_mapping = defaultdict(list)
    for node in nx_g.nodes:
        sig = get_node_type_signature(nx_g, node, node_attr_names, node_attrs)
        type_node_mapping[sig].append(node)
    return type_node_mapping


def get_type_edge_mapping(nx_g: nx.Graph,
                          node_attr_names,
                          edge_attr_names,
                          node_attrs=[],
                          edge_attrs=[]):
    # Get type edge mapping
    type_edge_mapping = defaultdict(list)
    for edge in nx_g.edges:
        sig = get_edge_type_mapping_signature(nx_g, edge, node_attr_names,
                                              edge_attr_names, node_attrs,
                                              edge_attrs)
        type_edge_mapping[sig].append(edge)
    return type_edge_mapping


def is_element_in_node_type(node_type,
                            node_attr_val: Dict[str, object]):
    # Check if attributes are listed in the node type with the corresponding value
    # node type is a tuple of (attr_name, attr_val, is_included)
    list_overlap = list(filter(lambda x: x[0] in node_attr_val, node_type))
    if len(list_overlap) != len(node_attr_val):
        return False
    for i, attr in enumerate(list_overlap):
        if node_attr_val[attr[0]] != attr[1]:
            return False
    return True


def retrieve_nodetype_index(node_types, filter_func) -> List[int]:
    # Retrieve node type index based on filter function
    return [
        str(i) for i, node_type in enumerate(node_types)
        if filter_func(node_type)
    ]


def get_meta_data(nx_gs: List[nx.Graph], node_attrs=[], edge_attrs=[]):
    node_attr_names, edge_attr_names = get_all_attr_names(nx_gs)
    node_types = infer_node_types(nx_gs, node_attr_names, node_attrs)
    edge_types = infer_edge_types(nx_gs, node_attr_names, edge_attr_names,
                                  node_attrs, edge_attrs)
    return node_attr_names, edge_attr_names, node_types, edge_types


###### TESTS #####
def test_get_all_attr_names():
    g1 = nx.Graph()
    g1.add_node(0, x=1, y=2)
    g1.add_node(1, x=2, y=3)
    g1.add_edge(0, 1, a=4, b=5)

    g2 = nx.Graph()
    g2.add_node(0, x=1, y=2)
    g2.add_node(1, x=2, y=3)
    g2.add_edge(0, 1, a=4, b=5)

    node_attr_names, edge_attr_names = get_all_attr_names([g1, g2])
    assert set(node_attr_names) == set(['x', 'y'])
    assert set(edge_attr_names) == set(['a', 'b'])


def test_get_node_type_signature():
    g1 = nx.Graph()
    g1.add_node(0, x=1, y=2)
    g1.add_node(1, x=2, y=3)
    g1.add_edge(0, 1, a=4, b=5)

    node_attr_names, edge_attr_names = get_all_attr_names([g1])
    sig = get_node_type_signature(g1, 0, node_attr_names, ['x'])
    assert sig == (('x', 1, True), ('y', None, False))


def test_get_edge_type_mapping_signature():
    g1 = nx.DiGraph()
    g1.add_node(0, x=1, y=2)
    g1.add_node(1, x=2, y=3)
    g1.add_edge(0, 1, a=4, b=5)

    node_attr_names, edge_attr_names = get_all_attr_names([g1])
    sig = get_edge_type_mapping_signature(g1, (0, 1), node_attr_names,
                                          edge_attr_names, ['x'], ['a'])
    assert set(sig) == set(((('x', 1, True), ('y', None, False)),
                            (('x', 2, True), ('y', None, False)),
                            (('a', 4, True), ('b', None, False))))


def test_infer_node_types():
    g1 = nx.Graph()
    g1.add_node(0, x=1, y=2)
    g1.add_node(1, x=2, y=3)
    g1.add_edge(0, 1, a=4, b=5)

    g2 = nx.Graph()
    g2.add_node(0, x=1, y=2)
    g2.add_node(1, x=2, y=3)
    g2.add_edge(0, 1, a=4, b=5)

    node_attr_names, edge_attr_names = get_all_attr_names([g1, g2])
    node_types = infer_node_types([g1, g2], node_attr_names, ['x'])
    assert node_types == {(('x', 1, True), ('y', None, False)),
                          (('x', 2, True), ('y', None, False))}


def test_infer_edge_types():
    g1 = nx.Graph()
    g1.add_node(0, x=1, y=2)
    g1.add_node(1, x=2, y=3)
    g1.add_edge(0, 1, a=4, b=5)

    g2 = nx.Graph()
    g2.add_node(0, x=1, y=2)
    g2.add_node(1, x=2, y=3)
    g2.add_edge(0, 1, a=4, b=5)

    node_attr_names, edge_attr_names = get_all_attr_names([g1, g2])
    edge_types = infer_edge_types([g1, g2], node_attr_names, edge_attr_names,
                                  ['x'], ['a'])
    assert edge_types == {((('a', 4, True), ('b', None, False)),
                           (('x', 1, True), ('y', None, False)),
                           (('x', 2, True), ('y', None, False)))}


def test_is_element_in_node_type():
    node_type = (('x', 1, True), ('y', None, False))
    node_attr_val = {'x': 1, 'y': 2}
    assert is_element_in_node_type(node_type, node_attr_val)


def test_retrieve_nodetype_index():
    node_types = [
        (('x', 1, True), ('y', None, False)),
        (('x', 2, True), ('y', None, False)),
        (('x', 3, True), ('y', None, False)),
        (('x', 4, True), ('y', None, False)),
    ]
    filter_func = lambda node_type: node_type[0][1] == 2
    assert retrieve_nodetype_index(node_types, filter_func) == ['1']


def test_get_type_node_mapping():
    g1 = nx.Graph()
    g1.add_node(0, x=1, y=2)
    g1.add_node(1, x=2, y=3)
    g1.add_edge(0, 1, a=4, b=5)

    g2 = nx.Graph()
    g2.add_node(0, x=1, y=2)
    g2.add_node(1, x=2, y=3)
    g2.add_edge(0, 1, a=4, b=5)

    node_attr_names, edge_attr_names = get_all_attr_names([g1, g2])
    node_types = infer_node_types([g1, g2], node_attr_names, ['x'])
    type_node_mapping = get_type_node_mapping([g1, g2], node_attr_names, ['x'])
    assert type_node_mapping == {
        (('x', 1), ('y', None, False)): [0, 1],
        (('x', 2), ('y', None, False)): [0, 1]
    }


def test_get_type_edge_mapping():
    g1 = nx.Graph()
    g1.add_node(0, x=1, y=2)
    g1.add_node(1, x=2, y=3)
    g1.add_edge(0, 1, a=4, b=5)

    g2 = nx.Graph()
    g2.add_node(0, x=1, y=2)
    g2.add_node(1, x=2, y=3)
    g2.add_edge(0, 1, a=4, b=5)

    node_attr_names, edge_attr_names = get_all_attr_names([g1, g2])
    node_types = infer_node_types([g1, g2], node_attr_names, ['x'])
    edge_types = infer_edge_types([g1, g2], node_attr_names, edge_attr_names,
                                  ['x'], ['a', 'b'])
    type_edge_mapping = get_type_edge_mapping(g1, node_attr_names,
                                              edge_attr_names, ['x'], ['a'])
    sig = ((('a', 4, True), ('b', None, False)),
           (('x', 1, True), ('y', None, False)), (('x', 2, True), ('y', None,
                                                                   False)))
    assert len(type_edge_mapping[sig]) == 1, "Wrong len"
    assert type_edge_mapping[sig][0] == (0, 1)


def test_get_type_node_mapping():
    g1 = nx.DiGraph()
    g1.add_node(0, x=1, y=2)
    g1.add_node(1, x=2, y=3)
    g1.add_edge(0, 1, a=4, b=5)

    g2 = nx.Graph()
    g2.add_node(0, x=1, y=2)
    g2.add_node(1, x=2, y=3)
    g2.add_edge(0, 1, a=4, b=5)

    node_attr_names, edge_attr_names = get_all_attr_names([g1, g2])
    type_node_mapping = get_type_node_mapping(g1, node_attr_names, ['x'])
    assert type_node_mapping == {
        (('x', 1, True), ('y', None, False)): [0],
        (('x', 2, True), ('y', None, False)): [1]
    }


if __name__ == '__main__':
    # Load each test, print the name and run test
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and name.startswith('test_'):
            print(name)
            obj()
            print("Success")
