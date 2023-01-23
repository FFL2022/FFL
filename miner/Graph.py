
# Graph.py
#
# Written by Larry Holder (holder@wsu.edu).
#
# Copyright (c) 2017-2021. Washington State University.

import json

# The Graph class allows the representation of an attributed, mixed multi-graph with time stamps on nodes
# and edges. A graph has an id and a className (for now, either "positive" or "negative"). Each node has
# an id and a timestamp, along with other user-defined attrs. Each edge has an id, src, tgt,
# directed, and timestamp, along with other user-defined attrs.
# Note: Assumes time stamps are integers.
class Graph:

    def __init__(self):
        self.vs = {}
        self.es = {}

    def size(self):
        return len(self.vs), len(self.es)

    def Compress(self,iteration,pattern):
        """Compress graph using given pattern at given iteration. Replaces each instance of pattern with a new
           vertex, and reconnects edges incident on the instance to the new vertex. Assumes no overlap among instances."""
        instanceNum = 0
        for inst in pattern.instances:
            instanceNum += 1
            # Create and add new vertex representing pattern instance
            n_vl = 'PATTERN-' + str(iteration)
            n_vid = 'PATTERN-' + str(iteration) + '-' + str(instanceNum)
            n_v = Vertex(n_vid)
            n_v.timestamp = inst.max_timestamp()
            n_v.add_attribute('label', n_vl)
            self.vs[n_vid] = n_v
            # Remove instance's edges from graph and from src/tgt vertex edge lists
            for inst_e in inst.es:
                inst_e.src.es.remove(inst_e)
                inst_e.tgt.es.remove(inst_e)
                del self.es[inst_e.id]
            # Remove instance's vertices from graph; remaining edges incident on this vertex should be made incident on n_v
            for v in inst.vs:
                for e in v.es:
                    if e.src == v:
                        e.src = n_v
                    if e.tgt == v:
                        e.tgt = n_v
                    if e not in n_v.es:
                        n_v.es.append(e)
                del self.vs[v.id]

    def TemporalOrder(self):
        """Set the temporal property of vertices and edges according to their order of arrival."""
        # Collect and sort all unique timestamps in graph
        timestamps = []
        for v in self.vs.values():
            if v.timestamp not in timestamps:
                timestamps.append(v.timestamp)
        for e in self.es.values():
            if e.timestamp not in timestamps:
                timestamps.append(e.timestamp)
        timestamps.sort()
         # Set temporal property based on order of timestamp
        for v in self.vs.values():
            v.temporal = timestamps.index(v.timestamp)
        for e in self.es.values():
            e.temporal = timestamps.index(e.timestamp)

    # Load graph from given JSON array of vertices and edges.
    # ***** todo: Read graph from stream, rather than all at once
    def from_json (self, g_json):
        # Initialize graph (just in case it's being reused)
        self.vs = {}
        self.es = {}
        for json_object in g_json:
            if ('vertex' in json_object):
                v_dict = json_object['vertex']
                v_id = v_dict['id']
                # in case fused graph with duplicate vertices
                if (v_id not in self.vs):
                    v = Vertex(v_id)
                    if ('timestamp' in v_dict):
                        v.timestamp = int(v_dict['timestamp'])
                    if ('attrs' in v_dict):
                        json_attrs = v_dict['attrs']
                        for attr, val in json_attrs.items():
                            v.add_attribute(attr, val)
                    self.vs[v_id] = v
            if ('edge' in json_object):
                e_dict = json_object['edge']
                print(e_dict)
                e_id, s_id, t_id = e_dict['id'], e_dict['src'], e_dict['tgt']
                s_v, t_v = self.vs[s_id], self.vs[t_id]
                directed = False

                if (e_dict['directed'] == 'true'):
                    directed = True
                e = Edge(e_id, s_v, t_v, directed)
                if ('timestamp' in e_dict):
                    e.timestamp = int(e_dict['timestamp'])
                if ('attrs' in e_dict):
                    json_attrs = e_dict['attrs']
                    for attr, val in json_attrs.items():
                        e.add_attribute(attr, val)
                self.es[e_id] = e
                s_v.add_edge(e)
                t_v.add_edge(e)

    def from_nx(
        self,
        networkx_graph,
        n_attrs=None,
        e_attrs=None,
    ):
        # helper routine
        def select_attrs_from_dict(dict_, attrs=None):
            if attrs is None:
                return dict_
            else:
                return {attr: dict_[attr] for attr in attrs}

        directed = networkx_graph.is_directed()
        subdue_format = list()
        for n_id in networkx_graph.nodes:
            subdue_format.append({
                'vertex': {
                    'id': n_id,
                    'attrs': select_attrs_from_dict(networkx_graph.nodes[n_id], n_attrs).copy(),
                }
            })
        for (u, v, e) in networkx_graph.edges(data=True):
            subdue_format.append({
                'edge': {
                    'id': f'{u}-{v}',
                    'src': u,
                    'tgt': v,
                    'directed': directed,
                    'attrs': select_attrs_from_dict(e, e_attrs).copy(),
                }
            })

        self.from_json(subdue_format)

    def write_to_dot(self, outputFileName):
        """Write graph to given file name in DOT format."""
        outputFile = open(outputFileName, 'w')
        outputFile.write('digraph {\n')
        for vertex in self.vs.values():
            labelStr = str(vertex.id)
            if ('label' in vertex.attrs):
                labelStr = str(vertex.attrs['label'])
            outputFile.write(str(vertex.id) + ' [label=' + labelStr + '];\n')
        for edge in self.es.values():
            labelStr = str(edge.id)
            if ('label' in edge.attrs):
                labelStr = str(edge.attrs['label'])
            outputStr = str(edge.src.id) + ' -> ' + str(edge.tgt.id) + ' [label=' + labelStr
            if (not edge.directed):
                outputStr += ',dir=none'
            outputStr += '];\n'
            outputFile.write(outputStr)
        outputFile.write('}\n')
        outputFile.close()

    def write_to_file(self, outputFileName):
        """Write graph to given file name in JSON format."""
        outputFile = open(outputFileName, 'w')
        outputFile.write('[\n')
        firstOne = True
        for vertex in self.vs.values():
            if firstOne:
                firstOne = False
            else:
                outputFile.write(',\n')
            vertex.write_to_file(outputFile)
        outputFile.write(',\n')
        firstOne = True
        for edge in self.es.values():
            if firstOne:
                firstOne = False
            else:
                outputFile.write(',\n')
            edge.write_to_file(outputFile)
        outputFile.write('\n]\n')
        outputFile.close()

    def print_graph(self, tab=""):
        print(tab + "Graph:")
        for vertex in self.vs.values():
            vertex.print_vertex(tab+'  ')
        for edge in self.es.values():
            edge.print_edge(tab+'  ')


class Vertex:

    def __init__(self, id):
        self.id = id # must be unique for each vertex
        self.timestamp = 0
        self.temporal = 0 # used to set arrival order of vertex internally for graph matcher
        self.attrs = {}
        self.es = []

    def add_attribute(self, key, value):
        self.attrs[key] = value

    def add_edge(self, e):
        self.es.append(e)

    def print_vertex(self, tab=""):
        attr = ""
        for key,value in self.attrs.items():
            attr += ', ' + key + '=' + str(value)
        print('{tab}vertex {self.id}: timestamp={self.timestamp} {attr}')

    def write_to_file(self, outputFile):
        """Write vertex to given file stream in JSON format"""
        outputFile.write('  {"vertex": {\n')
        outputFile.write('     "id": "' + self.id + '",\n')
        outputFile.write('     "attrs": {')
        firstOne = True
        for key,value in self.attrs.items():
            if firstOne:
                firstOne = False
            else:
                outputFile.write(',')
            outputFile.write('"' + key + '": "' + value + '"')
        outputFile.write('},\n')
        outputFile.write('     "timestamp": "' + str(self.timestamp) + '"}}')

class Edge:

    def __init__(self, id, src, tgt, directed = False):
        self.id = id # must be unique for each edge
        self.src = src
        self.tgt = tgt
        self.directed = directed
        self.timestamp = 0
        self.temporal = 0 # used to set arrival order of edge internally for graph matcher
        self.attrs = {}

    def add_attribute(self, key, value):
        self.attrs[key] = value

    def print_edge(self, tab=""):
        attributeString = ""
        for key,value in self.attrs.items():
            attributeString += ', ' + key + '=' + str(value)
        edgeString = str(self.src.id)
        if self.directed:
            edgeString += '->'
        else:
            edgeString += '--'
        edgeString += str(self.tgt.id)
        print(tab + 'edge "' + self.id + '" (' + edgeString + '): timestamp=' + str(self.timestamp) + attributeString)

    def write_to_file(self, outputFile):
        """Write edge to given file stream in JSON format"""
        outputFile.write('  {"edge": {\n')
        outputFile.write('     "id": "' + self.id + '",\n')
        outputFile.write('     "src": "' + self.src.id + '",\n')
        outputFile.write('     "tgt": "' + self.tgt.id + '",\n')
        outputFile.write('     "attrs": {')
        firstOne = True
        for key,value in self.attrs.items():
            if firstOne:
                firstOne = False
            else:
                outputFile.write(',')
            outputFile.write('"' + key + '": "' + value + '"')
        outputFile.write('},\n')
        if self.directed:
            outputFile.write('     "directed": "true",\n')
        else:
            outputFile.write('     "directed": "false",\n')
        outputFile.write('     "timestamp": "' + str(self.timestamp) + '"}}')


# ----- Graph matcher

# New in version 1.2: poly-time-bounded graph matcher

gMaxMappings = 1 # Will be set to E^2 for each match

def match_g(graph1, graph2):
    """Returns True if given graphs are isomorphic.
    This is a poly-time, approximate version of graph isomorphism."""
    global gMaxMappings
    if (len(graph1.vs) != len(graph2.vs)):
        return False
    if (len(graph1.es) != len(graph2.es)):
        return False
    if (len(graph1.es) == 0):
        v1keys = list(graph1.vs.keys())
        v2keys = list(graph2.vs.keys())
        return MatchVertex(graph1, graph2, v1keys[0], v2keys[0])
    gMaxMappings = len(graph1.es) ** 2 # Limit search to E^2 mappings
    matchFound, numMappings = ExtendMapping(graph1, graph2)
    return matchFound

def ExtendMapping(graph1, graph2, mapping=None, numMappings=0):
    """Find the next unmapped edge in graph1 and try mapping it to each unmapped edge in graph2.
    Constrain number of mappings to be at most gMaxMappings. Return the match result and
    number of mappings so far."""
    global gMaxMappings
    if mapping is None:
        mapping = {}
    if (len(mapping) == len(graph1.es)):
        return True, numMappings
    if numMappings > gMaxMappings:
        return False, numMappings
    # Find unmapped edge in graph1 (should always exist at this point)
    e_id1 = None
    for e_id in graph1.es:
        if (not (e_id in mapping)):
            e_id1 = e_id
            break
    # Find unmapped, matching edge in graph2
    for e_id2 in graph2.es:
        if not (e_id2 in mapping.values()):
            if match_e(graph1, graph2, e_id1, e_id2, mapping):
                # Extend mapping
                mapping[e_id1] = e_id2
                matchFound, numMappings = ExtendMapping(graph1, graph2, mapping, numMappings + 1)
                if matchFound:
                    return True, numMappings
                mapping.pop(e_id1)
    return False, numMappings

def match_g_Orig(graph1, graph2):
    """Returns True if given graphs are isomorphic.
    This is a correct, non-approximate version of graph isomorphism."""
    if (len(graph1.vs) != len(graph2.vs)):
        return False
    if (len(graph1.es) != len(graph2.es)):
        return False
    if (len(graph1.es) == 0):
        v1keys = list(graph1.vs.keys())
        v2keys = list(graph2.vs.keys())
        return MatchVertex(graph1, graph2, v1keys[0], v2keys[0])
    return ExtendMapping_Orig(graph1, graph2)

def ExtendMapping_Orig(graph1, graph2, mapping=None):
    """Find the next unmapped edge in graph1 and try mapping it to each unmapped edge in graph2.
    Return True if leads to a match, else False."""
    if mapping is None:
        mapping = {}
    if (len(mapping) == len(graph1.es)):
        return True
    # Find unmapped edge in graph1 (should always exist at this point)
    e_id1 = None
    for e_id in graph1.es:
        if (not (e_id in mapping)):
            e_id1 = e_id
            break
    # Find unmapped, matching edge in graph2
    for e_id2 in graph2.es:
        if not (e_id2 in mapping.values()):
            if match_e(graph1,graph2,e_id1,e_id2,mapping):
                # Extend mapping
                mapping[e_id1] = e_id2
                if ExtendMapping_Orig(graph1,graph2,mapping):
                    return True
                mapping.pop(e_id1)
    return False

def match_e(graph1, graph2, e_id1, e_id2, mapping):
    """Return True if edges, corresponding to given edge IDs in given graphs, match;
    i.e., have same attrs, direction, temporal ordering, and src/tgt vertices."""
    edge1 = graph1.es[e_id1]
    edge2 = graph2.es[e_id2]
    if not (edge1.attrs == edge2.attrs):
        return False
    if (edge1.directed != edge2.directed):
        return False
    # Temporal field only set if discovering temporal patterns; otherwise, this test always True
    if (edge1.temporal != edge2.temporal):
        return False
    if (MatchVertex(graph1, graph2, edge1.src.id, edge2.src.id) and
        MatchVertex(graph1, graph2, edge1.tgt.id, edge2.tgt.id)):
        return True
    if ((not edge1.directed) and MatchVertex(graph1, graph2, edge1.src.id, edge2.tgt.id) and
        MatchVertex(graph1, graph2, edge1.tgt.id, edge2.src.id)):
        return True
    return False

def MatchVertex(graph1, graph2, v_id1, v_id2):
    """Returns True if vertices, corresponding to given vertex IDs in given graphs, match;
       i.e., same attrs and temporal edges."""
    # First check for same attrs
    v1 = graph1.vs[v_id1]
    v2 = graph2.vs[v_id2]
    if not (v1.attrs == v2.attrs):
        return False
    if (len(v1.es) != len(v2.es)):
        return False
    # Temporal field only set if discovering temporal patterns; otherwise, this test always True
    if (v1.temporal != v2.temporal):
        return False
    return True


# ----- Graph Creation

def edge2graph(edge):
    """Create a generic one-edge graph with the same properties as the given edge, but with new vertex/edge IDs."""
    g = Graph()
    src = Vertex("1")
    g.vs["1"] = src
    src.timestamp = edge.src.timestamp
    src.attrs = edge.src.attrs
    tgt = Vertex("2")
    g.vs["2"] = tgt
    tgt.timestamp = edge.tgt.timestamp
    tgt.attrs = edge.tgt.attrs
    e = Edge("1", src, tgt, edge.directed)
    g.es["1"] = e
    e.timestamp = edge.timestamp
    e.attrs = edge.attrs
    src.es.append(e)
    tgt.es.append(e)
    return g

def inst2g(inst):
    """Create graph with same properties and isomorphic to given instance,
       but with new vertex/edge IDs."""
    g = Graph()
    # Add vertices
    v_id = 1
    v_map = {}
    for v in inst.vs:
        n_v = Vertex(str(v_id))
        n_v.timestamp = v.timestamp
        n_v.attrs = v.attrs
        g.vs[n_v.id] = n_v
        v_map[v.id] = n_v
        v_id += 1
    # Add edges
    e_id = 1
    for edge in inst.es:
        src = v_map[edge.src.id]
        tgt = v_map[edge.tgt.id]
        n_e = Edge(str(e_id), src, tgt, edge.directed)
        n_e.timestamp = edge.timestamp
        n_e.attrs = edge.attrs
        g.es[n_e.id] = n_e
        src.es.append(n_e)
        tgt.es.append(n_e)
        e_id += 1
    return g
