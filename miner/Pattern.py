# Pattern.py
#
# Written by Larry Holder (holder@wsu.edu).
#
# Copyright (c) 2017-2021. Washington State University.

from miner.OrderedSet import OrderedSet # specialized Subdue version
from miner import Graph

class Pattern:

    def __init__(self):
        self.definition = None # Graph
        self.insts = []
        self.value = 0.0

    def evaluate (self, graph):
        """Compute value of using given pattern to compress given graph, where 0 means no compression, and 1 means perfect compression."""
        # (instances-1) because we would also need to retain the definition of the pattern for compression
        self.value = float(((len(self.insts) - 1) * len(self.definition.es)) / float(len(graph.es)))

    def print_pattern(self, tab):
        print(tab + "Pattern (value=" + str(self.value) + ", instances=" + str(len(self.insts)) + "):")
        self.definition.print_graph(tab+'  ')
        # May want to make instance printing optional
        instanceNum = 1
        for instance in self.insts:
            instance.print_instance(instanceNum, tab+'  ')
            instanceNum += 1

    def write_instances_to_file(self, out_fn):
        """Write instances of pattern to given file name in JSON format."""
        outputFile = open(out_fn, 'w')
        outputFile.write('[\n')
        firstOne = True
        for instance in self.insts:
            if firstOne:
                firstOne = False
            else:
                outputFile.write(',\n')
            instance.write_to_file(outputFile)
        outputFile.write('\n]\n')
        outputFile.close()

class Instance:

    def __init__(self):
        self.vs = OrderedSet()
        self.es = OrderedSet()

    def print_instance (self, instanceNum, tab=""):
        print(tab + "Instance " + str(instanceNum) + ":")
        for vertex in self.vs:
            vertex.print_vertex(tab+'  ')
        for edge in self.es:
            edge.print_edge(tab+'  ')

    def write_to_file(self, outputFile):
        """Write instance to given file stream in JSON format."""
        firstOne = True
        for vertex in self.vs:
            if firstOne:
                firstOne = False
            else:
                outputFile.write(',\n')
            vertex.write_to_file(outputFile)
        outputFile.write(',\n')
        firstOne = True
        for edge in self.es:
            if firstOne:
                firstOne = False
            else:
                outputFile.write(',\n')
            edge.write_to_file(outputFile)

    def max_timestamp(self):
        """Returns the maximum timestamp over all vertices and edges in the instance."""
        maxTimeStampVertex = max(self.vs, key = lambda v: v.timestamp)
        maxTimeStampEdge = max(self.es, key = lambda e: e.timestamp)
        return max(maxTimeStampVertex.timestamp, maxTimeStampEdge.timestamp)


# ----- Pattern and Instance Creation

def edge2inst(edge):
    i = Instance()
    i.es.add(edge)
    i.vs.add(edge.src)
    i.vs.add(edge.tgt)
    return i

def inst2ptrn(definition, instances):
    """Create pattern from given definition graph and its instances. Note: Pattern not evaluated here."""
    pattern = Pattern()
    pattern.definition = definition
    pattern.insts = instances
    return pattern

# ----- Pattern Extension

def extend_ptrn (parameters, pattern):
    """Return list of patterns created by extending each instance of the given
       pattern by one edge in all possible ways, and then collecting matching
       extended instances together into new patterns."""
    # TODO: Extend pattern - all edge labels
    e_insts = []
    # For each instance of this pattern
    for inst in pattern.insts:
        # Extend it according to new instance
        n_insts = ExtendInstance(inst)
        for n_inst in n_insts:
            insert_new_instance(e_insts, n_inst)
    n_ptrns = []
    while e_insts:      # While len > 0
        n_inst = e_insts.pop(0)
        n_ig = Graph.inst2g(n_inst)
        if parameters.temporal:
            n_ig.TemporalOrder()
        m_insts = [n_inst]          # Matching instances
        nm_insts = []
        for e_inst in e_insts:
            e_ig = Graph.inst2g(e_inst)
            if parameters.temporal:
                e_ig.TemporalOrder()
            # Graph match
            if Graph.match_g(n_ig, e_ig) and\
                    (not InstancesOverlap(parameters.overlap, m_insts, e_inst)):
                m_insts.append(e_inst)
            else:
                nm_insts.append(e_inst)
        e_insts = nm_insts
        n_ptrn = inst2ptrn(n_ig, m_insts)
        n_ptrns.append(n_ptrn)
    return n_ptrns

def ExtendInstance (instance):
    """Returns list of new instances created by extending the given instance by one new edge in all possible ways."""
    newInstances = []
    unusedEdges = OrderedSet([e for v in instance.vs for e in v.es]) - instance.es
    for edge in unusedEdges:
        newInstance = ExtendInstanceByEdge(instance, edge)
        newInstances.append(newInstance)
    return newInstances

def ExtendInstanceByEdge(instance, edge):
    """Create and return new instance built from given instance and adding given edge and vertices of edge if new."""
    newInstance = Instance()
    newInstance.vs = OrderedSet(instance.vs)
    newInstance.es = OrderedSet(instance.es)
    newInstance.es.add(edge)
    newInstance.vs.add(edge.src)
    newInstance.vs.add(edge.tgt)
    return newInstance

def insert_new_instance(insts, n_inst):
    """Add newInstance to instanceList if it does not match an instance already on the list."""
    # python: any does short circuit
    match = any(match_i(inst, n_inst) for inst in insts)
    if not match:
        insts.append(n_inst)

def match_i(instance1,instance2):
    """Return True if given instances match, i.e., contain the same vertex and edge object instances."""
    return (instance1.vs == instance2.vs) and (instance1.es == instance2.es)

def InstancesOverlap(overlap, instanceList, instance):
    """Returns True if instance overlaps with an instance in the given instanceList
    according to the overlap parameter, which indicates what type of overlap ignored.
    Overlap="none" means no overlap ignored. Overlap="vertex" means vertex overlap
    ignored. Overlap="edge" means vertex and edge overlap ignored, but the instances
    cannot be identical."""
    for instance2 in instanceList:
        if InstanceOverlap(overlap, instance, instance2):
            return True
    return False

def InstanceOverlap(overlap, instance1, instance2):
    """Returns True if given instances overlap according to given overlap parameter.
    See InstancesOverlap for explanation."""
    if overlap == "edge":
        return match_i(instance1, instance2)
    elif overlap == "vertex":
        return instance1.es.intersect(instance2.es)
    else: # overlap == "none"
        return instance1.vs.intersect(instance2.vs)


# ----- Pattern List Operations

def insert_ptrn(newPattern, patternList, maxLength, valueBased):
    """Insert newPattern into patternList. If newPattern is isomorphic to an existing pattern on patternList, then keep higher-valued
       pattern. The list is kept in decreasing order by pattern value. If valueBased=True, then maxLength represents the maximum number
       of different-valued patterns on the list; otherwise, maxLength represents the maximum number of patterns on the list.
       Assumes given patternList already conforms to maximums."""
    # Check if newPattern unique (i.e., non-isomorphic or isomorphic but better-valued)
    for pattern in patternList:
        if (Graph.match_g(pattern.definition ,newPattern.definition)):
            if (pattern.value >= newPattern.value):
                return # newPattern already on list with same or better value
            else:
                # newpattern isomorphic to existing pattern, but better valued
                patternList.remove(pattern)
                break
    # newPattern unique, so insert in order by value
    insertAtIndex = 0
    for pattern in patternList:
        if newPattern.value > pattern.value:
            break
        insertAtIndex += 1
    patternList.insert(insertAtIndex, newPattern)
    # check if patternList needs to be trimmed
    if valueBased:
        uniqueValues = UniqueValues(patternList)
        if len(uniqueValues) > maxLength:
            removeValue = uniqueValues[-1]
            while (patternList[-1].value == removeValue):
                patternList.pop(-1)
    else:
        if len(patternList) > maxLength:
            patternList.pop(-1)

def UniqueValues(patternList):
    """Returns list of unique values of patterns in given pattern list, in same order."""
    uniqueValues = []
    for pattern in patternList:
        if pattern.value not in uniqueValues:
            uniqueValues.append(pattern.value)
    return uniqueValues
