# Subdue.py
#
# Written by Larry Holder (holder@wsu.edu).
#
# Copyright (c) 2017-2021. Washington State University.

import sys
import time
import json
import contextlib
import miner.Parameters as Parameters
import miner.Graph as Graph
import miner.Pattern as Pattern
import functools

DEBUGFLAG = False

# ***** todos: read graph file incrementally


def read_graph(fp):
    """Read graph from given filename."""
    g_json = json.load(open(fp))
    graph = Graph.Graph()
    graph.from_json(g_json)
    return graph


def discover_ptrns(params, graph):
    """The main discovery loop. Finds and returns best patterns in given
    graph."""
    n_ptrns = 0
    # get initial one-edge patterns
    l_p_ptrns = init_ptrns(params, graph)
    if DEBUGFLAG:
        print("Initial patterns (" + str(len(l_p_ptrns)) + "):")
        for pattern in l_p_ptrns:
            pattern.print_pattern('  ')
    l_disc_ptrns = []
    while ((n_ptrns < params.limit) and l_p_ptrns):
        n_ptrn_remaining = params.limit - n_ptrns
        print(f"{n_ptrn_remaining} patterns left\n")
        l_child_ptrns = []
        # extend each pattern in parent list (***** todo: in parallel)
        while (l_p_ptrns):
            p_ptrn = l_p_ptrns.pop(0)
            if ((len(p_ptrn.insts) > 1) and (n_ptrns < params.limit)):
                n_ptrns += 1
                # Pattern: extend the pattern
                l_ptrn_extend = Pattern.extend_ptrn(params, p_ptrn)
                while (l_ptrn_extend):
                    e_ptrn = l_ptrn_extend.pop(0)
                    if DEBUGFLAG:
                        print("Extended Pattern:")
                        e_ptrn.print_pattern('  ')
                    if (len(e_ptrn.definition.es) <= params.max_size):
                        # evaluate each extension and add to child list
                        e_ptrn.evaluate(graph)
                        if ((not params.prune) or (e_ptrn.value >= p_ptrn.value)):
                            Pattern.insert_ptrn(e_ptrn, l_child_ptrns, params.k_beam, params.valueBased)
            # add parent pattern to final discovered list
            if (len(p_ptrn.definition.es) >= params.min_size):
                Pattern.insert_ptrn(p_ptrn, l_disc_ptrns, params.n_best,
                                    False)  # valueBased = False
        l_p_ptrns = l_child_ptrns
        if not l_p_ptrns:
            print("No more patterns to consider", flush=True)
    # insert any remaining patterns in parent list on to discovered list
    while (l_p_ptrns):
        p_ptrn = l_p_ptrns.pop(0)
        if (len(p_ptrn.definition.es) >= params.min_size):
            Pattern.insert_ptrn(p_ptrn, l_disc_ptrns, params.n_best, False) # valueBased = False
    return l_disc_ptrns

def init_ptrns(params, graph):
    """Returns list of single-edge, evaluated patterns in given graph with more than one instance."""
    l_ptrns_init = []
    # Create a graph and an instance for each edge
    pairs = []
    for edge in graph.es.values():
        g1 = Graph.edge2graph(edge)
        if params.temporal:
            g1.TemporalOrder()
        inst1 = Pattern.edge2inst(edge)
        pairs.append((g1, inst1))
    while len(pairs):
        g1, inst1 = pairs.pop(0)
        ptrn = Pattern.Pattern()
        ptrn.definition = g1
        ptrn.insts.append(inst1)
        non_match_pairs = []
        for g2, inst2 in pairs:
            if Graph.match_g(g1, g2) and (not Pattern.InstancesOverlap(
                    params.overlap, ptrn.insts, inst2)):
                ptrn.insts.append(inst2)
            else:
                non_match_pairs.append((g2, inst2))
        if len(ptrn.insts) > 1:
            ptrn.evaluate(graph)
            l_ptrns_init.append(ptrn)
        pairs = non_match_pairs
    return l_ptrns_init

def Subdue(params, graph):
    """
    Top-level function for Subdue that discovers best pattern in graph.
    Optionally, Subdue can then compress the graph with the best pattern, and iterate.

    :param graph: instance of Subdue.Graph
    :param params: instance of Subdue.Parameters
    :return: patterns for each iteration -- a list of iterations each containing discovered patterns.
    """
    t_start = time.time()
    it = 1
    done = False
    ptrns = list()
    while ((it <= params.n_iters) and (not done)):
        t_it_start = time.time()
        if (it > 1):
            print(f"----- Iteration {it} -----\n")
        n_vs, n_es = graph.size()
        print(f"Graph: {n_vs} vertices, {n_es} edges")
        l_disc_ptrns = discover_ptrns(params, graph)
        if (not l_disc_ptrns):
            done = True
            print("No patterns found.\n")
        else:
            ptrns.append(l_disc_ptrns)
            print("\nBest " + str(len(l_disc_ptrns)) + " patterns:\n")
            for ptrn in l_disc_ptrns:
                ptrn.print_pattern('  ')
                print("")
            # write machine-readable output, if requested
            if (params.writePattern):
                out_fn = f"{params.out_fn}-pattern-{it}.json"
                l_disc_ptrns[0].definition.write_to_file(out_fn)
            if (params.writeInstances):
                out_fn = f"{params.out_fn}-instances-{it}.json"
                l_disc_ptrns[0].write_instances_to_file(out_fn)
            if ((it < params.n_iters) or (params.writeCompressed)):
                graph.Compress(it, l_disc_ptrns[0])
            if (it < params.n_iters):
                # consider another iteration
                if (len(graph.es) == 0):
                    done = True
                    print("Ending iterations - graph fully compressed.\n")
            if ((it == params.n_iters) and (params.writeCompressed)):
                out_fn = f"{params.out_fn}-compressed-{it}.json"
                graph.write_to_file(out_fn)
        if (params.n_iters > 1):
             t_it_delt = time.time() - t_it_start
             print(f"Elapsed time for iteration {it} = {t_it_delt} seconds.\n")
        it += 1
    t_end = time.time()
    print("SUBDUE done. Elapsed time = " + str(t_end - t_start) + " seconds\n")
    return ptrns


def nx_subdue(graph, v_attribs=None, e_attribs=None, verbose=False,
              **subdue_params
              ):
    """
    :param graph: networkx.Graph
    :param v_attribs: (Default: None)   -- attributes on the nodes to use for pattern matching, use `None` for all
    :param e_attribs: (Default: None)   -- attributes on the edges to use for pattern matching, use `None` for all
    :param verbose: (Default: False)          -- if True, print progress, as well as report each found pattern

    :param k_beam: (Default: 4)            -- Number of patterns to retain after each expansion of previous patterns; based on value.
    :param iterations: (Default: 1)           -- Iterations of Subdue's discovery process. If more than 1, Subdue compresses graph with best pattern before next run. If 0, then run until no more compression (i.e., set to |E|).
    :param limit: (Default: 0)                -- Number of patterns considered; default (0) is |E|/2.
    :param max_size: (Default: 0)              -- Maximum size (#edges) of a pattern; default (0) is |E|/2.
    :param min_size: (Default: 1)              -- Minimum size (#edges) of a pattern; default is 1.
    :param numBest: (Default: 3)              -- Number of best patterns to report at end; default is 3.
    :param overlap: (Defaul: none)            -- Extent that pattern instances can overlap (none, vertex, edge)
    :param prune: (Default: False)            -- Remove any patterns that are worse than their parent.
    :param valueBased: (Default: False)       -- Retain all patterns with the top beam best values.
    :param temporal: (Default: False)         -- Discover static (False) or temporal (True) patterns

    :return: list of patterns, where each pattern is a list of pattern instances, with an instance being a dictionary
    containing
        `nodes` -- list of IDs, which can be used with `networkx.Graph.subgraph()`
        `edges` -- list of tuples (id_from, id_to), which can be used with `networkx.Graph.edge_subgraph()`

    For `iterations`>1 the the list is split by iterations, and some patterns will contain node IDs not present in
    the original graph, e.g. `PATTERN-X-Y`, such node ID refers to a previously compressed pattern, and it can be
    accessed as `output[X-1][0][Y]`.

    """
    params = Parameters.Parameters()
    if len(subdue_params) > 0:
        params.set_params_from_kwargs(**subdue_params)
    subdue_g = Graph.Graph()
    subdue_g.from_nx(graph, v_attribs, e_attribs)
    params.set_defaults_for_graph(subdue_g)
    if verbose:
        l_ptrns_per_iters = Subdue(params, subdue_g)
    else:
        with contextlib.redirect_stdout(None):
            l_ptrns_per_iters = Subdue(params, subdue_g)
    l_ptrns = unwrap_output(l_ptrns_per_iters)
    if params.n_iters == 1:
        if len(l_ptrns) == 0:
            return None
        return l_ptrns[0]
    else:
        return l_ptrns


def unwrap_output(l_ptrns_iters):
    """
    Subroutine of `nx_Subdue` -- unwraps the standard Subdue output into pure
    python objects compatible with networkx
    """
    out = list()
    for l_ptrns_iter in l_ptrns_iters:
        iter_out = []
        for ptrn in l_ptrns_iter:
            iter_out.append([{
                'nodes': [v.id for v in inst.vs],
                'edges': [(e.src.id, e.tgt.id) for e in inst.es]
            } for inst in ptrn.insts])
        out.append(iter_out)
    return out


def main():
    print("SUBDUE v1.4 (python)\n")
    params = Parameters.Parameters()
    params.set_params(sys.argv)
    graph = read_graph(params.in_fp)
    #out_fn = params.out_fn + ".dot"
    #graph.write_to_dot(out_fn)
    params.set_defaults_for_graph(graph)
    params.print()
    Subdue(params, graph)

if __name__ == "__main__":
    main()
