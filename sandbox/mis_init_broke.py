import cupy as cp
import numpy as np
import pykokkos as pk

import argparse
import random

import networkx as nx
import scipy as sp
import scipy.io
import io
import sys

@pk.functor
class Workload:
    def __init__(self, indices: pk.View1D[int], indptr: pk.View1D[int]):
        self.size: int = len(indptr) - 1

        # init variable list
        self.done: pk.View1D[int] = pk.View([1], int)
        self.priority: pk.View1D[int] = pk.View([self.size], int)
        self.status: pk.View1D[int] = pk.View([self.size], int)

        # graph info needed (CSR minus data array)
        # make const later
        self.indices : pk.View1D[int] = indices
        self.indptr : pk.View1D[int] = indptr

        # init random priorites and set status to undecided (-1)
        s = set()
        for idx in range(self.size):
            r = 0 
            while len(s) == idx:
                r = random.randint(1,self.size)
                s.add(r)
            self.priority[idx] = r
            self.status[idx] = -1

        # set completion status to False (0)
        self.done[0] = 0

    # find maximum independent set
    @pk.workunit
    def compute_kernel(self, i: int):
        d : int = 0
        # check if self is undecided (-1)
        if self.status[i] == -1:
            best: int = 1 
            # figure out if a neighbor has a higher priority
            for j in range(self.indptr[i], self.indptr[i+1]):
                neighbor: int = self.indices[j]
                if neighbor == -1:
                    break
                if self.status[neighbor] != 0:
                    if self.priority[i] < self.priority[neighbor]:
                        best = 0
                        break
            # if highest priority among neighbors, set self to in (1)
            # set all neighbors to out (0)
            if best == 1:
                self.status[i] = 1
                for j in range(indptr[i], self.indptr[i+1]):
                    neighbor: int  = self.indices[j]
                    self.status[neighbor] = 0
            # might need another round to check resolution status (0)
            else:
                self.done[0] = 0

@pk.functor(indices = pk.ViewTypeInfo(space=pk.CudaSpace),
            indptr  = pk.ViewTypeInfo(space=pk.CudaSpace))
class _Workload(Workload):
#    def __init__(self, indices: pk.View1D[int], indptr: pk.View1D[int]):
#        self.size: int = len(indptr) - 1
#
#        # init variable list
#        self.done: pk.View1D[int] = pk.View([1], int)
#        self.priority: pk.View1D[int] = pk.View([self.size], int)
#        self.status: pk.View1D[int] = pk.View([self.size], int)
#
#        # graph info needed (CSR minus data array)
#        # make const later
#        self.indices: pk.View1D[int] = indices
#        self.indptr : pk.View1D[int] = indptr
#
#        # init random priorites and set status to undecided (-1)
#        s = set()
#        for idx in range(self.size):
#            r = 0
#            while len(s) == idx:
#                r = random.randint(1,self.size)
#                s.add(r)
#            self.priority[idx] = r
#            self.status[idx] = -1
#
#        # set completion status to False (0)
#        self.done[0] = 0
#  
    # find maximum independent set
    @pk.workunit
    def compute_kernel(self, i: int):
        d : int = 0
        # check if self is undecided (-1)
        if self.status[i] == -1:
            best: int = 1
            # figure out if a neighbor has a higher priority
            for j in range(self.indptr[i], self.indptr[i+1]):
                neighbor: int = self.indices[j]
                if neighbor == -1:
                    break
                if self.status[neighbor] != 0:
                    if self.priority[i] < self.priority[neighbor]:
                        best = 0
                        break
            # if highest priority among neighbors, set self to in (1)
            # set all neighbors to out (0)
            if best == 1:
                self.status[i] = 1
                for j in range(indptr[i], self.indptr[i+1]):
                    neighbor: int  = self.indices[j]
                    self.status[neighbor] = 0
            # might need another round to check resolution status (0)
            else:
                self.done[0] = 0


def run(G: nx.Graph, verbosity, space) -> None:
    print("Total Nodes: ", G.number_of_nodes(), "\n")
    print("Total Edges: ", G.number_of_edges(), "\n\n")

    if verbosity > 1:
        print("Nodelist: ", list(G.nodes), "\n")
        print("Edgelist: ", list(G.edges), "\n\n")

    # workspace init variables    
    size = G.number_of_nodes() 

    if size == 0:
        print("This Graph has no nodes!\n")
        return

    # Get sparse matrix (of edges) out of G 
    A = nx.to_scipy_sparse_array(G, dtype='float64',  format='csr')
	
    if verbosity > 2:
    	print("EdgeDense:\n", A.toarray(), "\n")
    	print("EdgeCSR:\n", "  Data: ", A.data, "\n", "  Indx: ", A.indices, "\n", "  IPrt: ", A.indptr, "\n\n")

    w = None 
    if space == "OpenMP":
        pk.set_default_space(pk.ExecutionSpace.OpenMP)
        w = Workload(pk.from_numpy(A.indices), pk.from_numpy(A.indptr))
    if space == "Cuda":
        _A = cp.sparse.csr_matrix(A)
        pk.set_default_space(pk.ExecutionSpace.Cuda)
        w = _Workload(pk.from_cupy(_A.indices), pk.from_cupy(_A.indptr))

    if verbosity > 3:
    	# print node status before
    	print("Node status (pre): ", w.status, "\n")
    	print("Node priority: ", w.priority, "\n\n")

    p = pk.RangePolicy(pk.get_default_space(), 0, size)
    # while more rounds are needed (0); run kernel
    while(w.done[0] != -1):
        if 5 > verbosity > 0:
            print('.', end = '')

        w.done[0] = -1
        if space == "OpenMP":
          pk.parallel_for(p, w.compute_kernel)
        if space == "Cuda":
          pk.parallel_for(p, w.compute_kernel)
        if verbosity > 4:
            # print node status during
            print("    Node status: ", w.status, "\n")

    if verbosity > 3:
    	# print node status after
    	print("\nNode status (post): ", w.status, "\n")

    # Print nodes that are in the MIS
    mis = []
    for idx, node in enumerate(list(G.nodes)):
        if w.status[idx] == 1:
            mis.append(node)

    print("\n\nMaximum Independent Set: ", mis, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='MIS',
                                     description='Calculates the Maximal Independent Set of a Graph',
                                     epilog='For more pykokkos help, please head over to: https://github.com/kokkos/kokkos-python')
    parser.add_argument('-g','--graphtype',
                        action='store',
                        help='the type of graph to run on. Options are atlas:[0-1235], tutte, cycle:[number of nodes], from-file:[path to Matrix Market coordinate format file]')
    parser.add_argument('-i','--input',
                        action='store',
                        help='the input to graph-type if more information is required')
    parser.add_argument('-v', '--verbosity',
                        action='count',
                        default=0,
                        help='increase output verbosity')
    parser.add_argument('-s','--space',
                        action='store',
                        default='OpenMP',
                        help='the execution space. Options are OpenMP or Cuda')

    args = parser.parse_args()

    if args.graphtype is None and args.input is not None:
        print("Cannot use input field without a valid graphtype field")
        parser.print_help()
        sys.exit(1)
    
    if args.graphtype == 'tutte' and args.input is not None:
        print(args.graphtype, " does not requires input field")
        parser.print_help()
        sys.exit(2)        

    if((args.graphtype == 'atlas' and args.input is None) or
       (args.graphtype == 'cycle' and args.input is None) or
       (args.graphtype == 'from-file' and args.input is None)):
        print(args.graphtype, " requires input field")
        parser.print_help()
        sys.exit(3)

    if (args.space != 'OpenMP' and args.space != 'Cuda'):
        print(args.space, " is not a recognized value")
        parser.print_help()
        sys.exit(4)

    G = nx.Graph()
    if args.graphtype == 'tutte':
        G = nx.tutte_graph()
    elif args.graphtype == 'atlas':
        i = int(args.input)
        if i < 0 or i > 1235:
            print(args.graphtype, " requires input field between 0 and 1235")
            parser.print_help()
            sys.exit(5)
        G = nx.graph_atlas(i)
    elif args.graphtype == 'cycle':
        i = int(args.input)
        G = nx.cycle_graph(i)
    elif args.graphtype == 'from-file':
        m = sp.io.mmread(args.input)
        G = nx.from_numpy_array(m.tocsr()) 
    elif args.graphtype:
        print(args.graphtype, " is not a recognized value")
        parser.print_help()
        sys.exit(6)
    else:
        G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'F')])
        G.add_edges_from([('B', 'A'), ('B', 'G')])
        G.add_edges_from([('C', 'A'), ('C', 'G')])
        G.add_edges_from([('D', 'A'), ('D', 'E'), ('D', 'G')])
        G.add_edges_from([('E', 'D'), ('E', 'F'), ('E', 'G')])
        G.add_edges_from([('F', 'A'), ('F', 'E'), ('F', 'G')])
        G.add_edges_from([('G', 'B'), ('G', 'C'), ('G', 'D'), ('G', 'E'), ('G', 'F'), ('G', 'H'), ('G', 'I')])
        G.add_edges_from([('H', 'G'), ('H', 'I')])
        G.add_edges_from([('I', 'G'), ('I', 'H')])

    run(G, args.verbosity, args.space)
