import cupy as cp
import numpy as np
import pykokkos as pk

import random
import networkx as nx
import scipy as sp

@pk.functor
class Workload:
    def __init__(self, indices: pk.View1D[int], indptr: pk.View1D[int]):
        # init variable list
        self.done: pk.View1D[int] = pk.View([1], int)
        self.size: int = len(indptr) - 1
        self.priority: pk.View1D[int] = pk.View([self.size], int)
        self.status: pk.View1D[int] = pk.View([self.size], int)

        # graph info need
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
            for j in range(indptr[i], self.indptr[i+1]):
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

def run() -> None:
    # create graph from paper
    G = nx.Graph()

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

    print("Nodelist: ", list(G.nodes), "\n")
    print("Edgelist: ", list(G.edges), "\n\n")

    # workspace init variables    
    size = G.number_of_nodes() 

    # Get sparse matrix (of edges) out of G 
    A = nx.to_scipy_sparse_array(G, format='csr')

    #print("EdgeDense:\n", A.toarray(), "\n")
    #print("EdgeCSR:\n", "  Data: ", A.data, "\n", "  Indx: ", A.indices, "\n", "  IPrt: ", A.indptr, "\n\n")

    pk.set_default_space(pk.ExecutionSpace.OpenMP)
    #pk.set_default_space(pk.ExecutionSpace.Cuda)
    w = Workload(pk.from_numpy(A.indices), pk.from_numpy(A.indptr))
    thread_count = 10 

    # print node status before
    #print("Node status (pre): ", w.status)

    # while more rounds are needed (0); run kernel
    while(w.done[0] == 0):
        w.done[0] = 1
        p = pk.RangePolicy(pk.get_default_space(), 0, thread_count)
        pk.parallel_for(p, w.compute_kernel)
        # print node status during
        #print("    Node status: ", w.status)

    # print node status after
    #print("Node status (post): ", w.status)

    # Print nodes that are in the MIS
    mis = []
    for idx, node in enumerate(list(G.nodes)):
        if w.status[idx] == 1:
            mis.append(node)

    print("Maximum Independent Set: ", mis, "\n")

if __name__ == "__main__":
    run()
