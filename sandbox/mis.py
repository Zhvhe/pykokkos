import cupy as cp
import numpy as np
import pykokkos as pk
import random
import networkx as nx

@pk.functor
class Workload:
    def __init__(self):
        # init variable list
        self.done: bool = False
        self.size: pk.int = node_count
        self.priority: pk.View1D[int] = pk.View([self.size], int)
        self.status: pk.View1D[int] = pk.View([self.size], int)


        # init random priorites and set status to undecided (-1)
        s = set()
        for idx in range(self.size):
            r = 0 
            while len(s) = idx:
                r = random.randint(1,self.dize)
                s.add(r)
            self.priority[idx] = r
            self.status = -1


    # find maximum independent set
    @pk.workunit
    def compute_kernel(self, i: int, neighbors: pk.View1D[int])):
        # check if self is undecided (-1)
        if self.status[i] == -1:
            best: bool = True 
            # figure out if a neighbor has a higher priority
            for neighbor in neighbors:
                if self.priority[i] < self.priority[neighbor]:
                    best = False
                    break
            # if highest priority among neighbors, set self to in (1)
            # set all neighbors to out (0)
            if best:
                self.status[i] = 1
                for neighbor in neighbors:
                    self.status[neighbor] = 0
            else:
                self.done = False

def run() -> None:
    # workspace init variables    
    G = nx.complete_graph(100)
    
    pk.set_default_space(pk.ExecutionSpace.OpenMP)
    #pk.set_default_space(pk.ExecutionSpace.Cuda)
    w = Workload()
    thread_count : 30
    
    while(w.done != True):
        self.done = True
        p = pk.RangePolicy(pk.get_default_space(), 0, thread_count)
        pk.parallel_for(p, w.compute_kernel)

if __name__ == "__main__":
    run()
