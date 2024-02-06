import numpy as np

#This is an implementation of the algorithm for maintaining a topological order
# described in the paper "On-line Graph Algorithms for Incremental Compilation"
# by Alberto Marchetti-Spaccamela, Umberto Nanni, and Hans Rohhert.
#This algorithm is denoted by DFSII in our paper
#The terms "edge" and "arc" are used interchangeably throughout the code. Both of them refer to directed edges.

class Graph:
    def __init__(self, n):
        self.n = n #number of nodes in the graph. Nodes are indexed from 0 to n-1
        self.m = 0 #number of edges in the graph
        self.outNeighbors = [[] for _ in range(n)]
        self.ord = list(np.random.permutation(n)) #this list maps each node to its position in the topological order. Initially, ord is set to be a random permutation.
        self.A = [None] * n #this list represents the topological order, i.e., A[i] equals the node that is stored in position i in the topological order
        for v in range(n):
            self.A[self.ord[v]] = v
        self.mark = [False] * self.n #used in the depth first searches
        self.visited = [] #the set of the visited nodes after each search
        self.arcs = set() #the set of the arcs in the current graph. This is to make sure we do not insert an arc twice.
        self.cost = 0 #sum of the number of nodes and edges processed
            
    #inserts the arc (u,v)
    #if a cycle is formed by adding this arc it returns True, otherwise it returns False
    def insertArc(self, u, v):
        if (u,v) in self.arcs:
            return False
        self.arcs.add((u,v))
        self.m += 1
        self.outNeighbors[u].append(v)
        #STAGE 1
        #self.mark is all-False and self.visited is empty at this point
        self.forwardSearch(v,u)
        if self.mark[u] == True:
            print("Found cycle!")
            return True
        #STAGE 2
        shift = 0
        L = []
        if self.ord[u] > self.ord[v]:
            self.cost += self.ord[u] - self.ord[v] + 1
        for i in range (self.ord[v], self.ord[u]+1):
            if self.mark[self.A[i]] == True:
                L.append(self.A[i])
                shift += 1
            else:
                self.ord[self.A[i]] = i-shift
                self.A[i-shift] = self.A[i]
        index = self.ord[u] + 1 #we start inserting the elements in L starting this position
        for x in L:
            self.A[index] = x
            self.ord[x] = index
            index += 1
        #make sure self.mark is all-False and self.visited is empty before the next forward search
        for x in self.visited:
            self.mark[x] = False
        self.visited = [] 
        return False    

    #This is a truncated forward search from node w. If ord[w] > ord[u], i.e., w is after u in the topological order, we prune the search
    def forwardSearch(self, w, u):
        self.cost += 1
        self.mark[w] = True
        self.visited.append(w)
        if self.ord[w] > self.ord[u]:
            return
        for x in self.outNeighbors[w]:
            self.cost += 1
            if self.mark[x] == False:
                self.forwardSearch(x, u)