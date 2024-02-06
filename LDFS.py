#This code implements the LDFS algorithm in the paper "Incremental Topological Ordering and Cycle Detection with Predictions"
#The terms "edge" and "arc" are used interchangeably throughout the code. Both of them refer to directed edges.

class Graph:
    def __init__(self, n, mMax, predictions):
        self.n = n #number of nodes in the graph. Nodes are indexed from 0 to n-1
        self.mMax = mMax #maximum number of arcs that are going to be in the graph. This is inly used for the labels in the topological ordering
        self.level = predictions #in the paper level of u is denoted by l(u)
        self.a = n * mMax + 1 #this global variable is used to set the labels for the nodes in the topological ordering
        self.index = [n * mMax + 2] * n #the indices used to find the topological ordering of the nodes in the same level. In the paper index of u is denoted by j(u).
        self.m = 0 #number of arcs in the graph
        self.outNeighbors = [[] for _ in range(n)]
        self.inMatchingNeighbors = [[] for _ in range(n)] #the list of in-neighbors of a node within the same level
        self.mark = [False] * self.n #used in the depth first searches to mark visited vertices
        self.visited = [] #the set of the visited nodes after each search
        self.arcs = set() #the set of the arcs in the current graph. This is to make sure we do not insert an arc twice.
        self.cost = 0 #sum of the number of nodes and edges processed
                    
    #inserts the arc (v1,v2)
    #if a cycle is formed by adding this arc it returns True, otherwise it returns False
    def insertArc(self, v1, v2):
        if (v1,v2) in self.arcs:
            return False
        self.arcs.add((v1,v2))
        self.m += 1
        self.outNeighbors[v1].append(v2)
        F = [] #the list of the vertices visited in the forward search. Each node comes before its successors in this list.
        B = [] #the list of the vertices visited in the backward search. Each node comes before its successors in this list.
        if self.level[v1] < self.level[v2]:
            return False
        elif self.level[v1] == self.level[v2]:
            self.inMatchingNeighbors[v2].append(v1)
            if self.index[v1] < self.index[v2]:
                return False
        else:
            self.level[v2] = self.level[v1]
            self.inMatchingNeighbors[v2] = [v1] #reset the matching in-neighbors of v2 because its level is changed
            #self.mark is all-False and self.visited is empty at this point
            self.forwardSearch(self.level[v1], v2)
            self.visited.reverse()
            F = self.visited #F contains the nodes visited in the forward search, and each node comes before its successors in this list
            for v in self.visited:
                self.mark[v] = False
            self.visited = []   
        #self.mark is all-False and self.visited is empty at this point     
        self.backwardSearch(v1)
        B = self.visited #B contains the nodes visited in the backward search, and each node comes before its successors in this list
        if self.mark[v2] == True: #we visited v2 during the backward search
            print("Found cycle!")
            return True
        for v in self.visited:
            self.mark[v] = False
        self.visited = []
        #update the indices to maintain topological ordering
        T = B + F
        for v in reversed(T):
            self.index[v] = self.a
            self.a -= 1
        return False    
    #a truncated forward DFS: it only recurses on the out neighbors of v with levels less than destLevel, and moves all such nodes to destLevel
    def forwardSearch(self, destLevel, v):
        self.cost += 1
        self.mark[v] = True
        self.level[v] = destLevel
        for u in self.outNeighbors[v]:
            self.cost += 1
            if self.level[u] <= destLevel:
                if self.mark[u] == False and self.level[u] < destLevel:
                    self.inMatchingNeighbors[u] = []
                    self.forwardSearch(destLevel, u)
                self.inMatchingNeighbors[u].append(v)  
        self.visited.append(v)        
    #a backward DFS
    def backwardSearch(self, v):
        self.cost += 1
        self.mark[v] = True
        for u in self.inMatchingNeighbors[v]:
            self.cost += 1
            if self.mark[u] == False:
                self.backwardSearch(u)
        self.visited.append(v)        
    #Returns the label of node v in the topological ordering
    def getLabel(self, v):
        return self.level[v] * (self.n * self.mMax + 2) + self.index[v]