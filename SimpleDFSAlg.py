

#Nodes are indexed from 0 to n-1

class Graph:
    def __init__(self, n, predictions):
        self.n = n
        self.labels = predictions
        self.m = 0
        self.outNeighbors = [[] for _ in range(n)]
        self.inMatchingNeighbors = [[] for _ in range(n)]
        self.mark = [False] * self.n #this is used in the depth first searches
        self.visited = [] #this set contains the set of the visited nodes after each search

    def __str__(self):
        #topSort = sorted(range(self.n), key=lambda k: self.labels[k])    
        ret = ""
        #for u in topSort:
        #    ret += str(u) + ", "
        
        buckets = [[] for _ in range(self.n)]
        for v in range(self.n):
            buckets[self.labels[v]].append(v)

        ret += 'Levels: '
        for i in range(self.n):
            ret += '{'
            for v in buckets[i]:
                if v == buckets[i][-1]:
                    ret += str(v)
                else:    
                    ret += str(v) + ','
            ret += '} '

        ret += '\n'
        ret += 'Edges: '    
        for u in range(self.n):
            for v in self.outNeighbors[u]:
                ret += '(' + str(u) + ',' + str (v) + '), '

        ret += '\n'
        for v in range(self.n):
            ret += str(v) + ": "
            for u in self.inMatchingNeighbors[v]:
                ret += str(u) + ','
            ret += ' || '            
        return ret
            
    #inserts the arc (v1,v2)
    def insertArc(self, v1, v2):
        self.m += 1
        self.outNeighbors[v1].append(v2)

        if self.labels[v1] < self.labels[v2]:
            return
        elif self.labels[v1] == self.labels[v2]:
            self.inMatchingNeighbors[v2].append(v1)
        else: #self.labels[v1] > self.labels[v2]
            #print('Back Edge!')
            self.labels[v2] = self.labels[v1]
            self.inMatchingNeighbors[v2] = [v1] #reset the matching in-neighbors of v2 becuase its label is changed
            #self.mark is all-False and self.visited is empty before we call the forwardSearch function
            self.forwardSearch(self.labels[v2], v2)
            #maintain the invariant for self.mark and self.visited for the next search
            for v in self.visited:
                self.mark[v] = False
            self.visited = []    

        isCycle = self.cycleCheck(v1, v2)
        if isCycle:
            print("Found cycle!")
            quit()

    def forwardSearch(self, label, v):
        self.mark[v] = True
        self.visited.append(v)
        self.labels[v] = label
        for u in self.outNeighbors[v]:
            if self.labels[u] <= label:
                if self.mark[u] == False and self.labels[u] < label:
                    self.inMatchingNeighbors[u] = []
                    self.forwardSearch(label, u)
                self.inMatchingNeighbors[u].append(v)  
                    

    def cycleCheck(self, v1, v2):
        #self.mark is all-False and self.visited is empty before we call the function
        isCycle = self.backwardSearch(v1, v2)
        for v in self.visited:
            self.mark[v] = False
        self.visited = []
        return isCycle 

    def backwardSearch(self, v, dest):
        self.mark[v] = True
        self.visited.append(v)
        if v == dest:
            return True
        for u in self.inMatchingNeighbors[v]:
            if self.mark[u] == False and self.labels[u] == self.labels[dest]:
                isPath = self.backwardSearch(u,dest)
                if isPath:
                    return True
        return False
    
def main():
    g = Graph(8, [0,1,2,3,4,5,6,7])
    edges = [(1,2),(1,3),(2,3),(2,4),(5,1),(5,6),(6,7),(3,6),(4,5)]
    for (u,v) in edges:
        g.insertArc(u,v)
        print(g)

if __name__ == "__main__":
    main()    