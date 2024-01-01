import random
import SimpleDFSAlg
import time
import numpy as np
import pandas as pd
import xlsxwriter
#import openpyxl
from openpyxl import load_workbook

path = 'Datasets\\'

class Graph:
    def __init__(self, n):
        self.n = n
        self.m = 0
        self.outNeighbors = [[] for _ in range(n)]
        self.inNeighbors = [[] for _ in range(n)]

    def insertArc(self, u, v):
        self.outNeighbors[u].append(v)
        self.inNeighbors[v].append(u)

    def numAncestors(self, v):
        self.mark = [False] * self.n
        self.countAncestor = 0
        self.backwardDFS(v)
        return self.countAncestor

    def backwardDFS(self, v):
        self.mark[v] = True
        self.countAncestor += 1
        for u in self.inNeighbors[v]:
            if self.mark[u] == False:
                self.backwardDFS(u)            

def readTwitch():
    with open(path + 'large_twitch_edges.csv') as f:
        lines = f.readlines()              
    arcs = []
    n = 10000 #we will just consider the induced graph on nodes 0,...,n
    for line in lines[1:]:
        u, v = line.split(',')
        if int(v) > n or int(u) > n:
            continue
        #direct all edges from smaller to larger id
        if int(u) < int(v):
            arcs.append((int(u),int(v)))
        else:
            arcs.append((int(v), int(u)))    
        #n = max(int(u), n)
        #n = max(int(v), n)         
    return n+1, arcs  

def readLastFM():
    with open(path + 'lastfm_asia_edges.csv') as f:
        lines = f.readlines()              
    arcs = []
    n = 0
    for line in lines[1:]:
        u, v = line.split(',')
        #direct all edges from smaller to larger id
        if int(u) < int(v):
            arcs.append((int(u),int(v)))
        else:
            arcs.append((int(v), int(u)))    
        n = max(int(u), n)
        n = max(int(v), n)         
    return n+1, arcs  

def readWikiVote():
    with open(path + 'wiki-Vote.txt') as f:
        lines = f.readlines()              
    arcs = []
    n = 0
    for line in lines[4:]:
        u, v = line.split()
        #only keep the arcs from smaller to larger id
        if int(u) < int(v):
            arcs.append((int(u),int(v)))
        #else:
        #    arcs.append((int(v), int(u)))    
        n = max(int(u), n)
        n = max(int(v), n)             
    return n+1, arcs


def generatePredictions(data, trainingDataStart, trainingDataEnd, n):
    trainingData = data[trainingDataStart:trainingDataEnd]
    g = Graph(n)
    for (u,v) in trainingData:
        g.insertArc(u,v)
    predictions = []
    for v in range(n):
        #predictions.append(g.numAncestors(v)*len(data)//len(trainingData))
        predictions.append(g.numAncestors(v))
    return predictions    


#n, arcs = readLastFM()
#n, arcs = readTwitch()
n, arcs = readWikiVote()
random.seed(3124)
random.shuffle(arcs)
m = len(arcs)
#print(n,m)
perfectPredictions = generatePredictions(arcs, 0, m, n)

trainingPercentages = [5 * i for i in range(21)]
numExperiments = 1
classicResults = []
learnedResults = []
averageErrors = []
maxErrors = []
totalStartTime = time.time()

for percentage in trainingPercentages:
    learnedRuntime = 0
    classicRuntime = 0
    predictions = generatePredictions(arcs, 0, m * percentage // 100, n)
    averageError = np.linalg.norm(np.array(predictions) - np.array(perfectPredictions), ord=1) / n
    maxError = np.amax(np.absolute(np.array(predictions) - np.array(perfectPredictions)))
    print('Training Data Percentage=', percentage, ', Prediction Average Error=', averageError, ', Prediction Max Error=', maxError)
    for _ in range(numExperiments):
        startTime = time.time()
        g1 = SimpleDFSAlg.Graph(n, predictions)
        for (u,v) in arcs:
            g1.insertArc(u,v)
        learnedRuntime += time.time()-startTime

        startTime = time.time()
        g2 = SimpleDFSAlg.Graph(n, [0] * n)
        for (u,v) in arcs:
            g2.insertArc(u,v)
        classicRuntime += time.time()-startTime

    learnedRuntime /= numExperiments
    classicRuntime /= numExperiments  
    print(learnedRuntime, classicRuntime)     
    learnedResults.append(round(learnedRuntime,2))
    classicResults.append(round(classicRuntime,2))
    averageErrors.append(round(averageError,2))
    maxErrors.append(round(maxError,2))

print('n=', n, ', m=', m)
print('trainingPercentages=', trainingPercentages)
print('learnedResults=', learnedResults)
print('classicResults=', classicResults)
print('averageErrors=', averageErrors)
print('maxErrors=', maxErrors)


df = pd.DataFrame({'Training Percentage': trainingPercentages, 'Learned Runtime': learnedResults, 'Classic Runtime': classicResults, 'Average Error': averageErrors, 'Max Error': maxErrors})
with pd.ExcelWriter('Results.xlsx', engine='openpyxl', mode='a') as writer:
    df.to_excel(writer, sheet_name='wiki-Vote', index=False)

print('Total Time = ', time.time() - totalStartTime)