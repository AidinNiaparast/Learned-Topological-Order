import random
import LDFS
import DFSII
import time
import numpy as np
import statistics 
import matplotlib.pyplot as plt
from matplotlib import rc
import math

#The terms "edge" and "arc" are used interchangeably throughout the code. Both of them refer to directed edges.

#This class is for computing the number of ancestor edges of vertices. This is used to generate predictions for LDFS.
class Graph:
    def __init__(self, n):
        self.n = n #Number of nodes. Nodes are indexed from 0 to n-1.
        self.m = 0 #Number of edges
        self.inNeighbors = [[] for _ in range(n)]
        self.arcs = set() #The set of the arcs in the current graph. This is to make sure we do not insert an arc twice.

    def insertArc(self, u, v):
        if (u,v) in self.arcs:
            return
        self.arcs.add((u,v))
        self.inNeighbors[v].append(u)

    def numAncestorEdges(self, v):
        self.mark = [False] * self.n
        self.countAncestorEdges = 0
        self.backwardDFS(v)
        return self.countAncestorEdges

    def backwardDFS(self, v):
        self.mark[v] = True
        self.countAncestorEdges += len(self.inNeighbors[v])
        for u in self.inNeighbors[v]:
            if self.mark[u] == False:
                self.backwardDFS(u)            

def readDataset(datasetName):
    path = 'Datasets\\'
    fileName = ''
    match datasetName:
        case 'CollegeMsg':
            fileName = 'CollegeMsg.txt'
        case 'email-Eu-core': 
            fileName = 'email-Eu-core-temporal.txt'
        case 'MathOverflow':
            fileName = 'sx-mathoverflow-a2q.txt'
    with open(path + fileName) as f:
        lines = f.readlines()
    #Each line in the data contains sourceID, targetID, and a timestamp
    #Sort the dataset in increasing order of timestamp
    lines.sort(key=lambda x: int(x.split()[2]))
    #Determine maximum node ID
    maxID = 0
    for line in lines:
        source, target, time = line.split()
        maxID = max(int(source), maxID)
        maxID = max(int(target), maxID)
    #Commit to a permutation of the numbers 0,...,maxID as the ranks of the nodes, 
    #and only keep the edges that go from smaller to larger ranks
    #This is to make sure the graph is acyclic
    ranks = list(range(maxID+1))
    random.shuffle(ranks)
    arcs = []
    for line in lines:
        source, target, time = line.split()
        if ranks[int(source)] < ranks[int(target)]:
            arcs.append((int(source), int(target)))       
    return mapNodes(arcs)

#If the graph has n arcs, this functions maps them to {0,...,n-1}. 
#This is to make sure there are no isolated vertices in the graphs.
def mapNodes(arcs):
    nodes = set()
    for (u,v) in arcs:
        nodes.add(u)
        nodes.add(v)
    nodes = list(nodes)
    n = len(nodes)
    #nodes[i] is mapped to i, for i=0,...,n-1
    mapping = {}
    for i in range(n):
        mapping[nodes[i]] = i   
    newArcs = [(mapping[u],mapping[v]) for (u,v) in arcs]   
    return n, newArcs        

#Generates a DAG with nodes {0,1,...,n-1}, and for each 0 <= u < v <= n-1, 
# it adds the edge (u,v) independently with probability p
def generateDAG(n, p):
    #Commit to a permutation of the numbers 0,...,n-1 as the ranks of the nodes, 
    #and only add edges from smaller to larger ranks
    #This is to make sure the final graph is acyclic
    ranks = list(range(n))
    random.shuffle(ranks)
    arcs = []
    for u in range(n):
        for v in range(u+1,n):
            if np.random.binomial(1, p) == 1:
                if ranks[u] < ranks[v]:
                    arcs.append((u,v))
                else:
                    arcs.append((v,u))  
    #Once the arc set of the dag is determined, the list of arcs are randomly permutated                
    random.shuffle(arcs)                              
    return arcs            

#Generates predictions for LDFS. It uses a contiguous subsequence of the data between 
# positions trainingDataStart and trainingDataEnd as the training data
#It inserts all of the training arcs into an empty graph, and then returns the number
# of ancestor edges of each node in that graph as the prediction for that node
def generatePredictions(data, trainingDataStart, trainingDataEnd, n):
    trainingData = data[trainingDataStart:trainingDataEnd]
    g = Graph(n)
    #The duplicate arcs are added only once
    for (u,v) in trainingData:
        g.insertArc(u,v)
    predictions = [g.numAncestorEdges(v) for v in range(n)] 
    return predictions    

#Adds an independent normal noise with mean 0 and standard deviation SD to each value in predictions 
def perturbPredictions(predictions, SD):
    n = len(predictions)
    noise = np.random.normal(0, SD, n)
    return (np.array(predictions) + noise).tolist()    

#This function runs one of the algorithms 'LDFS', 'DFSI', 'DFSII'.
#It inserts the list arcs using the algorithm, and returns its performance, 
# both in terms of time and in terms of cost, which is the number of edges and nodes processed
#The function receives a list of predictions too, which it only uses in case the algorithm is LDFS  
#DFSI is essentially the LDFS algorithm but with all-zero predictions. So all the nodes are in the same level.
#DFSII is the algorithm for maintaining a topological order described in the paper 
# "On-line Graph Algorithms for Incremental Compilation" 
# by Alberto Marchetti-Spaccamela, Umberto Nanni, and Hans Rohhert.
def runSingleAlg(algName, arcs, predictions, n):
    match algName:
        case 'LDFS':
            g = LDFS.Graph(n, len(arcs), predictions)
        case 'DFSI':
            g = LDFS.Graph(n, len(arcs), [0] * n)
        case 'DFSII':
            g = DFSII.Graph(n)
        case _:
            print('Invalid Algorithm!')
            quit()
    startTime = time.time()
    for (u,v) in arcs:
        g.insertArc(u,v)
    runtime = time.time() - startTime
    cost = g.cost #sum of the number of nodes and edges processed
    return runtime, cost

#This function prints and plots the performance of the algorithms LDFS, DFSI, and DFSII in terms of runtime 
# and cost (number of nodes and edges processed), for different amount of training data
#Note that the only learned algorithm is LDFS, and the other two algorithms do not depend on the training data.
#All the algorithms insert the arcs in the second half of the data, i.e., test data is the last 50% of the data.
#The training data always comes right before the test data 
def scaleTrainingDataSize(n, arcs, plotName):
    m = len(arcs)
    testArcs = arcs[m//2:] #always the second half of the input is used as the test data
    #The training set percentage can be 0%, 5%, ..., 50% of the whole data
    #When t percent of the data is used for training, the training set is from (50-t)'th percentile of the arcs to the 50'th percentile
    #This is to make sure the training data comes right before the test data
    trainingPercentages = [5 * i for i in range(11)]
    #Get results for the LDFS data structure
    LDFSRuntimes = []
    LDFSCosts = []
    for percentage in trainingPercentages:
        predictions = generatePredictions(arcs, m//2 - m*percentage//100, m//2, n)
        LDFSRuntime, LDFSCost = runSingleAlg('LDFS', testArcs, predictions, n)
        LDFSRuntimes.append(round(LDFSRuntime,4))
        LDFSCosts.append(round(LDFSCost))
    #Get results for DFSII algorithm
    #The initial order is obtained by randomly permutating the vertices. We run the experiment 5 times and take the average.
    numExperiments = 5
    DFSIIRuntime = 0
    DFSIICost = 0 #sum of the number of nodes and edges processed
    for _ in range(numExperiments):
        runtime, cost = runSingleAlg('DFSII', testArcs, [], n)
        DFSIIRuntime += runtime
        DFSIICost += cost
    DFSIIRuntime /= numExperiments
    DFSIIRuntime = round(DFSIIRuntime, 4)
    DFSIICost //= numExperiments
    #Print results
    numDataPoints = len(LDFSRuntimes)
    #The results for DFSI are basically the results for LDFS when we do not use any training data, which results in all-zero predictions
    DFSIRuntimes = [LDFSRuntimes[0]] * numDataPoints  
    DFSICosts = [LDFSCosts[0]] * numDataPoints 
    DFSIIRuntimes = [DFSIIRuntime] * numDataPoints
    DFSIICosts = [DFSIICost] * numDataPoints
    print('Experiment: Scaling Training Data Size')
    print('trainingPercentages=', trainingPercentages)
    print('LDFSRuntimes=', LDFSRuntimes)
    print('DFSIRuntimes=', DFSIRuntimes)
    print('DFSIIRuntimes=', DFSIIRuntimes)
    print('LDFSCosts=', LDFSCosts)
    print('DFSICosts=', DFSICosts)
    print('DFSIICosts=', DFSIICosts)
    print()
    #Plots
    lineWidth = 7
    #Time Plot (in log scale)
    plt.plot(trainingPercentages, [math.log10(x) for x in LDFSRuntimes], color='blue', linestyle='-', label='LDFS', linewidth = lineWidth)
    plt.plot(trainingPercentages, [math.log10(x) for x in DFSIRuntimes], color='seagreen', linestyle='-', label='DFS I', linewidth = lineWidth)
    plt.plot(trainingPercentages, [math.log10(x) for x in DFSIIRuntimes], color='darkorange', linestyle='-', label='DFS II', linewidth = lineWidth)   
    plt.ylabel('$\log_{10}$ Total Time (s)')
    plt.xlabel("Training Data Size Percentage")    
    plt.title(plotName+': Scaling Training Data Size', pad=20)
    plt.legend(loc='center right')
    plt.show()

    #Cost Plot (in log scale)
    plt.plot(trainingPercentages, [math.log10(x) for x in LDFSCosts], color='blue', linestyle='-', label='LDFS', linewidth = lineWidth)
    plt.plot(trainingPercentages, [math.log10(x) for x in DFSICosts], color='seagreen', linestyle='-', label='DFS I', linewidth = lineWidth)
    plt.plot(trainingPercentages, [math.log10(x) for x in DFSIICosts], color='darkorange', linestyle='-', label='DFS II', linewidth = lineWidth)
    plt.ylabel("$\log_{10}$ Total Cost")
    plt.xlabel("Training Data Size Percentage")    
    plt.title(plotName+': Scaling Training Data Size', pad=20)
    plt.legend(loc='center right')
    plt.show()

#This function always uses 5 percent of the arcs as training data and the rest as tets data.
#After getting the predictions using the training set, it perturbs the predictions using a normal noise with standard deviation SD.
#It prints and plots the cost (number of nodes and edges processed) and the runtime of the algorithms for different values of SD.
def robustness(n, arcs, plotName):
    m  = len(arcs)
    #always the last 95% of the input is used as the test data
    testArcs = arcs[m//20:]
    #Get results for DFSI algorithm
    DFSIRuntime, DFSICost = runSingleAlg('DFSI', testArcs, [], n)
    DFSIRuntime = round(DFSIRuntime, 4)
    #Get results for DFSII
    #The initial order is obtained by randomly permutating the vertices. We run the experiment 5 times and take the average
    numExperiments = 5
    DFSIIRuntime = 0
    DFSIICost = 0
    for _ in range(numExperiments):
        runtime, cost = runSingleAlg('DFSII', testArcs, [], n)
        DFSIIRuntime += runtime
        DFSIICost += cost
    DFSIIRuntime /= numExperiments
    DFSIIRuntime = round(DFSIIRuntime, 4)
    DFSIICost //= numExperiments
    #Get results for the LDFS data with different amounts of perturbation
    LDFSRuntimes = {}
    LDFSCosts = {} 
    predictions = generatePredictions(arcs, 0, m//20, n) #We use first 5 percent of the data as training data
    predictionsSD = statistics.pstdev(predictions) #Standard deviation of the predictions
    #The standard deviation of the noise added to the predictions is a percentage of the standard deviation of the initial predictions
    #SDPercentages contains these percentages, which are [0%,20%,40%,...,200%].
    SDPercentages = [20 * i for i in range(11)]
    for SDPercentage in SDPercentages:
        SD = SDPercentage / 100 * predictionsSD
        LDFSRuntimes[SDPercentage] = []
        LDFSCosts[SDPercentage] = []
        #We regenerate the noise 10 times, and plot the average and the standard deviation of these experiments
        numExperiments = 10
        for _ in range(numExperiments):
            perturbedPredictions = perturbPredictions(predictions, SD)
            runtime, cost = runSingleAlg('LDFS', testArcs, perturbedPredictions, n)
            LDFSRuntimes[SDPercentage].append(round(runtime,4))
            LDFSCosts[SDPercentage].append(cost) 
    #Print results          
    numDataPoints = len(SDPercentages)
    DFSIRuntimes = [DFSIRuntime] * numDataPoints  
    DFSICosts = [DFSICost] * numDataPoints 
    DFSIIRuntimes = [DFSIIRuntime] * numDataPoints
    DFSIICosts = [DFSIICost] * numDataPoints
    #We compute the average and the standard deviation of the results of the 10 runs for each value of SDPercentage
    averageLDFSRuntimes = [round(statistics.mean(LDFSRuntimes[SDPercentage]),4) for SDPercentage in SDPercentages]
    SDLDFSRuntimes = [round(statistics.pstdev(LDFSRuntimes[SDPercentage]),4) for SDPercentage in SDPercentages]
    averageLDFSCosts = [round(statistics.mean(LDFSCosts[SDPercentage]),4) for SDPercentage in SDPercentages]
    SDLDFSCosts = [round(statistics.pstdev(LDFSCosts[SDPercentage]),4) for SDPercentage in SDPercentages]
    #Printing results
    print('Experiment: Extreme Stress Test')
    print('SDPercentages=', SDPercentages)
    print('averageLDFSRuntimes=', averageLDFSRuntimes)
    print('SDLDFSRuntimes=', SDLDFSRuntimes)
    print('DFSIRuntimes=', DFSIRuntimes)
    print('DFSIIRuntimes=', DFSIIRuntimes)
    print('averageLDFSCosts=', averageLDFSCosts)
    print('SDLDFSCosts=', SDLDFSCosts)
    print('DFSICosts=', DFSICosts)
    print('DFSIICosts=', DFSIICosts)
    
    lineWidth = 7
    plt.plot(xValues, [math.log10(x) for x in averageLDFSCosts], color='blue', linestyle='-', label='LDFS', linewidth = lineWidth)
    plt.fill_between(xValues, [math.log10(max(x,0.0000001)) for x in (np.array(averageLDFSCosts) - np.array(SDLDFSCosts)).tolist()], 
                [math.log10(x) for x in (np.array(averageLDFSCosts) + np.array(SDLDFSCosts)).tolist()], alpha=0.5,
                edgecolor='blue', facecolor='skyblue', linestyle = '-')
    plt.plot(xValues, [math.log10(x) for x in DFSICosts], color='seagreen', linestyle='-', label='DFS I', linewidth = lineWidth)
    plt.plot(xValues, [math.log10(x) for x in DFSIICosts], color='darkorange', linestyle='-', label='DFS II', linewidth = lineWidth)          
    plt.ylabel("$\log_{10}$ Total Cost")
    plt.xlabel('SD(noise)/SD(predictions)')
    plt.title(plotName + ': Extreme Stress Test', pad=20)
    plt.legend(loc='center left')
    plt.show()
    #Time Plot
    xValues = [x/100 for x in SDPercentages] #The ratio of the SD of the noise to the SD of the predictions
    plt.plot(xValues, [math.log10(x) for x in averageLDFSRuntimes], color='blue', linestyle='-', label='LDFS', linewidth = lineWidth)
    plt.fill_between(xValues, [math.log10(max(x,0.0000001)) for x in (np.array(averageLDFSRuntimes) - np.array(SDLDFSRuntimes)).tolist()], 
                [math.log10(x) for x in (np.array(averageLDFSRuntimes) + np.array(SDLDFSRuntimes)).tolist()], alpha=0.5,
                edgecolor='blue', facecolor='skyblue', linestyle = '-')
    plt.plot(xValues, [math.log10(x) for x in DFSIRuntimes], color='seagreen', linestyle='-', label='DFS I', linewidth = lineWidth)
    plt.plot(xValues, [math.log10(x) for x in DFSIIRuntimes], color='darkorange', linestyle='-', label='DFS II', linewidth = lineWidth)
    plt.ylabel("$\log_{10}$ Total Time (s)")
    plt.xlabel('SD(noise)/SD(predictions)')
    plt.title(plotName + ': Extreme Stress Test', pad=20)
    plt.legend(loc='center left')
    plt.show()
    #Cost Plot
    

#This function prints and plots the results for synthetic DAGs generated by the generateDAG(n,p) function for different values of p (edge density)
#For the LDFS results, we also add a normal noise to the predictions. The ratio SD(noise)/SD(predictions) can be 0, 1, or 2.
#To get the initial predictions for LDFS, we only use the first 5% of the list of arcs as training.
#The test data for all the experiments is the last 95% of the list of arcs 
def changeDensity(n):
    pVals = [1]
    p = 1
    while p >= 4/n:
        p /= 2
        pVals.append(p)
    pVals.reverse()    
    mVals = []
    DFSIRuntimes = []
    DFSICosts = []
    DFSIIRuntimes = []
    DFSIICosts = []
    LDFSRuntimes = {}
    LDFSCosts = {}
    averageLDFSRuntimes = {}
    SDLDFSRuntimes = {}
    averageLDFSCosts = {}
    SDLDFSCosts = {}
    SDratios = [0, 1, 2] #The ratio of the standard deviation of the noise to the standard deviation of the initial predictions, i.e., SD(noise)/SD(predictions)
    numExperiments = 5
    for p in pVals:
        arcs = generateDAG(n, p)
        m = len(arcs)
        mVals.append(m)
        testArcs = arcs[m//20:] #We use the last 95% of the data for testing
        #LDFS results
        predictions = generatePredictions(arcs, 0, m//20, n) #we use the first 5% of the data as training data
        predictionsSD = statistics.pstdev(predictions)
        for ratio in SDratios:
            LDFSRuntimes[(ratio,p)] = []
            LDFSCosts[(ratio,p)] = []
            SD = ratio * predictionsSD
            for _ in range(numExperiments):
                perturbedPredictions = perturbPredictions(predictions, SD)
                runtime, cost = runSingleAlg('LDFS', testArcs, perturbedPredictions, n)
                LDFSRuntimes[(ratio,p)].append(round(runtime,4))
                LDFSCosts[(ratio,p)].append(cost)
        #DFSI results
        DFSIRuntime, DFSICost = runSingleAlg('DFSI', testArcs, [], n)
        DFSIRuntime = round(DFSIRuntime, 4)
        DFSIRuntimes.append(DFSIRuntime)
        DFSICosts.append(DFSICost)
        #DFSII
        DFSIIRuntime = 0
        DFSIICost = 0
        for _ in range(numExperiments):
            runtime, cost = runSingleAlg('DFSII', testArcs, [], n)
            DFSIIRuntime += runtime
            DFSIICost += cost    
        DFSIIRuntime /= numExperiments
        DFSIIRuntime = round(DFSIIRuntime, 4)
        DFSIICost //= numExperiments
        DFSIIRuntimes.append(DFSIIRuntime)
        DFSIICosts.append(DFSIICost)
    #Compute the average and the standard deviation of different runs for LDFS
    for ratio in SDratios:
        averageLDFSRuntimes[ratio] = [round(statistics.mean(LDFSRuntimes[(ratio,p)]),4) for p in pVals]
        SDLDFSRuntimes[ratio] = [round(statistics.pstdev(LDFSRuntimes[(ratio,p)]),4) for p in pVals]
        averageLDFSCosts[ratio] = [round(statistics.mean(LDFSCosts[(ratio,p)]),4) for p in pVals]
        SDLDFSCosts[ratio] = [round(statistics.pstdev(LDFSCosts[(ratio,p)]),4) for p in pVals]
    #Print results
    print('Synthetic DAG: Scaling Edge Density, n=', n)
    for ratio in SDratios:
        print('averageLDFSRuntimes(SD(noise)/SD(predictions)='+ str(ratio) + ')=', averageLDFSRuntimes[ratio])
        print('SDLDFSRuntimes(SD(noise)/SD(predictions)=' + str(ratio) + ')=', SDLDFSRuntimes[ratio])
    print('DFSIRuntimes=', DFSIRuntimes)
    print('DFSIIRuntimes=', DFSIIRuntimes)
    for ratio in SDratios:
        print('averageLDFSCosts(SD(noise)/SD(predictions)='+ str(ratio) + ')=', averageLDFSCosts[ratio])
        print('SDLDFSCosts(SD(noise)/SD(predictions)=' + str(ratio) + ')=', SDLDFSCosts[ratio])
    print('DFSICosts=', DFSICosts)
    print('DFSIICosts=', DFSIICosts)
    #Plot results
    lineWidth = 7
    xValues = [math.log2(p) for p in pVals]
    #Cost Plot
    for i in range(len(SDratios)):
        ratio = SDratios[i]
        lineStyle = lineStyles[i] 
        plt.plot(xValues, [math.log10(x) for x in averageLDFSCosts[ratio]], color='blue', linestyle=lineStyle, label='LDFS (C=' + str(ratio) + ')', linewidth = lineWidth)           
    plt.plot(xValues, [math.log10(x) for x in DFSICosts], color='seagreen', linestyle='-', label='DFS I', linewidth = lineWidth)
    plt.plot(xValues, [math.log10(x) for x in DFSIICosts], color='darkorange', linestyle='-', label='DFS II', linewidth = lineWidth)
    plt.ylabel("$\log_{10}$ Total Cost")        
    plt.xlabel('$\log_2(p)$')
    plt.title('Synthetic DAG: Scaling Edge Density', pad=20)
    plt.legend(loc='upper left')
    plt.show()
    #Time Plot
    lineStyles = ['solid', 'dashed', 'dotted']
    for i in range(len(SDratios)):
        ratio = SDratios[i]
        lineStyle = lineStyles[i]        
        plt.plot(xValues, [math.log10(max(x,0.0001)) for x in averageLDFSRuntimes[ratio]], color='blue', linestyle=lineStyle, label='LDFS (C=' + str(ratio) + ')', linewidth = lineWidth)          
    plt.plot(xValues, [math.log10(max(x,0.0001)) for x in DFSIRuntimes], color='seagreen', linestyle='-', label='DFS I', linewidth = lineWidth)
    plt.plot(xValues, [math.log10(max(x,0.0001)) for x in DFSIIRuntimes], color='darkorange', linestyle='-', label='DFS II', linewidth = lineWidth)
    plt.ylabel("$\log_{10}$ Total Time (s)")     
    plt.xlabel('$\log_2(p)$')
    plt.title('Synthetic DAG: Scaling Edge Density', pad=20)
    plt.legend(loc='upper left')
    plt.show()

#This function prints and plots the results of the following experiments 
# on real temporal datasets: Scaling Training Data Size, Robustness
#datasetName can be 'CollegeMsg', 'email-Eu-core', 'MathOverflow'
def getResults(datasetName):
    n, arcs = readDataset(datasetName) 
    m = len(arcs)
    staticEdges = len(set(arcs)) #Number of the static edges in the input
    print(datasetName, ': n=', n, ', Static Edges=', staticEdges, ', Temporal Edges=', m)
    plotName = datasetName   
    scaleTrainingDataSize(n, arcs, plotName)
    robustness(n, arcs, plotName)

def main():
    plt.rcParams.update({'font.size': 32})
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({'mathtext.default':  'regular' })
    #Real datasets
    getResults('email-Eu-core')
    #getResults('CollegeMsg')
    #getResults('MathOverflow')
    #Synthetic DAG
    #changeDensity(1000)
if __name__ == "__main__":
    main()