import numpy as np
import pandas as pd
import random

from utils.graph import DGraph, createFullyConnectedDGraph
from utils.scores import Score, BICScore
from structureLearning import StructureLearner

class LocalSearch(StructureLearner):
    def __init__(self, data: pd.DataFrame, initialGraph: DGraph=None, score: Score=BICScore,
                 movements: list = [DGraph.addEdge, DGraph.reverseEdge, DGraph.removeEdge], #, DGraph.removeEdge, DGraph.reverseEdge], # Estas 2 se pueden añadir cuando se considere el caso bien
                 maxIterations: int = 100, neighbourhoodSize: int = 10,
                 *args,**kwargs):
        '''
            Create a Local Search structure learner.
            :param data: A pandas DataFrame containing the data to learn from.
            :para score: A score class to evaluate the quality of the graphs.
            :param initialGraph: Optional para to have an initial graph to start the learning process.
                                 If it is left as None, the learner will start with an empty graph.
            :param movements: A list of functions that will be used to move between graphs.
            :param maxIterations: The maximum number of iterations to perform.
            :param neighbourhoodSize: The number of neighbours to consider in each iteration.
        '''
        self.movements = movements
        self.maxIterations = maxIterations
        self.score = score(data)
        self.neighbourhoodSize = neighbourhoodSize
        self.iteration = 0
        super().__init__(data, initialGraph, *args, **kwargs)
        
    def learn(self) -> DGraph:
        pass
    
    def getNeighbour(self, graph: DGraph):
        movement = random.choice(self.movements)
        newGraph = self.graph.copy()
        try:
            movement(newGraph)
        except: 
            print('Not able to perform a tried movement; retrying with another.')
            self.getNeighbour(graph)
        if not newGraph.isAcyclic(): # If obtained graph is not acyclic, we retry
            return self.getNeighbour(graph)
        return newGraph, self.score.getScore(newGraph)

    
class SteepestAscent(LocalSearch):
    def learn(self) -> DGraph:
        it = 0
        self.bestScore = self.score.getScore(self.graph)
        while it < self.maxIterations:
            newGraph, newScore = self.getBestNeighbour(self.graph)
            if newScore < self.bestScore:
                self.graph = newGraph
                self.bestScore = newScore
            it += 1
        return self.graph

    def getBestNeighbour(self, graph: DGraph):
        tries = 0
        bestScore = 0
        bestNeighbour, bestScore = self.getNeighbour(graph)
        while tries < self.neighbourhoodSize:
            newNeighbour, newScore = self.getNeighbour(graph)
            if newScore < bestScore:
                bestScore = newScore
                bestNeighbour = newNeighbour
            tries += 1
        return bestNeighbour, bestScore
    
def graphGreedyConstructor(data: pd.DataFrame, nOfEdges: int = None, score=BICScore, nTriesPerIteration = 10) -> DGraph:
    '''
        Create a graph using a greedy constructor, that uses the given 
        Score to evaluate the quality of the graphs.
        :param data: A pandas DataFrame containing the data to learn from.
        :param nOfEdges: The number of edges to add to the graph. If it is left as None,
                            the graph will have half of all possible edges.
        :param nTriesPerIteration: The number of tries to make in each addition of edges.
    '''
    if nOfEdges is None:
        maximumNumberOfEdges = (len(data.columns) - 1) * len(data.columns) / 2
        nOfEdges = maximumNumberOfEdges // 2
    score = score(data)
    
    graph = DGraph(nodes=data.columns)
    # Add edges until the desired number is reached
    while graph.nOfEdges() < nOfEdges:
        bestGraph = graph
        bestScore = score.getScore(graph)
        for _ in range(nTriesPerIteration):
            newGraph = graph.copy()
            newGraph.addEdge()
            # We need to get sure that the graph is acicly to calculate its score
            while not newGraph.isAcyclic():
                newGraph = graph.copy()
                newGraph.addEdge()
            newScore = score.getScore(newGraph)
            if newScore < bestScore:
                bestScore = newScore
                bestGraph = newGraph
        if bestGraph is not None:
            graph = bestGraph
    
    return graph
    
    
            
            
if __name__ == "__main__":
    data = pd.read_csv("data/titanic.csv")
    # Borrar aquellas columnas que no son descriptivas
    data = data.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
    
    greedylyObtainedGraph = graphGreedyConstructor(data)
    print('Obtained BIC Score:', BICScore(data).getScore(greedylyObtainedGraph))
    greedylyObtainedGraph.plot()
    
    learner = SteepestAscent(data, score=BICScore, initialGraph=greedylyObtainedGraph)
    learner.learn()
    print('Obtained BIC Score:', learner.score.getScore(learner.graph))
    learner.graph.plot()
    
