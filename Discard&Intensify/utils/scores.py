# TODO: Implement the following scores:
from pgmpy.estimators import BicScore, K2Score, AICScore
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd
from utils.graph import DGraph

class Score:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def getScore(self, graph: DGraph) -> float:
        pass
    
class BICScore(Score):
    def getScore(self, graph: DGraph) -> float:
        if not graph.isAcyclic():
            # If the graph is not acylic, we can't even calculate its score, 
            #    so we set it to the biggest possible
            return np.inf
        return BicScore(self.data).score(BayesianNetwork(graph.getEdges()))
    