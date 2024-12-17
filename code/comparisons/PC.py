import pandas as pd
from itertools import combinations
from scipy.stats import kstest, chi2_contingency

class PC:
    def __init__(self, alpha=0.01, endogeneous=[], exogeneous=[], directional=False, maxSeparatingDepth=10):
        self.alpha = alpha
        self.endogeneous = endogeneous
        self.exogeneous = exogeneous
        self.directional = directional
        self.maxDepth = maxSeparatingDepth


    def causalDiscovery(self, data: pd.DataFrame):
        # Initialize the complete graph
        self.graph = {column: set(data.columns.drop(column)) \
                            for column in data.columns}
        
        self.separatingSets = [] # Tuples ((X, Y), Z) where X and Y are nodes and Z is their separating set
        
        # Case with depth=0
        for X, Y in combinations(self.graph.keys(), 2):
            p_value = kstest(data[X], data[Y])[1]
            if p_value > self.alpha:
                self.graph[X].difference_update( [(Y)] )
                self.graph[Y].difference_update( [(X)] )
                self.separatingSets.append(((X, Y), {}))
                break
        print('Depth 0 completed')
        
        depth = 1
        while not self.__testStopCriteria(depth):
            for X, Y in combinations(self.graph.keys(), 2):
                if Y not in self.__adjacent(X):
                    continue # We just want adjacent nodes
                
                for Z in combinations(self.__adjacent(X).difference({Y}), depth):
                    if self.__testConditionalIndependence(X, Y, list(Z), data):
                        self.graph[X].difference_update( [(Y)] )
                        self.graph[Y].difference_update( [(X)] )
                        
                        self.separatingSets.append(((X, Y), Z))
                        break
            print(f'Depth {depth} completed')
            depth += 1

        self.__cleanGraph()
        
        return self.graph, self.separatingSets
    
    def __cleanGraph(self):
        if self.directional:
            for nodes, Z in self.separatingSets:
                for z in Z:
                    X, Y = nodes
                    self.graph[X].difference_update( [(z)] )
                    self.graph[Y].difference_update( [(z)] )
        
        
        for node in self.exogeneous:
            self.__setExogeneous(node)
            
        for node in self.endogeneous:
            self.__setEndogeneous(node)
        
        self.__removeBidirectionalEdges()
    
    def __setEndogeneous(self, node):
        while len(self.graph[node]) != 0:
            addjacent = self.graph[node].pop()
            self.graph[addjacent].add(node)
        
    def __setExogeneous(self, node):
        for key, addjacents in self.graph.items():
            if node in addjacents:
                addjacents.remove(node)
                self.graph[node].add(key)
    
    def __removeBidirectionalEdges(self):
        for key, value in self.graph.items():
            for addjacent in value:
                if key in self.graph[addjacent]:
                    self.graph[addjacent].remove(key)
        
    def __adjacent(self, node: str)->set:
        return set(self.graph[node])

    def __testStopCriteria(self, depth):
        '''Returns True if the algorithm should stop, False otherwise'''
        if depth > self.maxDepth:
            return True
        
        for X, Y in combinations(self.graph.keys(), 2):
            if len(self.__adjacent(X).difference(Y)) > depth:
                return False
        return True

    def __testConditionalIndependence(self, X, Y, Z, data):
        '''Returns True if X and Y are conditionally independent given Z, False otherwise'''        
        p_values = []
        for z in self.__allValuesOf(Z, data):
            conditionalData = data.loc[(data[Z]==z).all(axis=1)]
            
            res = chi2_contingency(pd.crosstab(conditionalData[X], conditionalData[Y]))
            
            p_values.append(res[1])
        
        # Bonferroni correction
        adjusted_alpha = self.alpha / len(p_values)
        
        # Check if any p-value is significant after Bonferroni correction
        if any(p_value < adjusted_alpha for p_value in p_values):
            return False # H0: X and Y are dependent given Z
        else:
            return True # H1: X and Y are independent given Z
            
            
    def __allValuesOf(self, nodes, data):
        '''Returns all possible values of the nodes in the dataframe data'''
        return data[nodes].drop_duplicates().values
    