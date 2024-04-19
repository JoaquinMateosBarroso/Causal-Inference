import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency

class PC:
    def __init__(self, alpha=0.01, directional=False):
        self.alpha = alpha
        self.directional = directional
        
    def causalDiscovery(self, data: pd.DataFrame):
        # Initialize the complete graph
        self.graph = {column: set(data.columns.drop(column)) \
                            for column in data.columns}
        
        self.separatingSets = [] # Tuples ((X, Y), Z) where X and Y are nodes and Z is their separating set
        
        # Case with depth=0
        for X, Y in combinations(self.graph.keys(), 2):
            p_value = chi2_contingency(pd.DataFrame([data[X], data[Y]]).T)[1]
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
                
                for Z in combinations(self.__adjacent(X).difference(Y), depth):
                    p_value = chi2_contingency(pd.crosstab(index=[data[X], data[Y]], 
                                    columns=[data[z] for z in Z]))[1]
                    if p_value > self.alpha:
                        self.graph[X].difference_update( [(Y)] )
                        self.graph[Y].difference_update( [(X)] )
                        
                        if self.directional: # Optional step to remove separating implications
                            for z in Z:
                                self.graph[X].difference_update( [(z)] )
                                self.graph[Y].difference_update( [(z)] )
                        
                        self.separatingSets.append(((X, Y), Z))
                        break
            print(f'Depth {depth} completed')
            depth += 1
            
        
        return self.graph, self.separatingSets
    
        
    def __adjacent(self, node: str)->set:
        return set(self.graph[node])

    def __testStopCriteria(self, depth):
        '''Returns True if the algorithm should stop, False otherwise'''
        for X, Y in combinations(self.graph.keys(), 2):
            if len(self.__adjacent(X).difference(Y)) > depth:
                return False
        return True

