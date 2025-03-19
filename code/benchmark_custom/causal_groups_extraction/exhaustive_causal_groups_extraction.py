import numpy as np
from memory_profiler import memory_usage
from typing import Callable

from causal_groups_extraction.causal_groups_extraction import CausalGroupsExtractor



class ExhaustiveCausalGroupsExtractor(CausalGroupsExtractor): # Abstract class
    '''
    Class to extract a set of groups of variables by using an exhaustive search
    '''
    def __init__(self, data: np.ndarray, score_getter: Callable, **kwargs):
        '''
        Create an object that is able to extracat meaningful groups 
        from a dataset of time series variables
        
        Parameters
            data : np.array with the data, shape (n_samples, n_variables)
            score_getter : function that receives a set of groups and returns a score to maximize
        '''
        super().__init__(data, **kwargs)
        
        
    
    def extract_groups(self) -> tuple[list[set[int]]]:
        '''
        Get score over all possible partitions of dataset and return the optimal one
        
        Returns
            groups : list of sets with the variables that compound each group
        '''
        pass
    