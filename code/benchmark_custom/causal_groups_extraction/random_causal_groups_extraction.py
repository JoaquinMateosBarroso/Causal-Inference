import random
import numpy as np
from typing import Callable

from deap import base, creator, tools, algorithms

from causal_groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase



class RandomCausalGroupsExtractor(CausalGroupsExtractorBase): # Abstract class
    '''
    Class to extract a set of groups of variables by using an exhaustive search
    '''
    def __init__(self, data: np.ndarray, scores_getter: function, scores_weights: list, **kwargs):
        '''
        Create an object that is able to extracat meaningful groups 
        from a dataset of time series variables
        
        Parameters
            data : np.array with the data, shape (n_samples, n_variables)
            score_getter : function that receives a set of groups and returns a score to maximize
            scores_weights : list with the weights of the scores to optimize (a score of 1.0 means to maximize, -1.0 to minimize)
        '''
        super().__init__(data, **kwargs)
        self.scores_getter = scores_getter
        self.scores_weights = scores_weights
    
    def extract_groups(self) -> tuple[list[set[int]]]:
        '''
        Get score over all possible partitions of dataset and return the optimal one
        
        Returns
            groups : list of sets with the variables that compound each group
        '''
        # Define the set to partition
        ELEMENTS = list(range(1, n_variables+1))
        n_variables = self.data.shape[1]
        
        # Generate a random partition
        indices = list(range(n_variables))
        random.shuffle(indices)
        num_groups = random.randint(2, n_variables)  # Random number of subsets
        cuts = sorted(random.sample(range(1, n_variables), num_groups - 1))  # Cut points
        partition = []
        start = 0
        for cut in cuts + [n_variables]:
            partition.append([ELEMENTS[i] for i in indices[start:cut]])
            start = cut
        
        return partition 
