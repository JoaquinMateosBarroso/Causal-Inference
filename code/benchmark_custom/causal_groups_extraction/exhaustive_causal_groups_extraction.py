import numpy as np
from typing import Callable
from more_itertools import set_partitions

from causal_groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase
from causal_groups_extraction.stat_utils import get_scores_getter



class ExhaustiveCausalGroupsExtractor(CausalGroupsExtractorBase): # Abstract class
    '''
    Class to extract a set of groups of variables by using an exhaustive search
    '''
    def __init__(self, data: np.ndarray, scores: list[str], **kwargs):
        '''
        Create an object that is able to extracat meaningful groups 
        from a dataset of time series variables
        
        Parameters
            data : np.array with the data, shape (n_samples, n_variables)
            scores : list[str] with the name of the score to optimize (only one)
        '''
        super().__init__(data, **kwargs)
        self.score_getter = get_scores_getter(data, scores)
    
    def extract_groups(self) -> tuple[list[set[int]]]:
        '''
        Get score over all possible partitions of dataset and return the optimal one
        
        Returns
            groups : list of sets with the variables that compound each group
        '''
        all_posible_partitions = list(set_partitions(range(self.data.shape[1])))
        best_score = float('-inf')
        best_partition = None
        for partition in all_posible_partitions:
            [score] = self.score_getter(partition)
            if score > best_score:
                best_score = score
                best_partition = partition
        
        return best_partition
    


