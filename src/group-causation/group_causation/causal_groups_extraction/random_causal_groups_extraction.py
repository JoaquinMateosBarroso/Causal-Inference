import random
import numpy as np


from group_causation.causal_groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase



class RandomCausalGroupsExtractor(CausalGroupsExtractorBase): # Abstract class
    '''
    Class to extract a set of groups of variables by using an exhaustive search
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        score_getter : function that receives a set of groups and returns a score to maximize
        scores_weights : list with the weights of the scores to optimize (a score of 1.0 means to maximize, -1.0 to minimize)
    '''
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
    
    def extract_groups(self) -> tuple[list[set[int]]]:
        '''
        Get score over all possible partitions of dataset and return the optimal one
        
        Returns
            groups : list of sets with the variables that compound each group
        '''
        # Define the set to partition
        n_variables = self.data.shape[1]
        ELEMENTS = list(range(0, n_variables))
        
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
