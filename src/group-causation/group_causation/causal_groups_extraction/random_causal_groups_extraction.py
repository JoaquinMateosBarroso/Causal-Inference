import random
import numpy as np
from sympy import bell


from group_causation.causal_groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase
from group_causation.causal_groups_extraction.stat_utils import get_scores_getter



class RandomCausalGroupsExtractor(CausalGroupsExtractorBase): # Abstract class
    '''
    Class to extract a set of groups of variables by using a random search
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        scores : list[str] with the name of the score to optimize (only one)
    '''
    def __init__(self, data: np.ndarray, scores: list[str], **kwargs):
        super().__init__(data, **kwargs)
        self.score_getter = get_scores_getter(data, scores)
        
    def extract_groups(self) -> tuple[list[set[int]]]:
        '''
        
        
        Returns
            groups : list of sets with the variables that compound each group
        '''
        # Define the set to partition
        n_variables = self.data.shape[1]
        ELEMENTS = list(range(0, n_variables))
        def get_random_partition():
            # Generate a random partition
            indices = list(range(n_variables))
            random.shuffle(indices)
            num_groups = random.randint(1, n_variables)  # Random number of subsets
            cuts = sorted(random.sample(range(1, n_variables), num_groups - 1))  # Cut points
            partition = []
            start = 0
            for cut in cuts + [n_variables]:
                partition.append([ELEMENTS[i] for i in indices[start:cut]])
                start = cut
            return partition
        
        best_partition = None
        best_score = float('-inf')
        for i in range(bell(max(n_variables//2, 1))):
            partition = get_random_partition()
            [score] = self.score_getter(partition)
            if score > best_score:
                best_score = score
                best_partition = partition
        
        return best_partition 
