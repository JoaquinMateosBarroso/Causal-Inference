import numpy as np
from abc import abstractmethod

from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscoveryBase


class GroupCausalDiscoveryBase(MicroCausalDiscoveryBase): # Abstract class
    '''
    Base class for causal discovery on groups of variables algorithms
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        groups : list[set[int]] list with the sets that will compound each group of variables.
                    We will suppose that the groups are known beforehand.
                    The index of a group will be considered as its position in groups list.
                    By default, each variable is considered a group.
        standarize : bool indicating if the data should be standarized before applying the algorithm.
    '''
    def __init__(self, data: np.ndarray, groups: list[set[int]]=None, 
                 standarize: bool=True, **kwargs):
        if standarize:
            self.data = data - data.mean(axis=0)
            if np.all((std:=data.std(axis=0))!=0): data /=std
        else:
            self.data = data
        if groups is None:
            self.groups = [set([i]) for i in range(data.shape[1])]
        else:
            self.groups = groups
        self.extra_args = kwargs

    @abstractmethod
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        To be implemented by subclasses
        
        Returns
            Dictionary with the parents of each group of variables.
        '''
        pass