import numpy as np
from abc import ABC, abstractmethod
from memory_profiler import memory_usage



class DirectionExtractorBase(ABC): # Abstract class
    '''
    Base class for direction extraction algorithms on groups of variables
    '''
    def __init__(self, data: np.ndarray, groups: list[set[int]], **kwargs):
        '''
        Create an object that is able to predict over groups of time series variables
        
        Parameters
            data : np.array with the data, shape (n_samples, n_variables)
            groups : list[set[int]] list with the sets that will compound each group of variables.
                        We will suppose that the groups are known beforehand.
                        The index of a group will be considered as its position in groups list.
        '''
        pass
