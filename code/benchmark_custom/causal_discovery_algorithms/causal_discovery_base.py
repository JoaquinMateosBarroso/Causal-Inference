'''
Module with the base class for causal discovery algorithms.
'''


import time
import numpy as np
from abc import ABC, abstractmethod
from memory_profiler import memory_usage


class CausalDiscoveryBase(ABC): # Abstract class
    '''
    Base class for causal discovery algorithms
    
    To be implemented by subclasses
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        standarize : bool indicating if the data should be standarized. default=True
    '''
    @abstractmethod
    def __init__(self, data: np.ndarray, standarize: bool=True, **kwargs):
        if standarize:
            self.data = (data - data.mean(axis=0)) / data.std(axis=0)
        else:
            self.data = data
    
    @abstractmethod
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        To be implemented by subclasses
        
        Returns
            Dictionary with the parents of each node
        '''
        pass
    
    def extract_parents_time_and_memory(self) -> tuple[dict[int, list[int]], float, float]:
        '''
        Execute the extract_parents method and return the parents dict, the time that took to run the algorithm
        
        Returns:
            parents : dictionary of extracted parents
            execution_time : execution time in seconds
            memory : volatile memory used by the process, in MB
        '''
        tic = time.time()
        memory, parents = memory_usage( self.extract_parents, retval=True, include_children=True, multiprocess=True)
        toc = time.time()
        execution_time = toc - tic
        
        memory = max(memory) - min(memory)  # Memory usage in MiB
        memory = memory * 1.048576 # Exact division   1024^2 / 1000^2
        
        return parents, execution_time, memory


