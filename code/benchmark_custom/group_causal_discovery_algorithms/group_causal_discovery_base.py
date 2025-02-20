import time
import numpy as np
from abc import ABC, abstractmethod
from memory_profiler import memory_usage



class GroupCausalDiscoveryBase(ABC): # Abstract class
    '''
    Base class for causal discovery on groups of variables algorithms
    '''
    def __init__(self, data: np.ndarray, groups: list[set[int]], **kwargs):
        '''
        Create an object that is able to predict over groups of time series variables
        
        Parameters
        ---------
            data : np.array with the data, shape (n_samples, n_variables)
            groups : list[set[int]] list with the sets that will compound each group of variables.
                        We will suppose that the groups are known beforehand.
                        The index of a group will be considered as its position in groups list.
        '''
        pass

    @abstractmethod
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        To be implemented by subclasses
        
        Returns
        -------
            Dictionary with the parents of each group of variables.
        '''
        pass

    def extract_parents_time_and_memory(self) -> tuple[dict[int, list[int]], float, float]:
        '''
        Execute the extract_parents method and return the parents dict, the time that took to run the algorithm
        
        Returns:
        --------
            parents : dictionary of extracted parents for each group of variables
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
    