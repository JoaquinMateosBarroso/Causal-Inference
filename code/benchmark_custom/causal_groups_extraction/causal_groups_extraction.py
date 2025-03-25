import time
import numpy as np
from abc import ABC, abstractmethod
from memory_profiler import memory_usage



class CausalGroupsExtractorBase(ABC): # Abstract class
    '''
    Base class to extract a set of groups of variables that may be used to later
    predict the causal structure over these groups of variables
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        standarize : bool indicating if the data should be standarized before applying the algorithm
    '''
    def __init__(self, data: np.ndarray, standarize: bool=True, **kwargs):
        if standarize:
            self.data = (data - data.mean(axis=0)) / data.std(axis=0)
        else:
            self.data = data
        self.extra_args = kwargs
    
    @abstractmethod
    def extract_groups(self) -> tuple[list[set[int]]]:
        '''
        To be implemented by subclasses
        
        Returns
            groups : list of sets with the variables that compound each group
        '''
        pass
    
    def extract_groups_time_and_memory(self) -> tuple[dict[int, list[int]], float, float]:
        '''
        Execute the extract_groups method and return the set of groups, the time that took to run the algorithm
        
        Returns:
            groups : list of sets with the variables that compound each group
            execution_time : execution time in seconds
            memory : volatile memory used by the process, in MB
        '''
        tic = time.time()
        memory, groups = memory_usage( self.extract_groups, retval=True, include_children=True, multiprocess=True)
        toc = time.time()
        execution_time = toc - tic
        
        memory = max(memory) - min(memory)  # Memory usage in MiB
        memory = memory * 1.048576 # Exact division   1024^2 / 1000^2
        
        return groups, execution_time, memory