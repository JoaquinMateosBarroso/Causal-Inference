import time
import numpy as np
from memory_profiler import memory_usage


class CausalDiscoveryBase:
    '''
    Base class for causal discovery algorithms
    '''
    def __init__(self, data: np.array, **kwargs):
        '''
        To be implemented by subclasses
        :param data: np.array with the data
        '''
        pass
    
    def __extract_parents(self) -> dict[int, list[int]]:
        '''
        To be implemented by subclasses
        Returns a dictionary with the parents of each node
        '''
        pass
    
    def extract_parents_time_and_memory(self) -> tuple[dict[int, list[int]], float, int]:
        '''
        Execute the extract_parents method and return the parents dict, the time that took to run the algorithm
        '''
        tic = time.time()
        parents, memory = memory_usage( self.__extract_parents() )
        toc = time.time()
        execution_time = tic - toc
        
        return parents, execution_time, memory


