import time
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from memory_profiler import memory_usage


class EdgeDirection(Enum):
    '''
    Enum class to indicate the direction of the edge in the causal graph
    '''
    LEFT2RIGHT = 1
    RIGHT2LEFT = 2
    BIDIRECTED = 3
    NONE = 4


class DirectionExtractorBase(ABC): # Abstract class
    '''
    Base class for direction extraction algorithms on groups of variables
    '''
    def __init__(self, data: np.ndarray, groups: list[set[int]], **kwargs):
        '''
        Create an object that is able to predict over groups of time series variables
        
        Parameters:
            data : np.array with the data, shape (n_samples, n_variables)
            groups : list[set[int]] list with the sets that will compound each group of variables.
                We will suppose that the groups are known beforehand.
                The index of a group will be considered as its position in groups list.
        '''
        self.data = data
        self.groups = groups
        # Create a dictionary with the set of data of each group
        self.groups_data = {i: data[:, list(group)] for i, group in enumerate(groups)}
        # This graph will be a dictionary whose keys will be pairs of groups index and 
        # values are the direction of the edge between them
        self.directions_graph: dict = None
        
        self.extra_args = kwargs

    @abstractmethod
    def identify_causal_direction(self, X: np.ndarray, Y: np.ndarray) -> EdgeDirection:
        '''
        To be implemented by subclasses
        
        Parameters:
            X : np.ndarray. Data of the first group of variables, shape (n_samples, n_variables)
            Y : np.ndarray. Data of the second group of variables, shape (n_samples, n_variables)
        Returns:
            The direction of the edge in the causal graph
        '''
        pass
    
    def extract_direction(self, X_index: int, Y_index: int) -> EdgeDirection:
        '''
        Uses the specific algorithm to extract the direction of the edge between two groups of variables
        present in the class data
        
        Returns:
            The direction of the edge in the causal graph
        '''
        X = self.groups_data[X_index]
        Y = self.groups_data[Y_index]
        
        return self.identify_causal_direction(X, Y)
    
    def extract_graph_directions(self) -> dict[set[int, int], EdgeDirection]:
        '''
        Extract the directions of the edges in the causal graph for each group of variables
        
        Returns:
            dict[int, EdgeDirection] : dictionary with the directions of the edges in the causal graph
        '''
        self.directions = {}
        for i in range(len(self.groups)):
            for j in range(i+1, len(self.groups)):
                self.directions[(i, j)] = self.extract_direction(i, j)
                self.directions[(j, i)] = _opposite_direction(self.directions[(i, j)])

        return self.directions


def _opposite_direction(direction: EdgeDirection) -> EdgeDirection:
    '''
    Get the direction that would have the adjacent node in the same edge
    
    Parameters:
        direction : EdgeDirection. The direction of the edge
    
    Returns:
        EdgeDirection. The opposite direction of the edge
    '''
    if direction == EdgeDirection.LEFT2RIGHT:
        return EdgeDirection.RIGHT2LEFT
    elif direction == EdgeDirection.RIGHT2LEFT:
        return EdgeDirection.LEFT2RIGHT
    else:
        return direction