import pandas as pd
from typing import Any
import numpy as np
from causalnex.structure import dynotears

from group_causation.causal_discovery_algorithms.causal_discovery_base import CausalDiscoveryBase


class DynotearsWrapper(CausalDiscoveryBase):
    '''
    Wrapper for DYNOTEARS algorithm
    
    Args:
        data : np.array with the data, shape (n_samples, n_features)
        max_lag : maximum lag to consider
    '''
    def __init__(self, data: np.ndarray, max_lag: int, **kwargs):
        super().__init__(data, **kwargs)
        
        self.df = pd.DataFrame(self.data, columns=range(self.data.shape[1]))
        self.max_lag = max_lag
        self.kwargs = kwargs
        
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Returns the parents dict
        
        Args:
            data : np.array with the data, shape (n_samples, n_features)
        '''
        graph_structure = dynotears.from_pandas_dynamic(time_series=self.df,
                                                        p=self.max_lag,
                                                        **self.kwargs)
        
        parents_dict = get_parents_from_causalnex_edges(graph_structure.edges)
        
        return parents_dict


def get_parents_from_causalnex_edges(edges: list[tuple[str, str]]) -> dict[int, list[int]]:
    '''
    Function to extract the parents from the edges list.
    
    Args:
        edges : list of tuples with the edges, where each tuple is (parent, child),
                being a node represented by '{origin}_lag{lag}'. E.g. '0_lag1'.
    Returns:
        parents : dict with the parents of each node.
    '''
    parents = {}
    for edge in edges:
        origin, destiny = edge
        child = origin.split('_lag')
        child = (int(child[0]), -int(child[1]))
        parent = int(destiny.split('_lag')[0])
        if child[1] <  0: # Include just lagged edges
            parents[parent] = parents.get(parent, []) + [child]
        
    return parents
