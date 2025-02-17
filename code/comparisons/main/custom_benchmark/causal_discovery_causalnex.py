import pandas as pd
from typing import Any
import numpy as np
import tigramite
from causalnex.structure import dynotears

# To admit the use of this package's data structures
from causalai.data.time_series import TimeSeriesData

from causal_discovery_base import CausalDiscoveryBase


class DynotearsWrapper(CausalDiscoveryBase):
    '''
    Wrapper for Granger algorithm
    '''
    def __init__(self, data: np.ndarray, max_lag: int, **kwargs):
        '''
        Initialize the Granger object
        Parameters:
        '''
        self.df = pd.DataFrame(data, columns=range(data.shape[1]))
        self.max_lag = max_lag
        self.kwargs = kwargs
        
        self.model = dynotears

    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Returns the parents dict
        Parameters:
            data : np.array with the data, shape (n_samples, n_features)
        '''
        graph_structure = self.model.from_pandas_dynamic(time_series=self.df, p=self.max_lag, **self.kwargs)
        print(f'{graph_structure.graph=}')
        
        return {}
