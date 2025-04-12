'''
Module with the base class for causal discovery algorithms.
'''


import time
import numpy as np
from abc import ABC, abstractmethod
from memory_profiler import memory_usage

from group_causation.causal_discovery_base import CausalDiscoveryBase

class MicroCausalDiscoveryBase(CausalDiscoveryBase):
    '''
    Class for micro causal discovery algorithms
    Inherits from MicroCausalDiscoveryBase
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        standarize : bool indicating if the data should be standarized. default=True
    '''