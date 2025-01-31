import numpy as np
from typing import Union

import tigramite
import tigramite.data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from causalai.data.time_series import TimeSeriesData
from causalai.models.time_series.pc import PC
from causalai.models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests

from discretization.methods_discretization import sax_method

def convert_to_tigramite_format(data: Union[TimeSeriesData, np.ndarray]) -> tigramite.data_processing.DataFrame:
    '''
    Convert the data to tigramite dataframe format
    Note: It only works if there is only one data array in the data object
    '''
    if isinstance(data, TimeSeriesData):
        names = data.var_names
        data_arrays = data.data_arrays
        dataframe = tigramite.data_processing.DataFrame(data_arrays[0], var_names=names)
    
    elif isinstance(data, np.ndarray):
        dataframe = tigramite.data_processing.DataFrame(data, var_names=[str(i) for i in range(data.shape[1])])
    
    return dataframe

def convert_to_causalai_format(parents_graph: dict[int, tuple[int, int]], names: list[str]) -> dict:
    '''
    Convert the graph to causalai format
    '''
    parents_getter = lambda parents: \
            [(str(index), time_lag) for index, time_lag in parents]
    
    return {name: {'parents': parents_getter(parents)} for name, parents \
                                             in zip(names, parents_graph.values())}

class Extractor_PCMCI():
    def __init__(self, data: TimeSeriesData, cond_ind_test: str = 'parcorr'):
        '''
        :param data: this is a TabularData object and contains attributes likes data.data_arrays, which is a
            list of numpy array of shape (observations N, variables D).
        :type data: TabularData object
        '''
        self.data = data
        self.dataframe = convert_to_tigramite_format(data)
        
        self.cond_ind_test = {'parcorr': ParCorr(significance='analytic')}[cond_ind_test]
        
        self.pcmci = PCMCI(dataframe=self.dataframe,
                           cond_ind_test=self.cond_ind_test)
        
    def run(self, **kargs):
        '''
        Get the parents graph
        '''
        tau_max = kargs['tau_max']
        pc_alpha = kargs['pc_alpha']
        
        results = self.pcmci.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
        
        parents_graph = self.pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
        
        result = convert_to_causalai_format(parents_graph, self.data.var_names)
        
        return result

class Extractor_LPCMCI():
    def __init__(self, data: TimeSeriesData, cond_ind_test: str = 'parcorr'):
        '''
        :param data: this is a TabularData object and contains attributes likes data.data_arrays, which is a
            list of numpy array of shape (observations N, variables D).
        :type data: TabularData object
        '''
        self.data = data
        self.dataframe = convert_to_tigramite_format(data)
        
        self.cond_ind_test = {'parcorr': ParCorr(significance='analytic')}[cond_ind_test]
        
        self.lpcmci = LPCMCI(
            dataframe=self.dataframe,
            cond_ind_test=self.cond_ind_test)
        

    def run(self, **kargs):
        tau_max = kargs['tau_max']
        pc_alpha = kargs['pc_alpha']
        
        self.lpcmci.run_lpcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)

        # Return parents dict manually
        graph_parents = dict()
        for j in range(self.lpcmci.N):
            current_parents = []
            for ((i, lag_i), link) in self.lpcmci.graph_dict[j].items():
                if len(link) > 0 and (lag_i < 0 or i < j):
                    current_parents.append((str(i), lag_i))
            graph_parents[str(j)] = {'parents': current_parents}

        return graph_parents

class Extractor_FullCI():
    def __init__(self, data: TimeSeriesData, cond_ind_test: str = 'parcorr'):
        '''
        :param data: this is a TabularData object and contains attributes likes data.data_arrays, which is a
            list of numpy array of shape (observations N, variables D).
        :type data: TabularData object
        '''
        self.data = data
        self.dataframe = convert_to_tigramite_format(data)
        
        self.cond_ind_test = {'parcorr': ParCorr(significance='analytic')}[cond_ind_test]
        
        self.pcmci = PCMCI(dataframe=self.dataframe,
                           cond_ind_test=self.cond_ind_test)
        

    def run(self, **kargs):
        '''
        Get the parents graph
        '''
        tau_max = kargs['tau_max']
        pc_alpha = kargs['pc_alpha']
        
        results = self.pcmci.run_fullci(tau_min=1, tau_max=tau_max, alpha_level=pc_alpha)
        
        parents_graph = self.pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
        
        result = convert_to_causalai_format(parents_graph, self.data.var_names)
        
        return result


class Extractor_DiscretizedPC():
    def __init__(self, data: TimeSeriesData, cond_ind_test: str = 'CMIsymb',
                            n_symbs: int = 5):
        '''
        :param data: this is a TabularData object and contains attributes likes data.data_arrays, which is a
            list of numpy array of shape (observations N, variables D).
        :type data: TabularData object
        '''
        self.data = data
        
        data_array = self.data.data_arrays[0]
        discretized_data = np.zeros_like(data_array, dtype=np.int32)
        discretized_datas = []
        for i in range(data_array.shape[1]):
            discretized_datas.append( sax_method(data_array[:, i], n_bins=n_symbs,
                                          expand=False, use_labels=True) )
            # Convert str array to int array
            discretized_datas[-1] = np.unique(discretized_datas[-1],
                                             return_inverse=True)[1]
        
        discretized_data = np.stack(discretized_datas, axis=1)
        
        self.discretized_data = convert_to_tigramite_format(discretized_data)
        
        cond_ind_test = {'CMIsymb': CMIsymb(significance='fixed_thres')}[cond_ind_test]
        self.pcmci = PCMCI(dataframe=self.discretized_data,
                           cond_ind_test=cond_ind_test)


    def run(self, **kargs):
        '''
        Generate the parents graph
        '''
        print('started DPC')
        tau_max = kargs['tau_max']
        pc_alpha = kargs['pc_alpha']
        
        results = self.pcmci.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
        
        parents_graph = self.pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
        
        result = convert_to_causalai_format(parents_graph, self.data.var_names)
        
        print(result)
        
        return result

