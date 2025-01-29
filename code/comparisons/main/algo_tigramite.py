import time

import tigramite
import tigramite.data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from causalai.data.time_series import TimeSeriesData


def convert_to_tigramite_format(data: TimeSeriesData) -> tigramite.data_processing.DataFrame:
    '''
    Convert the data to tigramite dataframe format
    Note: It only works if there is only one data array in the data object
    '''
    names = data.var_names
    data_arrays = data.data_arrays
    dataframe = tigramite.data_processing.DataFrame(data_arrays[0], var_names=names)
    
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
        
        results = self.pcmci.run_fullci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
        
        parents_graph = self.pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
        
        result = convert_to_causalai_format(parents_graph, self.data.var_names)
        
        return result

def extract_parents_lpcmci(dataframe) -> tuple[dict, float]:
    '''
    Returns the parents dict and the time that took to run the algorithm
    '''
    parcorr = ParCorr(significance='analytic')
    lpcmci = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0
    )
    
    tau_max = 3
    pc_alpha = 0.05 # Default value
    
    start_time = time.time()
    lpcmci.run_lpcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Return parents dict manually
    parents = dict()
    for j in range(lpcmci.N):
        for ((i, lag_i), link) in lpcmci.graph_dict[j].items():
            if len(link) > 0 and (lag_i < 0 or i < j):
                parents[j] = parents.get(j, []) + [(i, lag_i)]
    
    return parents, execution_time



