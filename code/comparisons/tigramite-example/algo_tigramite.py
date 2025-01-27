import time

import tigramite
import tigramite.data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from causalai.data.time_series import TimeSeriesData


class Extractor_PCMCI():
    def __init__(self, data: TimeSeriesData):
        '''
        :param data: this is a TabularData object and contains attributes likes data.data_arrays, which is a
            list of numpy array of shape (observations N, variables D).
        :type data: TabularData object
        '''
        self.data = data
        names = data.var_names
        data_arrays = data.data_arrays
        print(data_arrays)
        self.dataframe = tigramite.data_processing.DataFrame(data_arrays, var_names=names)
        

    def run(self, **kargs):
        tau_max = kargs['tau_max']
        pc_alpha = kargs['pc_alpha']
        
        result = {name: {'parents': []} for name in self.data.var_names} # result must follow this format
        # do something and compute parents
        return result

def extract_parents_pcmci(df):
    '''
    Returns the parents dict and the time that took to run the algorithm
    '''
    
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0
    )

    tau_max = 3
    pc_alpha = None # Optimize in a list
    
    start_time = time.time()
    results = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    end_time = time.time()
    execution_time = end_time - start_time

    parents = pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
    
    return parents, execution_time

def extract_parents_pcmciplus(dataframe):
    '''
    Returns the parents dict and the time that took to run the algorithm
    '''
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0
    )

    tau_max = 3
    pc_alpha = None # Default value
    
    
    start_time = time.time()
    results = pcmci.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    end_time = time.time()
    execution_time = end_time - start_time

    
    
    parents = pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
    return parents, execution_time

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



