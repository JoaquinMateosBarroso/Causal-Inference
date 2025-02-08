import time

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI


def extract_parents_pcmci(dataframe):
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



