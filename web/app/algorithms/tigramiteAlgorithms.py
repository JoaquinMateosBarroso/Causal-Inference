import pandas as pd

from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.jpcmciplus import JPCMCIplus

from tigramite.independence_tests.parcorr import ParCorr



tigramite_algorithms = ['PCMCI', 'LPCMCI', 'JPCMCIplus']
def runTigramiteCausalDiscovery(algorithmName: str,
                                   data: pd.DataFrame,
                                   max_lag: int,
                                   verbosity: int,
                                   cond_ind_test = ParCorr(),
                                   **kwargs)->dict:
    """
    Run the specified Tigramite algorithm.
    
    Args:
        algorithmName (str): The name of the algorithm to run.
        data (pd.DataFrame): The data to run the algorithm on.
        max_lag (int): The maximum lag to consider.
        verbosity (int): The verbosity level of the algorithm.
        cond_ind_test: The conditional independence test to use.
        **kwargs: Additional arguments to pass to the algorithm.
        
    Returns:
        dict: The result of the algorithm.
    """
    if algorithmName == 'PCMCI':
        pcmci = PCMCI(dataframe=data, cond_ind_test=cond_ind_test, verbosity=verbosity)
        results = pcmci.run_pcmci(tau_max=max_lag)
    elif algorithmName == 'LPCMCI':
        lpcmci = LPCMCI(dataframe=data, cond_ind_test=cond_ind_test, verbosity=verbosity)
        results = lpcmci.run_lpcmci(tau_max=max_lag, tau_min=1)
    elif algorithmName == 'JPCMCIplus':
        rpcmci = JPCMCIplus(dataframe=data, cond_ind_test=cond_ind_test, verbosity=verbosity)
        results = rpcmci.run_jpcmciplus(tau_max=max_lag)
    else:
        raise ValueError(f'Unknown algorithm: {algorithmName}')
    
    return results