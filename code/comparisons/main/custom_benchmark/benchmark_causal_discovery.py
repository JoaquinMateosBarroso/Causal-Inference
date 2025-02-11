# %%
import numpy as np
import pandas as pd
from create_toy_datasets import generate_toy_data
from functions_test_toy_data import test_toy_data
from causal_discovery_base import CausalDiscoveryBase
from causal_discovery_tigramite import PCMCIWrapper, LPCMCIWrapper
from typing import Any

# %%
from itertools import product

def benchmark_causal_discovery(algorithms: dict[str, CausalDiscoveryBase],
                               algorithms_parameters: dict[str, list],
                               options: dict[str, list] = None,
                               datasets: list[np.ndarray] = None) \
                                       -> dict[str, dict[str, Any]]:
    ''''
    Function to execute a series of algorithms for causal discovery over time series datasets,
        using a series of parameters for algorithms and options in the creation of the datasets.
    Parameters:
        algorithms : dict[str, CausalDiscoveryBase]
            A dictionary where keys are the names of the algorithms and values are instances of the algorithms to be tested.
        algorithms_parameters : dict[str, list]
            A dictionary where keys are the names of the algorithms and values are lists of parameters for each algorithm.
        options : dict[str, list], optional
            A dictionary where keys are the names of the options and values are lists of possible values for each option.
        datasets : list[np.ndarray], optional
            A list of numpy arrays representing the datasets to be used in the benchmark.
    Returns:
        dict[str, dict[str, Any]]
            A dictionary where keys are the names of the algorithms and values are dictionaries containing the results of the benchmark for each algorithm.
    '''
    # A list whose items are the lists of dictionaries of results and parameters of the different executions
    results = {
        'pcmci': [],
        'pcmciplus': [],
        'lpcmci': [],
    }
    
    combinations = list(product(*options.values()))
    
    for iteration, combination in enumerate(combinations):
        print(combination)
        current_option = dict(zip(options.keys(), combination))
        generate_toy_data(iteration, **current_option)
        
        for name, algorithm in algorithms.items():
            algorithm_results = test_toy_data(iteration, algorithm)
            algorithm_results.update(current_option) # Include the parameters in the information for results
            results[name].append(algorithm_results)
    
    
    # Save the results in a csv file
    for name in algorithms.keys():
        df = pd.DataFrame(results[name])
        df = df[['T', 'N', 'L', 'dependency_funcs', 'max_lag', 'time', 'f1', 'precision', 'recall']]
        df.to_csv(f'results_{name}.csv', index=False)
        
algorithms = {
    'pcmci': PCMCIWrapper,
    'pcmciplus': LPCMCIWrapper,
}

algorithms_parameters = {
    
}
options = {
    'max_lag': [3],
    'dependency_funcs': [ ['linear', 'nonlinear'] ],
    'L': [5], # Number of cross-links in the dataset
    'T': [100], # N time points in the dataset
    'N': [5], # Number of variables in the dataset
}

if __name__ == '__main__':
    benchmark_causal_discovery()


