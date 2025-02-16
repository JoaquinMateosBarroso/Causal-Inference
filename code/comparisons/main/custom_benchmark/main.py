from typing import Any, Iterator, Union

from matplotlib import pyplot as plt
import numpy as np

from benchmark_causal_discovery import BenchmarkCausalDiscovery
from causal_discovery_tigramite import PCMCIModifiedWrapper, PCMCIWrapper, LPCMCIWrapper, PCStableWrapper
from causal_discovery_causalai import GrangerWrapper, VARLINGAMWrapper
import shutil
import os


algorithms = {
    'pcmci': PCMCIWrapper,
    # 'fullpcmci': PCMCIWrapper,
    # 'fastpcmci': PCMCIWrapper,
    # 'pcmci-cmiknn': PCMCIWrapper,
    'pcmci-modified': PCMCIModifiedWrapper,
    'pc-stable': PCStableWrapper,
    # 'lpcmci': LPCMCIWrapper,
    'granger': GrangerWrapper,
    'varlingam': VARLINGAMWrapper,
}
def generate_parameters_iterator() -> Iterator[Union[dict[str, Any], dict[str, Any]]]:
    '''
    Function to generate the parameters for the algorithms and the data generation.
    
    Returns:
        parameters_iterator: function[dict[str, Any], dict[str, Any]]. A function that returns the parameters for the algorithms and the data generation.
    '''
    algorithms_parameters = {
        # pc_alpha to None performs a search for the best alpha
        'pcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3},
        'fullpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
        'fastpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 1, 'max_conds_dim': 3},
        'pcmci-cmiknn': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'cond_ind_test': 'cmiknn'},
        'pcmci-modified': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
        'pc-stable': {'pc_alpha': 0.01, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
        'lpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3},
        'granger': {'cv': 5, 'min_lag': 1, 'max_lag': 3},
        'varlingam': {'min_lag': 1, 'max_lag': 3},
    }
    options = {
        'max_lag': 10,
        'dependency_funcs': [ lambda x: x, # linear
                              lambda x: np.tanh(x), # 
                              lambda x: 1 / (1 + np.exp(-x)), # logistic
                              lambda x: np.sin(x) + np.log(1+np.abs(x)), # sin + log
                            ],
        'graph_density': 0.05, # Portion of posible links to generate
        'T': 500, # Number of time points in the dataset
        'N': 50, # Number of variables in the dataset
        # These parameters are used in generate_structural_causal_process:
        'dependency_coeffs': [-0.4, 0.4], # default: [-0.5, 0.5]
        'auto_coeffs': [0.5], # default: [0.5, 0.7]
        'noise_dists': ['gaussian'], # deafult: ['gaussian']
        'noise_sigmas': [0.2], # default: [0.5, 2]
    }
    
    for N_variables in [20]:
        # Increase data points in the same proportion as max_lag 
        options['T'] = int(options['T'] * (N_variables / options['N']))
        
        options['N'] = N_variables
        
        yield algorithms_parameters, options


if __name__ == '__main__':
    benchmark = BenchmarkCausalDiscovery()
    plt.style.use('ggplot')
    datasets_folder = 'toy_data'; results_folder = 'results'
    results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                         parameters_iterator=generate_parameters_iterator(),
                                         datasets_folder='toy_data',
                                         results_folder='results',
                                         n_executions=3,
                                         scores=['f1', 'precision', 'recall', 'time', 'memory'],
                                         verbose=1)
    
    benchmark.plot_ts_datasets('toy_data')
    
    benchmark.plot_moving_results('results', x_axis='N')
    benchmark.plot_particular_result('results')


    # Copy toy_data folder inside results folder, to have the datasets used in the benchmark
    destination_folder = os.path.join(results_folder, datasets_folder)

    if not os.path.exists(destination_folder):
        shutil.copytree(datasets_folder, destination_folder)