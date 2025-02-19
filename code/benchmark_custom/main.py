from typing import Any, Iterator, Union

from matplotlib import pyplot as plt
import numpy as np

from benchmark_causal_discovery import BenchmarkCausalDiscovery
from causal_discovery_tigramite import PCMCIModifiedWrapper, PCMCIWrapper, LPCMCIWrapper, PCStableWrapper
from causal_discovery_algorithms.causal_discovery_causalai import GrangerWrapper, VARLINGAMWrapper
from causal_discovery_algorithms.causal_discovery_causalnex import DynotearsWrapper
import shutil
import os


algorithms = {
    'pcmci-modified': PCMCIModifiedWrapper,
    # 'pcmci': PCMCIWrapper,
    # 'dynotears': DynotearsWrapper,
    # 'granger': GrangerWrapper,
    # 'varlingam': VARLINGAMWrapper,
    # 'pc-stable': PCStableWrapper,
    
    # 'fullpcmci': PCMCIWrapper,
    # 'fastpcmci': PCMCIWrapper,
    # 'lpcmci': LPCMCIWrapper,
}
def generate_parameters_iterator() -> Iterator[Union[dict[str, Any], dict[str, Any]]]:
    '''
    Function to generate the parameters for the algorithms and the data generation.
    
    Returns:
    --------
        parameters_iterator: function[dict[str, Any], dict[str, Any]]. A function that returns the parameters for the algorithms and the data generation.
    '''
    algorithms_parameters = {
        # pc_alpha to None performs a search for the best alpha
        'pcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'cond_ind_test': 'parcorr'},
        'granger': {'cv': 5, 'min_lag': 1, 'max_lag': 3},
        'varlingam': {'min_lag': 1, 'max_lag': 3},
        'dynotears': {'max_lag': 3, 'max_iter': 10000},
        'pc-stable': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
        'pcmci-modified': {'pc_alpha': 0.01, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 1, 'max_conds_dim': 5,
                           'max_crosslink_density': 0.5, 'preselection_alpha': 0.01},
        
        
        'fullpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
        'fastpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 1, 'max_conds_dim': 3},
        'lpcmci': {'pc_alpha': 0.05, 'min_lag': 1, 'max_lag': 3},
    }
    options = {
        'max_lag': 5,
        'dependency_funcs': [
                                lambda x: 0.5*x, # linear with 
                                lambda x: np.exp(-abs(x)) - 1 + np.tanh(x),
                                #   lambda x: x + x**2 * np.exp(-(x**2) / 2), # logistic
                                lambda x: np.sin(x), # + np.log(1+np.abs(x)), # sin + log
                                lambda x: np.cos(x),
                                lambda x: 1 if x > 0 else 0, # step function
                            ],
        'crosslinks_density': 0.75, # Portion of links that won't be in the kind of X_{t-1}->X_t
        'T': 500, # Number of time points in the dataset
        'N': 20, # Number of variables in the dataset
        # These parameters are used in generate_structural_causal_process:
        'dependency_coeffs': [-0.4, 0.4], # default: [-0.5, 0.5]
        'auto_coeffs': [0.7], # default: [0.5, 0.7]
        'noise_dists': ['gaussian'], # deafult: ['gaussian']
        'noise_sigmas': [0.2], # default: [0.5, 2]
    }
    
    # # To modify N variables together with the number of data points
    # for N_variables in [10, 20, 30, 40, 50]:
    #     # Increase data points in the same proportion as N_vars 
    #     options['T'] = int(options['T'] * (N_variables / options['N']))
        
    #     options['N'] = N_variables
        
    #     # options['max_lag'] = max_lag
        
    #     for algorithm_paramters in algorithms_parameters.values():
    #         algorithm_paramters['max_lag'] = options['max_lag']
        
    #     yield algorithms_parameters, options
    
    # To modify preselection_alpha
    
    for preselection_alpha in [0.01, 0.05, 0.1, 0.2]:
        algorithms_parameters['pcmci-modified']['preselection_alpha'] = preselection_alpha
        
        yield algorithms_parameters, options


if __name__ == '__main__':
    plt.style.use('ggplot')
    
    benchmark = BenchmarkCausalDiscovery()
    datasets_folder = 'toy_data'
    results_folder = 'results'
    
    results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                         parameters_iterator=generate_parameters_iterator(),
                                         datasets_folder=datasets_folder,
                                         results_folder=results_folder,
                                         n_executions=3,
                                         scores=['f1', 'precision', 'recall', 'time', 'memory'],
                                         verbose=1)
    
    benchmark.plot_ts_datasets(datasets_folder)
    
    benchmark.plot_moving_results(results_folder, x_axis='N')
    benchmark.plot_particular_result(results_folder)


    # Copy toy_data folder inside results folder, to have the datasets used in the benchmark
    destination_folder = os.path.join(results_folder, datasets_folder)
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(datasets_folder, destination_folder)
    
