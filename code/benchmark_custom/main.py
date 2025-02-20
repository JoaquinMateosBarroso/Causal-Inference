from typing import Any, Iterator, Union

from matplotlib import pyplot as plt
import numpy as np

from benchmark_causal_discovery import BenchmarkCausalDiscovery
from causal_discovery_tigramite import PCMCIModifiedWrapper, PCMCIWrapper, LPCMCIWrapper, PCStableWrapper
from causal_discovery_algorithms.causal_discovery_causalai import GrangerWrapper, VARLINGAMWrapper
from causal_discovery_algorithms.causal_discovery_causalnex import DynotearsWrapper
import shutil
import os

from functions_test_data import changing_N_variables, changing_preselection_alpha


algorithms = {
    'pcmci-modified': PCMCIModifiedWrapper,
    'pcmci': PCMCIWrapper,
    'dynotears': DynotearsWrapper,
    'granger': GrangerWrapper,
    'varlingam': VARLINGAMWrapper,
    'pc-stable': PCStableWrapper,
    
    # 'fullpcmci': PCMCIWrapper,
    # 'lpcmci': LPCMCIWrapper,
}




benchmark_options = {
'changing_N_variables': changing_N_variables,
'changing_preselection_alpha': changing_preselection_alpha,
}
chosen_option = 'changing_N_variables'

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
        'dynotears': {'max_lag': 3, 'max_iter': 1000, 'lambda_w': 0.05, 'lambda_a': 0.05},
        'pc-stable': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
        'pcmci-modified': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 1, 'max_conds_dim': 5,
                           'max_crosslink_density': 0.2, 'preselection_alpha': 0.01},
        
        
        'fullpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
        'lpcmci': {'pc_alpha': 0.05, 'min_lag': 1, 'max_lag': 3},
    }
    data_generation_options = {
        'max_lag': 20,
        'crosslinks_density': 0.75, # Portion of links that won't be in the kind of X_{t-1}->X_t
        'T': 500, # Number of time points in the dataset
        'N': 20, # Number of variables in the dataset
        # These parameters are used in generate_structural_causal_process:
        'dependency_coeffs': [-0.4, 0.4], # default: [-0.5, 0.5]
        'auto_coeffs': [0.7], # default: [0.5, 0.7]
        'noise_dists': ['gaussian'], # deafult: ['gaussian']
        'noise_sigmas': [0.2], # default: [0.5, 2]
        
        'dependency_funcs': ['linear', 'negative-exponential', 'sin', 'cos', 'step'],
    }
    
    for data_generation_options, algorithms_parameters in \
            benchmark_options[chosen_option](data_generation_options,
                                                  algorithms_parameters):
        yield data_generation_options, algorithms_parameters
    


if __name__ == '__main__':
    plt.style.use('ggplot')
    
    benchmark = BenchmarkCausalDiscovery()
    datasets_folder = 'toy_data'
    results_folder = 'results'
    execute_benchmark = True

    if execute_benchmark:
        results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                            parameters_iterator=generate_parameters_iterator(),
                                            datasets_folder=datasets_folder,
                                            results_folder=results_folder,
                                            n_executions=3,
                                            scores=['f1', 'precision', 'recall', 'time', 'memory'],
                                            verbose=1)
    
    benchmark.plot_ts_datasets(datasets_folder)
    
    benchmark.plot_moving_results(results_folder, x_axis='preselection_alpha')
    benchmark.plot_particular_result(results_folder)


    # Copy toy_data folder inside results folder, to have the datasets used in the benchmark
    destination_folder = os.path.join(results_folder, datasets_folder)
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(datasets_folder, destination_folder)
    

