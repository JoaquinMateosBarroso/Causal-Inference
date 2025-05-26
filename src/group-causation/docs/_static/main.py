from typing import Any, Iterator, Union

from matplotlib import pyplot as plt
import numpy as np

from benchmark import BenchmarkCausalDiscovery
from micro_causal_discovery import PCMCIModifiedWrapper, PCMCIWrapper, LPCMCIWrapper, PCStableWrapper
from micro_causal_discovery import GrangerWrapper, VARLINGAMWrapper
from micro_causal_discovery import DynotearsWrapper
import shutil
import os

# Ignore FutureWarnings, due to versions of libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from functions_test_data import changing_N_variables, changing_preselection_alpha, static_parameters


algorithms = {
    'pcmci': PCMCIWrapper,
    'dynotears': DynotearsWrapper,
    'granger': GrangerWrapper,
    'varlingam': VARLINGAMWrapper,
    # 'pc-stable': PCStableWrapper,
    
    # 'fullpcmci': PCMCIWrapper,
    # 'lpcmci': LPCMCIWrapper,
    
    # This works bad with big datasets
    # 'pcmci-modified': PCMCIModifiedWrapper,
}
algorithms_parameters = {
    # pc_alpha to None performs a search for the best alpha
    'pcmci':     {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05, 'cond_ind_test': 'parcorr'},
    'granger':   {'min_lag': 0, 'max_lag': 5, 'cv': 5, },
    'varlingam': {'min_lag': 0, 'max_lag': 5},
    'dynotears': {              'max_lag': 5, 'max_iter': 1000, 'lambda_w': 0.05, 'lambda_a': 0.05},
    'pc-stable': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': None, 'max_combinations': 100, 'max_conds_dim': 5},
    
    'pcmci-modified': {'pc_alpha': 0.05, 'min_lag': 1, 'max_lag': 5, 'max_combinations': 1,
                        'max_summarized_crosslinks_density': 0.2, 'preselection_alpha': 0.05},
    'fullpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
    'lpcmci': {'pc_alpha': 0.01, 'min_lag': 1, 'max_lag': 3},
}

data_generation_options = {
    'min_lag': 0,
    'max_lag': 5,
    'contemp_fraction': 1, # Fraction of contemporaneous links; between 0 and 1
    'crosslinks_density': 0.5, # Portion of links that won't be in the kind of X_{t-1}->X_t; between 0 and 1
    'T': 2000, # Number of time points in the dataset
    'N_vars': 5, # Number of variables in the dataset
    'confounders_density': 0.2, # Portion of dataset that will be overgenerated as confounders; between 0 and inf
    # These parameters are used in generate_structural_causal_process:
    'dependency_coeffs': [-0.3, 0.3], # default: [-0.5, 0.5]
    'auto_coeffs': [0.7], # default: [0.5, 0.7]
    'noise_dists': ['gaussian'], # deafult: ['gaussian']
    'noise_sigmas': [0.3], # default: [0.5, 2]
    
    'dependency_funcs': ['linear', 'negative-exponential', 'sin', 'cos', 'step'],
}

benchmark_options = {
    'changing_N_variables': (changing_N_variables, 
                                    {'list_N_variables': [5]}),
    
    'changing_preselection_alpha': (changing_preselection_alpha,
                                    {'list_preselection_alpha': [0.01, 0.05, 0.1, 0.2]}),
    'static': (static_parameters, {}),
}
chosen_option = 'static'


if __name__ == '__main__':
    plt.style.use('ggplot')
    
    benchmark = BenchmarkCausalDiscovery()
    datasets_folder = 'toy_data'
    results_folder = 'results'
    execute_benchmark = True

    if execute_benchmark:    
        options_generator, options_kwargs = benchmark_options[chosen_option]
        parameters_iterator = options_generator(data_generation_options,
                                                    algorithms_parameters,
                                                    **options_kwargs)
        
        results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                            parameters_iterator=parameters_iterator,
                                            datasets_folder=datasets_folder,
                                            results_folder=results_folder,
                                            n_executions=5,
                                            verbose=1)
    
    benchmark.plot_ts_datasets(datasets_folder)
    
    benchmark.plot_moving_results(results_folder, x_axis='N_vars')
    # Save results for whole graph scores
    benchmark.plot_particular_result(results_folder, dataset_iteration_to_plot=0)
    # Save results for summary graph scores
    benchmark.plot_particular_result(results_folder, results_folder + '/summary',
                                     scores=[f'{score}_summary' for score in \
                                                    ['shd', 'f1', 'precision', 'recall']],
                                     dataset_iteration_to_plot=0)

    # Copy toy_data folder inside results folder, to have the datasets used in the benchmark
    destination_folder = os.path.join(results_folder, datasets_folder)
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(datasets_folder, destination_folder)
    

