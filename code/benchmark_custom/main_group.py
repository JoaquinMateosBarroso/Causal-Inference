from typing import Any, Iterator, Union

from matplotlib import pyplot as plt

from benchmark_causal_discovery import BenchmarkGroupCausalDiscovery
import shutil
import os

from functions_test_data import changing_N_groups, changing_N_variables, changing_N_vars_per_group, changing_preselection_alpha, static_parameters
from group_causal_discovery import DimensionReductionGroupCausalDiscovery
from group_causal_discovery import MicroLevelGroupCausalDiscovery
from group_causal_discovery import HybridGroupCausalDiscovery

algorithms = {
    'hybrid': HybridGroupCausalDiscovery,
    'pca+pcmci': DimensionReductionGroupCausalDiscovery,
    'pca+dynotears': DimensionReductionGroupCausalDiscovery,
    'micro-level': MicroLevelGroupCausalDiscovery,
}
algorithms_parameters = {
    'pca+pcmci': {'dimensionality_reduction': 'pca', 'node_causal_discovery_alg': 'pcmci',
                            'node_causal_discovery_params': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05}},
    
    'pca+dynotears': {'dimensionality_reduction': 'pca', 'node_causal_discovery_alg': 'dynotears',
                            'node_causal_discovery_params': {'max_lag': 5, 'lambda_w': 0.05, 'lambda_a': 0.05}},
    
    'micro-level': {'node_causal_discovery_alg': 'pcmci',
                            'node_causal_discovery_params': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05}},
    
    'hybrid': {'dimensionality_reduction': 'pca', 'dimensionality_reduction_params': {'explained_variance_threshold': 0.5},
                            'node_causal_discovery_alg': 'pcmci',
                            'node_causal_discovery_params': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05},
                'verbose': 1},
}

data_generation_options = {
    'min_lag': 0,
    'max_lag': 5,
    'contemp_fraction': 0.25,
    'T': 1000, # Number of time points in the dataset
    'N_vars': 20, # Number of variables in the dataset
    'N_groups': 5, # Number of groups in the dataset
    'inner_group_crosslinks_density': 0.5,
    'outer_group_crosslinks_density': 0.5,
    'n_node_links_per_group_link': 2,
    # These parameters are used in generate_structural_causal_process:
    'dependency_coeffs': [-0.3, 0.3], # default: [-0.5, 0.5]
    'auto_coeffs': [0.4], # default: [0.5, 0.7]
    'noise_dists': ['gaussian'], # deafult: ['gaussian']
    'noise_sigmas': [0.2], # default: [0.5, 2]
    
    'dependency_funcs': ['linear']#, 'negative-exponential', 'sin', 'cos', 'step'], # Options: 'linear', 'negative-exponential', 'sin', 'cos', 'step'
}

benchmark_options = {
    'static_parameters': (static_parameters, {}),
    'changing_N_variables': (changing_N_variables,
                                    {'list_N_variables': [5]}),
    
    'changing_preselection_alpha': (changing_preselection_alpha,
                                    {'list_preselection_alpha': [0.01, 0.05, 0.1, 0.2]}),
    
    'changing_N_groups': (changing_N_groups,
                                    {'list_N_groups': [5, 10, 15, 20, 25, 30],
                                     'relation_vars_per_group': 3}),
    
    'chaning_N_vars_per_group': (changing_N_vars_per_group,
                                    {'list_N_vars_per_group': [2, 4, 6, 8, 10]})
}
chosen_option = 'chaning_N_vars_per_group'

def generate_parameters_iterator(algorithms_parameters, data_generation_options, 
                                 benchmark_options, chosen_option) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    '''
    Function to generate the parameters for the algorithms and the data generation.
    
    Args:
        algorithms_parameters : dict[str, dict[str, Any]]. Dictionary with the initial parameters for the algorithms.
        data_generation_options : dict[str, Any]. Dictionary with the options for the data generation.
        benchmark_options : dict[str, Tuple[Callable, dict[str, Any]]]. Dictionary with the options for the benchmark.
        chosen_option : str. The chosen option for the benchmark.
    
    Returns:
        parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]]. A function that returns the parameters for the algorithms and the data generation.
    '''    
    options_generator, options_kwargs = benchmark_options[chosen_option]
    for data_generation_options, algorithms_parameters in \
            options_generator(data_generation_options,
                                                  algorithms_parameters,
                                                  **options_kwargs):
        yield data_generation_options, algorithms_parameters



if __name__ == '__main__':
    plt.style.use('ggplot')
    
    benchmark = BenchmarkGroupCausalDiscovery()
    datasets_folder = 'toy_data'
    results_folder = 'results_group'
    execute_benchmark = True

    if execute_benchmark:
        parameters_iterator = generate_parameters_iterator(algorithms_parameters, data_generation_options,
                                                           benchmark_options, chosen_option)
        results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                            parameters_iterator=parameters_iterator,
                                            datasets_folder=datasets_folder,
                                            results_folder=results_folder,
                                            n_executions=5,
                                            scores=['f1', 'precision', 'recall', 'time', 'memory'],
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
    

