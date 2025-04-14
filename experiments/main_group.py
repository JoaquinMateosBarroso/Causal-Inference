from matplotlib import pyplot as plt
import numpy as np

from group_causation.benchmark import BenchmarkGroupCausalDiscovery
import os

from group_causation.utils import changing_N_groups, changing_N_variables, changing_N_vars_per_group, changing_alg_params, changing_preselection_alpha, static_parameters
from group_causation.group_causal_discovery import DimensionReductionGroupCausalDiscovery
from group_causation.group_causal_discovery import MicroLevelGroupCausalDiscovery
from group_causation.group_causal_discovery import HybridGroupCausalDiscovery

algorithms = {
    'group-embedding': HybridGroupCausalDiscovery,
    'subgroups': HybridGroupCausalDiscovery,
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
    
    'group-embedding': {'dimensionality_reduction': 'pca', 
               'dimensionality_reduction_params': {'explained_variance_threshold': 0.3,
                                                   'groups_division_method': 'group_embedding'},
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05},
                'verbose': 0},
    
    'subgroups': {'dimensionality_reduction': 'pca', 
               'dimensionality_reduction_params': {'explained_variance_threshold': 0.3,
                                                   'groups_division_method': 'subgroups'},
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05},
                'verbose': 0},
}

data_generation_options = {
    'min_lag': 0,
    'max_lag': 5,
    'contemp_fraction': 0.25,
    'T': 1000, # Number of time points in the dataset
    'N_vars': 60, # Number of variables in the dataset
    'N_groups': 6, # Number of groups in the dataset
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

# TODO: Borrar cuando tenga terminado el benchmark completo
def increasing_N_vars_per_group(options, algorithms_parameters,
                      list_N_vars_per_group=None):
    if list_N_vars_per_group is None:
        list_N_vars_per_group = [2, 4, 6, 8, 10, 12]
    
    for N_vars_per_group in list_N_vars_per_group:
        if N_vars_per_group <= 6:
            algorithms_parameters['group-embedding']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.6
            algorithms_parameters['subgroups']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.6
        elif N_vars_per_group < 10:
            algorithms_parameters['group-embedding']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.5
            algorithms_parameters['subgroups']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.5
        elif N_vars_per_group < 12:
            algorithms_parameters['group-embedding']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.4
            algorithms_parameters['subgroups']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.4
        else:
            algorithms_parameters['group-embedding']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.3
            algorithms_parameters['subgroups']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.3
        
        options['N_vars_per_group'] = N_vars_per_group
        options['N_vars'] = options['N_groups'] * N_vars_per_group
        
        for algorithm_parameters in algorithms_parameters.values():
            algorithm_parameters['max_lag'] = options['max_lag']
        
        yield algorithms_parameters, options

benchmark_options = {
    'static_parameters': (static_parameters, {}),
    'changing_N_variables': (changing_N_variables,
                                    {'list_N_variables': [5]}),
    
    'changing_preselection_alpha': (changing_preselection_alpha,
                                    {'list_preselection_alpha': [0.01, 0.05, 0.1, 0.2]}),
    
    'changing_N_groups': (changing_N_groups,
                                    {'list_N_groups': [5, 10, 15, 20, 25, 30],
                                     'relation_vars_per_group': 3}),
    
    # 'increasing_N_vars_per_group': (changing_N_vars_per_group,
    #                                 {'list_N_vars_per_group': [2, 4, 6, 8, 10, 12, 14, 16]}),
    
    'increasing_N_vars_per_group': (increasing_N_vars_per_group,
                                    {'list_N_vars_per_group': [2, 4, 6, 8, 10, 12, 14, 16]}),
    
    
    'changing_alg_params': (changing_alg_params,
                                    {'alg_name': 'subgroups',
                                     'list_modifying_algorithms_params': [
                                        {'dimensionality_reduction_params': {'explained_variance_threshold': variance,
                                                                             'groups_division_method': 'subgroups'}}\
                                            for variance in list(np.linspace(0.05, 0.95, 19)) + [0.9999]]})
}

chosen_option = 'increasing_N_vars_per_group'



if __name__ == '__main__':
    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    
    benchmark = BenchmarkGroupCausalDiscovery()
    results_folder = 'results_increasing_N_vars_per_group_new'
    datasets_folder = f'{results_folder}/toy_data'
    execute_benchmark = True
    plot_graphs = False
    generate_toy_data = True
    n_executions = 100
    
    dataset_iteration_to_plot = -1
    plot_x_axis = 'N_vars_per_group'
    
    
    options_generator, options_kwargs = benchmark_options[chosen_option]
    parameters_iterator = options_generator(data_generation_options,
                                                algorithms_parameters,
                                                **options_kwargs)
    if execute_benchmark:
        results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                            parameters_iterator=parameters_iterator,
                                            datasets_folder=datasets_folder,
                                            generate_toy_data=generate_toy_data,
                                            results_folder=results_folder,
                                            n_executions=n_executions,
                                            verbose=1)
    elif generate_toy_data:
        # Delete previous toy data
        if os.path.exists(datasets_folder):
            for filename in os.listdir(datasets_folder):
                os.remove(f'{datasets_folder}/{filename}')
        else:
            os.makedirs(datasets_folder)

        for iteration, current_parameters in enumerate(parameters_iterator):
            current_algorithms_parameters, data_option = current_parameters
            causal_datasets = benchmark.generate_datasets(iteration, n_executions, datasets_folder, data_option)
    
    if plot_graphs:
        # benchmark.plot_ts_datasets(datasets_folder)
        
        benchmark.plot_moving_results(results_folder, x_axis=plot_x_axis)
        # Save results for whole graph scores
        benchmark.plot_particular_result(results_folder,
                                        dataset_iteration_to_plot=dataset_iteration_to_plot)
        # Save results for summary graph scores
        benchmark.plot_particular_result(results_folder, results_folder + '/summary',
                                        scores=[f'{score}_summary' for score in \
                                                        ['shd', 'f1', 'precision', 'recall']],
                                        dataset_iteration_to_plot=dataset_iteration_to_plot)
    