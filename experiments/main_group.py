from matplotlib import pyplot as plt
import numpy as np

from group_causation.benchmark import BenchmarkGroupCausalDiscovery
import shutil
import os

from group_causation.functions_test_data import changing_N_groups, changing_N_variables, changing_N_vars_per_group, changing_alg_params, changing_preselection_alpha, static_parameters
from group_causation.group_causal_discovery import DimensionReductionGroupCausalDiscovery
from group_causation.group_causal_discovery import MicroLevelGroupCausalDiscovery
from group_causation.group_causal_discovery import HybridGroupCausalDiscovery

algorithms = {
    'particular_hybrid_principal_components': HybridGroupCausalDiscovery,
    'particular_hybrid_subgroups': HybridGroupCausalDiscovery,
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
    
    'particular_hybrid_principal_components': {'dimensionality_reduction': 'pca', 
               'dimensionality_reduction_params': {'explained_variance_threshold': 0.5,
                                                   'groups_division_method': 'principal_components'},
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05},
                'verbose': 1},
    
    'particular_hybrid_subgroups': {'dimensionality_reduction': 'pca', 
               'dimensionality_reduction_params': {'explained_variance_threshold': 0.5,
                                                   'groups_division_method': 'subgroups'},
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05},
                'verbose': 1},
}

data_generation_options = {
    'min_lag': 0,
    'max_lag': 5,
    'contemp_fraction': 0.25,
    'T': 1000, # Number of time points in the dataset
    'N_vars': 8, # Number of variables in the dataset
    'N_groups': 3, # Number of groups in the dataset
    'inner_group_crosslinks_density': 0.75,
    'outer_group_crosslinks_density': 0.2,
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
                                    {'list_N_vars_per_group': [2, 4, 6, 8, 10]}),
    
    'changing_alg_params': (changing_alg_params,
                                    {'alg_name': 'hybrid',
                                     'list_modifying_algorithms_params': [
                                        {'dimensionality_reduction_params': {'explained_variance_threshold': variance,
                                                                             'groups_division_method': 'subgroups'}}\
                                            for variance in list(np.linspace(0.05, 0.95, 19)) + [0.9999]]})
}

chosen_option = 'static_parameters'



if __name__ == '__main__':
    plt.style.use('ggplot')
    
    benchmark = BenchmarkGroupCausalDiscovery()
    datasets_folder = 'results/toy_data'
    results_folder = 'results'
    execute_benchmark = True
    plot_graphs = True
    generate_toy_data = False
    n_executions = 5
    
    dataset_iteration_to_plot = -1
    plot_x_axis = 'f1'
    
    
    
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
        benchmark.plot_ts_datasets(datasets_folder)
        
        benchmark.plot_moving_results(results_folder, x_axis=plot_x_axis)
        # Save results for whole graph scores
        benchmark.plot_particular_result(results_folder,
                                        dataset_iteration_to_plot=dataset_iteration_to_plot)
        # Save results for summary graph scores
        benchmark.plot_particular_result(results_folder, results_folder + '/summary',
                                        scores=[f'{score}_summary' for score in \
                                                        ['shd', 'f1', 'precision', 'recall']],
                                        dataset_iteration_to_plot=dataset_iteration_to_plot)
    
    # Copy toy_data folder inside results folder, to have the datasets used in the benchmark
    destination_folder = os.path.join(results_folder, datasets_folder)
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(datasets_folder, destination_folder)
    

