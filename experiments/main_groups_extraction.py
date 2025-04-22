from matplotlib import pyplot as plt
import numpy as np

from group_causation.benchmark import BenchmarkGroupsExtraction
import shutil
import os

from group_causation.causal_groups_extraction.random_causal_groups_extraction import RandomCausalGroupsExtractor
from group_causation.utils import changing_N_groups, static_parameters, changing_N_variables
from group_causation.causal_groups_extraction import ExhaustiveCausalGroupsExtractor, GeneticCausalGroupsExtractor

algorithms = {
    # 'exhaustive': ExhaustiveCausalGroupsExtractor,
    'genetic2': GeneticCausalGroupsExtractor,
    'random2': RandomCausalGroupsExtractor, 
}
algorithms_parameters = {
    'exhaustive': {'scores': ['explainability_score']}, # Exhaustive can get only one score
    
    'genetic2': {'scores': ['explainability_score'], 'scores_weights': [1.0]},
    
    'random2': {'scores': ['explainability_score'],},
}

data_generation_options = {
    'min_lag': 0,
    'max_lag': 5,
    'contemp_fraction': 0.25,
    'T': 1000, # Number of time points in the dataset
    'N_vars': 60, # Number of variables in the dataset
    'N_groups': 1, # Number of groups in the dataset
    'inner_group_crosslinks_density': 0.5,
    'outer_group_crosslinks_density': 0.5,
    'n_node_links_per_group_link': 2,
    # These parameters are used in generate_structural_causal_process:
    'dependency_coeffs': [-0.3, 0.3], # default: [-0.5, 0.5]
    'auto_coeffs': [0.4], # default: [0.5, 0.7]
    'noise_dists': ['gaussian'], # deafult: ['gaussian']
    'noise_sigmas': [0.2], # default: [0.5, 2]
    
    'dependency_funcs': ['linear']#, 'negative-exponential', 'sin', 'cos', 'step'], # Options: 'linear', 'negative-exponential', 'sin', 'cos', 'step'}
}

benchmark_options = {
    'static_parameters': (static_parameters, {}),
    
    'changing_N_groups': (changing_N_groups,
                                    {'list_N_groups': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     'relation_vars_per_group': 2}),
    
    'changing_N_variables': (changing_N_variables,
                                    {'list_N_variables': [12, 13, 14, 15, 16],}),
}

chosen_option = 'changing_N_variables'



if __name__ == '__main__':
    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    
    benchmark = BenchmarkGroupsExtraction()
    results_folder = 'results_group_extraction'
    datasets_folder = f'{results_folder}/toy_data'
    execute_benchmark = True
    plot_graphs = True
    generate_toy_data = True
    n_executions = 10
    
    dataset_iteration_to_plot = -1
    plot_x_axis = 'N_vars'
    
    if execute_benchmark:
        options_generator, options_kwargs = benchmark_options[chosen_option]
        parameters_iterator = options_generator(data_generation_options,
                                                    algorithms_parameters,
                                                    **options_kwargs)
        
        results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                            parameters_iterator=parameters_iterator,
                                            datasets_folder=datasets_folder,
                                            generate_toy_data=generate_toy_data,
                                            results_folder=results_folder,
                                            n_executions=n_executions,
                                            verbose=1)
    
    if plot_graphs:
        # benchmark.plot_ts_datasets(datasets_folder)
        
        scores = ['explainability_score', 'time', 'memory', 'average_explained_variance', 'n_groups']
        
        benchmark.plot_moving_results(results_folder, x_axis=plot_x_axis, scores=scores)
        # Save results for scores
        benchmark.plot_particular_result(results_folder,
                                        dataset_iteration_to_plot=dataset_iteration_to_plot,
                                        scores=scores)
    
