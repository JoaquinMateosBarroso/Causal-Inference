from matplotlib import pyplot as plt
import numpy as np

from group_causation.benchmark import BenchmarkGroupsExtraction
import shutil
import os

from group_causation.causal_groups_extraction.random_causal_groups_extraction import RandomCausalGroupsExtractor
from group_causation.functions_test_data import static_parameters
from group_causation.causal_groups_extraction import ExhaustiveCausalGroupsExtractor, GeneticCausalGroupsExtractor

algorithms = {
    'exhaustive': ExhaustiveCausalGroupsExtractor,
    'genetic': GeneticCausalGroupsExtractor,
    'random': RandomCausalGroupsExtractor, 
}
algorithms_parameters = {
    'exhaustive': {'scores': ['bic']}, # Exhaustive can get only one score
    
    'genetic': {'scores': ['bic'], 'scores_weights': [1.0]},
    
    'random': {},
}

data_generation_options = {}

benchmark_options = {
    'static_parameters': (static_parameters, {}),
}

chosen_option = 'static_parameters'



if __name__ == '__main__':
    plt.style.use('ggplot')
    
    benchmark = BenchmarkGroupsExtraction()
    datasets_folder = 'toy_data'
    results_folder = 'results_group_extraction'
    execute_benchmark = True
    plot_graphs = False
    generate_toy_data = False
    
    dataset_iteration_to_plot = -1
    plot_x_axis = ''
    
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
                                            n_executions=10,
                                            verbose=1)
    
    if plot_graphs:
        benchmark.plot_ts_datasets(datasets_folder)
        
        benchmark.plot_moving_results(results_folder, x_axis=plot_x_axis)
        # Save results for scores
        benchmark.plot_particular_result(results_folder,
                                        dataset_iteration_to_plot=dataset_iteration_to_plot)
    
    # Copy toy_data folder inside results folder, to have the datasets used in the benchmark
    destination_folder = os.path.join(results_folder, datasets_folder)
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(datasets_folder, destination_folder)
    

