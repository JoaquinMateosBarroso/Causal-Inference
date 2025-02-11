from typing import Any, Iterator, Union

from matplotlib import pyplot as plt
import numpy as np

from custom_benchmark.benchmark_causal_discovery import BenchmarkCausalDiscovery
from custom_benchmark.causal_discovery_tigramite import PCMCIWrapper


algorithms = {
    'pcmci': PCMCIWrapper,
    # 'lpcmci': LPCMCIWrapper,
}
def generate_parameters_iterator() -> Iterator[Union[dict[str, Any], dict[str, Any]]]:
    '''
    Function to generate the parameters for the algorithms and the data generation.
    
    Returns:
        parameters_iterator: function[dict[str, Any], dict[str, Any]]. A function that returns the parameters for the algorithms and the data generation.
    '''
    algorithms_parameters = {
        'pcmci': {'pc_alpha': 0.01, 'tau_max': 3},
        'lpcmci': {'pc_alpha': 0.01, 'tau_max': 3},
    }
    options = {
        'max_lag': 3,
        'dependency_funcs': [ lambda x: x, # linear
                                lambda x: x + np.exp(-(x**2)), # asymptotically linear
                            ],
        'L': 10, # Number of cross-links in the dataset
        'T': 200, # N time points in the dataset
        'N': 20, # Number of variables in the dataset
        # These parameters are used in generate_structural_causal_process:
        'dependency_coeffs': [-0.5, 0.5], # default: [-0.5, 0.5]
        'auto_coeffs': [0.5], # default: [0.5, 0.7]
        'noise_dists': ['gaussian'], # deafult: ['gaussian']
        'noise_sigmas': [1], # default: [0.5, 2]
    }
    for max_lag in [2, 4, 6]:
        options['max_lag'] = max_lag
        algorithms_parameters['pcmci']['tau_max'] = max_lag
        algorithms_parameters['lpcmci']['tau_max'] = max_lag
        
        yield algorithms_parameters, options



if __name__ == '__main__':
    benchmark = BenchmarkCausalDiscovery()
    results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                         parameters_iterator=generate_parameters_iterator(),
                                         datasets_folder='toy_data',
                                         results_folder='results',
                                         n_executions=3,
                                         scores=['f1', 'precision', 'recall', 'time', 'memory'],
                                         verbose=1)
    
    plt.style.use('ggplot')
    benchmark.plot_ts_datasets('toy_data')
    benchmark.plot_results('results')