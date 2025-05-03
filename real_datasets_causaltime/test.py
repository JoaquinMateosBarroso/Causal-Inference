# %% [markdown]
# ## Transforming data in csv and parents format

# %%
import os
import numpy as np
import sys
import random
random.seed(0)
np.random.seed(0)

import pandas as pd
sys.path.append('../')

data_names = ['medical', 'traffic', 'pm25']

def matrix_graph_to_parents(matrix_graph: np.ndarray) -> dict[int, list[int]]:
    """
    Convert a matrix graph to a dictionary representation of parents.

    Args:
        matrix_graph (np.ndarray): The adjacency matrix representing the graph.

    Returns:
        dict[int, list[int]]: A dictionary where keys are node indices and values are lists of parent node indices.
    """
    parents = {}
    for i in range(matrix_graph.shape[0]):
        parents[i] = np.where(matrix_graph[i] == 1)[0].tolist()
    
    parents = {k: [(v, -1) for v in vs] for k, vs in parents.items()}
    return parents

SAMPLE_NUM = 480 # All the datasets have 480 samples
for data_name in data_names:
    data = np.load('./' + data_name + '/gen_data.npy')
    data = data[:, 20:, :data.shape[2] // 2]  # Forget the residuals and for some reason first 20 values are random
    matrix_graph = np.load('./' + data_name + '/graph.npy')

    print(f"Data Name: {data_name}")
    print(f'Shape of Graph H: {matrix_graph.shape}')
    print(f'Shape of Time-series Data: {data.shape} (Sample_num, Time_step, Node_num)')

    os.makedirs(f'./data_{data_name}', exist_ok=True)
    for i in range(data.shape[0]): # Iterate over samples
        with open(f'./data_{data_name}/{i}_node_parents.txt', 'w') as f:
            f.write(str(matrix_graph_to_parents(matrix_graph))) # Same matrix graph for all samples
        
        current_data = data[i, :, :]  # Select the i-th sample
        
        pd.DataFrame(current_data).to_csv(f'./data_{data_name}/{i}_data.csv', index=False, header=False)


# %% [markdown]
# ## Find the groups we are going to use

# %%
import shutil
import os

from group_causation.groups_extraction import GeneticCausalGroupsExtractor


datasets_groups = {k: None for k in data_names}
def extract_and_save_groups(data_name):
    data = pd.read_csv(f'./data_{data_name}/0_data.csv', header=None).values
    if data.shape[1] > 30: # Since there are many variables, consider the harmonic variance
        group_extractor = GeneticCausalGroupsExtractor(data, 
                                                    scores=['harmonic_variance_explained', 'explainability_score'], 
                                                    scores_weights=[0.1, 1.0])
    else:
        group_extractor = GeneticCausalGroupsExtractor(data, 
                                                    scores=['explainability_score'], 
                                                    scores_weights=[1.0])
        
    groups = group_extractor.extract_groups()
    datasets_groups[data_name] = groups

    print(data_name, 'dataset obtained the groups:', groups)
    
    # with open(f'./data_{data_name}/0_groups.txt', 'w') as f:
    #     f.write(str(groups))
    
    for i in range(1, SAMPLE_NUM):
        shutil.copyfile(f'./data_{data_name}/0_groups.txt', f'./data_{data_name}/{i}_groups.txt')

for data_name in data_names:
    extract_and_save_groups(data_name)

# %% [markdown]
# ## Convert node-level parents to group-level parents

# %%
def find_index_with_element(groups, x):
    for i, group in enumerate(groups):
        if x in group: return i
    return None

for data_name in data_names:
    with open(f'./data_{data_name}/0_groups.txt', 'r') as f:
        datasets_groups[data_name] = eval(f.read())

for data_name, groups in datasets_groups.items():
    with open(f'./data_{data_name}/0_node_parents.txt', 'r') as f:
        node_parents = eval(f.read())
    
    group_parents = {}
    for son_group_idx, son_group in enumerate(groups):
        group_parents[son_group_idx] = []
        for son_node in son_group:
            for parent_node, lag in node_parents[son_node]:
                parent_group_idx = find_index_with_element(groups, parent_node)
                if (parent_group_idx, lag) not in group_parents[son_group_idx]:
                    group_parents[son_group_idx].append((parent_group_idx, -1))
                
    with open(f'./data_{data_name}/0_parents.txt', 'w') as f:
        f.write(str(group_parents))
    for i in range(1, SAMPLE_NUM):
        shutil.copyfile(f'./data_{data_name}/0_parents.txt', f'./data_{data_name}/{i}_parents.txt')

# %%
find_index_with_element(groups, 11)

# %% [markdown]
# ## Perform the benchmark for each of the datasets

# %%
from matplotlib import pyplot as plt

from group_causation.benchmark import BenchmarkGroupCausalDiscovery

from group_causation.utils import static_parameters
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
                            'node_causal_discovery_params': {'min_lag': 1, 'max_lag': 3, 'pc_alpha': 0.05}},
    
    'pca+dynotears': {'dimensionality_reduction': 'pca', 'node_causal_discovery_alg': 'dynotears',
                            'node_causal_discovery_params': {'min_lag': 1, 'max_lag': 3, 'lambda_w': 0.001, 'lambda_a': 0.001}},
    
    'micro-level': {'node_causal_discovery_alg': 'pcmci',
                            'node_causal_discovery_params': {'min_lag': 1, 'max_lag': 3, 'pc_alpha': 0.05}},
    
    'group-embedding': {'dimensionality_reduction': 'pca', 
               'dimensionality_reduction_params': {'explained_variance_threshold': 0.7,
                                                   'groups_division_method': 'group_embedding'},
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {'min_lag': 1, 'max_lag': 3, 'pc_alpha': 0.05},
                'verbose': 0},
    
    'subgroups': {'dimensionality_reduction': 'pca', 
               'dimensionality_reduction_params': {'explained_variance_threshold': 0.7,
                                                   'groups_division_method': 'subgroups'},
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {'min_lag': 1, 'max_lag': 3, 'pc_alpha': 0.05},
                'verbose': 0},
}

data_generation_options = {}

benchmark_options = {
    'static_parameters': (static_parameters, {}),
}

chosen_option = 'static_parameters'


def execute_benchmark(data_name):    
    benchmark = BenchmarkGroupCausalDiscovery()
    results_folder = f'results_{data_name}'
    datasets_folder = f'data_{data_name}'
    
    options_generator, options_kwargs = benchmark_options[chosen_option]
    parameters_iterator = options_generator(data_generation_options,
                                                algorithms_parameters,
                                                **options_kwargs)
    results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                        parameters_iterator=parameters_iterator,
                                        datasets_folder=datasets_folder,
                                        generate_toy_data=False,
                                        results_folder=results_folder,
                                        n_executions=5,
                                        verbose=1)
    
    return results, benchmark

# %%
from group_causation.benchmark import BenchmarkGroupCausalDiscovery

plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

for data_name in (data_names:=['pm25', 'medical', 'traffic']):
    print('Executing benchmark of', data_name)
    results, benchmark = execute_benchmark(data_name)
    
# Plot graphs
for data_name in (data_names:=['pm25', 'medical', 'traffic']):
    results_folder = f'results_{data_name}'
    benchmark = BenchmarkGroupCausalDiscovery()
    # benchmark.plot_particular_result(results_folder, results_folder + '/summary',
    #                                 scores=[f'{score}_summary' for score in \
    #                                                 ['shd', 'f1', 'precision', 'recall']],
    #                                 dataset_iteration_to_plot=0)


