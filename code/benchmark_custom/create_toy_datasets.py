# %%
import json
import os
import random
from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tigramite.toymodels.structural_causal_processes import generate_structural_causal_process, structural_causal_process
from tigramite import plotting as tp
from tigramite.graphs import Graphs


def get_parents_dict(causal_process):
    parents_dict = dict()
    for key in causal_process.keys():
        if key not in parents_dict:
            parents_dict[key] = []
        for i in range(len(causal_process[key])):
            parents_dict[key].append(causal_process[key][i][0])
    return parents_dict

class CausalDataset:
    def __init__(self):
        self.time_series = None
        self.parents_dict = None
    
    dependency_funcs_dict = {
        'linear': lambda x: 0.5*x,
        'negative-exponential': lambda x: 1 - np.exp(-abs(x)),
        'sin': lambda x: np.sin(x),
        'cos': lambda x: np.cos(x),
        'step': lambda x: 1 if x > 0 else -1,
    }
    
    
    def generate_toy_data(self, name, T=100, N=10, crosslinks_density=0.75,
                      max_lag=3, dependency_funcs=['nonlinear'],
                      datasets_folder = None, **kw_generation_args) \
                            -> tuple[np.ndarray, dict[int, list[int]]]:
        """
        Generate a toy dataset with a causal process and time series data.
        Node-level links are modeled as a linear combination of the parents, in the following way:
        
        .. math:: `X_i^t = \sum_{j \in \text{parents}(i)} \beta_{ij} X_j^{t - \tau_{ij}} + \epsilon_i(t)`
        
        Where the scalar coefficients are takien from kw_generation_args 'dependency_coeffs' and 'auto_coeffs' parameters,
        and the noise :math:`\epsilon_i(t)` is taken from kw_generation_args 'noise_dists' and 'noise_sigmas' parameters.
        
        Parameters:
            name : Name of the dataset
            T : Number of time points
            N : Number of variables
            crosslinks_density : Percentage of links that are cross-links
            max_lag : Maximum lag of the causal process
            dependency_funcs : List of dependency functions (in {'linear', 'nonlinear'}, or a function f:R->R)
            dataset_folder : Name of the folder where datasets and parents will be saved. By default they are not saved.
            
        Returns:
            time_series : np.ndarray with shape (n_samples, n_variables)
            parents_dict: dictionary whose keys are each node, and values are the lists of parents, [... (i, -tau) ...].
        """
        # Convert dependency_funcs names to functions
        dependency_funcs = [self.dependency_funcs_dict[func] if func in self.dependency_funcs_dict else func
                                for func in dependency_funcs ]
        
        L = int(N * crosslinks_density / (1 - crosslinks_density)) # Forcing crosslinks_density = L / (N + L)
        
        # Generate random causal process
        causal_process, noise = generate_structural_causal_process(N=N,
                                                            L=L,
                                                            max_lag=max_lag,
                                                            dependency_funcs=dependency_funcs,
                                                            **kw_generation_args)

        self.parents_dict = get_parents_dict(causal_process)

        # Generate time series data from the causal process
        self.time_series, _ = structural_causal_process(causal_process, T=T, noises=noise)
        
        if datasets_folder is not None:
            # If the folder does not exist, create it
            if not os.path.exists(datasets_folder):
                os.makedirs(datasets_folder)
            # Save the dataset
            self.save(name, datasets_folder)
                
        return self
    
    def _save(self, name, dataset_folder):
        # Save the time series data to a csv file
        df = pd.DataFrame(self.time_series)
        df.to_csv(f'{dataset_folder}/{name}_data.csv', index=False, header=True)
        # Save parents to a txt file
        with open(f'{dataset_folder}/{name}_parents.txt', 'w') as f:
            parents_representation = repr(self.parents_dict)
            f.write(parents_representation)
    
    def generate_group_toy_data(self, name, T=100, N_vars=20, N_groups=3,
                                inner_group_crosslinks_density=0.5, outer_group_crosslinks_density=0.5,
                                n_node_links_per_group_link=2,
                                max_lag=3, dependency_funcs=['nonlinear'],
                                datasets_folder = None, **kw_generation_args) \
                            -> tuple[np.ndarray, dict[int, list[int]], dict[int, list[int]], list[list[int]]]:
        '''
        Generate a toy dataset with a group based causal process and time series data.
        There will first be generated N_groups different groups of variables, each with between 2 and 
        N_vars - 2*N_groups variables. Then, there will be generated links between the groups,
        with a density of outer_group_crosslinks_density. Finally, there will be generated links between
        nodes of different groups, with a density of inner_group_crosslinks_density.
        Node-level links are modeled as a linear combination of the parents, in the following way:
        
        .. math:: `X_i^t = \sum_{j \in \text{parents}(i)} \beta_{ij} X_j^{t - \tau_{ij}} + \epsilon_i(t)`
        
        Where the scalar coefficients are takien from kw_generation_args 'dependency_coeffs' and 'auto_coeffs' parameters,
        and the noise :math:`\epsilon_i(t)` is taken from kw_generation_args 'noise_dists' and 'noise_sigmas' parameters.
        
        Parameters:
            name : Name of the dataset
            T : Number of time points
            N : Number of variables
            N_groups : Number of groups (this number must be greater than N/2, to allow groups creation)
            inner_group_crosslinks_density : Percentage of inner-group links that are cross-links
            n_node_links_per_group_link : Number of links between nodes of different groups that are linked
            max_lag : Maximum lag of the causal process
            dependency_funcs : List of dependency functions (in ['linear', 'nonlinear'], or a function f:R->R)
            dataset_folder : Name of the folder where datasets and parents will be saved. By default they are not saved.
            
        Returns:
            time_series : np.ndarray with shape (n_samples, n_variables).
            node_parents_dict : dictionary whose keys are each node, and values are the lists of parents, [... (i, -tau) ...].
            group_parents_dict : dictionary whose keys are each group, and values are the lists of parent groups, [... (i, -tau) ...].
            groups : List of lists, where each list is a group of variables
        '''
        # Convert dependency_funcs names to functions
        dependency_funcs = [self.dependency_funcs_dict[func] if func in self.dependency_funcs_dict else func\
                                for func in dependency_funcs ]
        
        self.groups = self._generate_groups(N_vars, N_groups)
        
        # Dictionary where keys will be the global index of the nodes, and values the causal processes
        groups_causal_processes = dict()
        
        # Generate inner causal processes
        for index, group in enumerate(self.groups):
            # Forcing crosslinks_density = L / (N + L)
            L = int(len(group) * inner_group_crosslinks_density / (1 - inner_group_crosslinks_density))
            causal_process, _ = generate_structural_causal_process(N=len(group), L=L, max_lag=max_lag,
                                                                    dependency_funcs=dependency_funcs,
                                                                    **kw_generation_args)
            
            # Change the keys of the causal process to the global index
            groups_causal_processes[index] = self._change_keys(causal_process, group)
        
        # Generate outer causal processes
        L = int(N_groups * outer_group_crosslinks_density / (1 - outer_group_crosslinks_density))
        outer_causal_process, _ = generate_structural_causal_process(N=N_groups, L=L,
                                                                    max_lag=max_lag, dependency_funcs=dependency_funcs,
                                                                    **kw_generation_args)
        
        global_causal_process = self._join_processes( outer_causal_process, groups_causal_processes,
                                                     n_node_links_per_group_link)
        
        # Generate noise
        _, noise = generate_structural_causal_process(N=N_vars, **kw_generation_args)
        
        self.parents_dict = get_parents_dict(global_causal_process)
        self.group_parents_dict = self.extract_group_parents(self.parents_dict)
        
        # Generate time series data from the causal process
        self.time_series, _ = structural_causal_process(global_causal_process, T=T, noises=noise)
        
        if datasets_folder is not None:
            # If the folder does not exist, create it
            if not os.path.exists(datasets_folder):
                os.makedirs(datasets_folder)
            # Save the dataset
            self._save_groups(name, datasets_folder)
        
        return self.time_series, self.parents_dict, self.group_parents_dict, self.groups
    
    def extract_group_parents(self, node_parents_dict):
        '''
        Given a dictionary with the parents of each node, return a dictionary with the parents of each group.
        
        Parameters:
            node_parents_dict : dictionary whose keys are each node, and values are the lists of parents, [... (i_node, -tau) ...].
            
        Returns:
            group_parents_dict: dictionary whose keys are each group, and values are the lists of parent groups, [... (i_group, -tau) ...].
        '''
        group_parents_dict = {i: [] for i in range(len(self.groups))}
        
        # Iterate over the nodes and their parents
        for son_node, parents in node_parents_dict.items():
            [son_group] = [i for i, group in enumerate(self.groups) if son_node in group]
            for parent, lag in parents:
                [parent_group] = [i for i, group in enumerate(self.groups) if parent in group]
                # Add the parent group to the son group
                group_parents_dict[son_group].append((parent_group, lag))
        
        return group_parents_dict
        
    def _generate_groups(self, N_vars, N_groups) -> list[list[int]]:
        '''
        Generate N_groups groups of variables, with at least 2 nodes each.
        
        Parameters:
            N_vars : Number of variables
            N_groups : Number of groups
        
        Returns:
            groups : List of lists, where each list is a group of variables
        '''
        if N_groups > N_vars/2:
            raise ValueError('The number of groups must be less than N_vars / 2')
        
        # Generate N_groups groups with 2 nodes each
        nodes = list( range(N_vars) )
        groups = [[nodes.pop(), nodes.pop()] for _ in range(N_groups)]
        
        # Distribute the remaining nodes randomly
        while len(nodes) != 0:
            group = groups[np.random.randint(0, N_groups)]
            group.append(nodes.pop())

        return groups
    
    def _change_keys(self, causal_process, group) -> dict[int, list[tuple[tuple, int, Callable]]]:
        '''
        Change the keys of the causal process to the global index, through a bijective mapping
        
        
        Parameters:
            causal_process : dictionary whose keys are the local index of the nodes, and values are the lists of tuples
                                [ ((parent, lag), coeff, func), ... ]
            group : list of the global index of the nodes
            
        Returns:
            new_causal_process : dictionary whose keys are the global index of the nodes, and values are the lists of tuples
                                [ ((parent, lag), coeff, func), ... ]
        '''
        # Dictionary where keys will be original key, and values their associated node
        self.association_relation = dict( zip(causal_process.keys(), group) )
        new_causal_process = dict()
        
        for original_key, new_index in self.association_relation.items():
            original_parents = causal_process[original_key]
            
            # We need to change the parents to the global index
            change_parent_index = lambda parent, lag: (self.association_relation[parent], lag)
            new_parents = [(change_parent_index(*node), coeff, func) for node, coeff, func in original_parents]
 
            new_causal_process[new_index] = new_parents
        
        return new_causal_process
            
    def _join_processes(self, outer_causal_process: dict[int, tuple[tuple, int, Callable]],
                            groups_causal_processes: dict[ int, dict[int, tuple[tuple, int, Callable]] ],
                            n_node_links_per_group_link) -> dict[int, list[tuple[tuple, int, Callable]]]:
        '''
        Join several causal processes into one. The outer_causal_process will be converted
        to the global index by assigning n_node_links_per_group_link links between nodes
        of the different groups.
        
        Parameters:
            outer_causal_process : dictionary whose keys are the index of the groups, and values are the lists of tuples
                                [ ((parent_group, lag), coeff, func), ... ]
            groups_causal_processes : dictionaries whose keys are the groups index, and whose items are dictionaries
                                with the same structure as outer_causal_process, i.e., keys are global index of the nodes,
                                and values are the lists of tuples [ ((parent, lag), coeff, func), ... ]
            n_node_links_per_group_link : Number of links between nodes of different groups that are linked
        
        Returns:
            global_causal_process : dictionary whose keys are the global index of the nodes, and values are the lists of tuples
                                [ ((parent, lag), coeff, func), ... ]
        '''
        global_causal_process = {i: [] for group in self.groups for i in group}
        
        # Assign the outer links
        for son_group, group_parents in outer_causal_process.items():
            for (parent_group_index, lag), coeff, func in group_parents:
                for _ in range(n_node_links_per_group_link):
                    # Choose a random node from each group
                    origin_node = random.choice(self.groups[son_group])
                    # Choose a random node from the parent group
                    parent_node = random.choice(self.groups[parent_group_index])
                    # Assign the link
                    global_causal_process[origin_node].append( ((parent_node, lag), coeff, func) )
        
        # Assign the inner links
        for son_group, causal_process in groups_causal_processes.items():
            for son_node, parents in causal_process.items():
                global_causal_process[son_node].extend(parents)
        
        return global_causal_process
    
    def _save_groups(self, name, dataset_folder):
        self._save(name, dataset_folder)
        # Save the groups to a txt file
        with open(f'{dataset_folder}/{name}_groups.txt', 'w') as f:
            groups_representation = repr(self.groups)
            f.write(groups_representation)
        # Save the groups parents to a txt file
        with open(f'{dataset_folder}/{name}_group_parents.txt', 'w') as f:
            group_parents_representation = repr(self.group_parents_dict)
            f.write(group_parents_representation)
    

def _plot_ts_graph(parents_dict):
    '''
    Function to plot the graph structure of the time series
    '''
    graph = Graphs.get_graph_from_dict(parents_dict)
    
    tp.plot_time_series_graph(
        graph=graph,
        var_names=list(parents_dict.keys()),
        link_colorbar_label='cross-MCI (edges)',
    )


if __name__ == '__main__':
    dataset = CausalDataset()
    # dataset.generate_toy_data('1')
    
    ts, node_parents_dict, group_parents_dict, groups = dataset.generate_group_toy_data('1',
                                                                                        N_vars=6,
                                                                                        N_groups=3,)
    
    _plot_ts_graph(node_parents_dict)
    _plot_ts_graph(group_parents_dict)
    plt.show()

