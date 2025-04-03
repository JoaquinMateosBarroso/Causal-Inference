import numpy as np
from sklearn.decomposition import PCA
from typing import Any

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscoveryBase
from group_causation.group_causal_discovery.micro_level import MicroLevelGroupCausalDiscovery

class HybridGroupCausalDiscovery(GroupCausalDiscoveryBase):
    '''
    Class that implements a group causal discovery algorithm which combines dimension reduction
    techniques with microlevel causal discovery.
    To do so, given a set of groups of variables, the algorithm will apply a dimension reduction
    technique to reduce the dimensionality of the problem. However, the dimension reduction won't
    necessarily give a one-dimensional time series, but it might give a set of time series (microgroups)
    with less variables than the original problem. Then, a microlevel causal discovery algorithm will
    be applied to the reduced time series, and the group causal graph will be extracted from the
    microgroups one.
    
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        groups : list[set[int]] list with the sets that will compound each group of variables.
                    We will suppose that the groups are known beforehand.
                    The index of a group will be considered as its position in groups list.
        dimensionality_reduction : str indicating the type of dimensionality reduction technique
                    that is applied to groups. options=['pca']. default='pca'
        dimensionality_reduction_params : dict with the parameters for the dimensionality reduction algorithm.
        node_causal_discovery_alg : str indicating the algorithm that will be used to discover the causal
                    relationships between the variables of each group. options=['pcmci', 'pc-stable', 'dynotears']
        node_causal_discovery_params : dict with the parameters for the node causal discovery algorithm.
    '''
    def __init__(self, data: np.ndarray,
                    groups: list[set[int]],
                    dimensionality_reduction: str = 'pca',
                    dimensionality_reduction_params: dict[str, Any] = None,
                    node_causal_discovery_alg: str = 'pcmci',
                    node_causal_discovery_params: dict[str, Any] = None,
                    verbose: int = 0,
                    **kwargs):
        super().__init__(data, groups, **kwargs)
        
        self.node_causal_discovery_alg = node_causal_discovery_alg
        self.node_causal_discovery_params = node_causal_discovery_params if node_causal_discovery_params is not None else {}
        self.extra_args = kwargs
        self.verbose = verbose
        
        
        if dimensionality_reduction == 'pca':
            self.micro_groups, self.micro_data = self._prepare_micro_groups_pca(**dimensionality_reduction_params)
        else:
            raise ValueError(f'Dimensionality reduction technique {dimensionality_reduction} not supported.')
        
        self.micro_level_causal_discovery = MicroLevelGroupCausalDiscovery(self.micro_data, self.micro_groups,
                                                                    node_causal_discovery_alg, node_causal_discovery_params)
    
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Extract the parents of each group of variables using the dimension reduction algorithm
        
        Returns:
            Dictionary with the parents of each group of variables.
        '''
        group_parents = self.micro_level_causal_discovery.extract_parents()
        
        # group_parents = self._convert_micro_to_group_parents(micro_parents)
        
        return group_parents
    
    
    def _prepare_micro_groups_pca(self, explained_variance_threshold: float = 0.9,
                                    groups_division_method: str='group_embedding') -> list[np.ndarray]:
        '''
        Execute the PCA dimensionality reduction algorithm to the groups of variables,
        in order to obtain a univariate time series for each group.
        
        Args:
            explained_variance_threshold : float indicating the minimum explained variance that the PCA
                        algorithm must achieve to stop the dimensionality reduction.
            groups_compresion_method : string indicating the method that will be used to compress the
                        groups of variables. options=['group_embedding', 'subgroups']
        
        Returns:
            micro_groups : list[ set[int] ] where keys are original groups, and values are the indexes of
                                    associated microvariables.
            micro_groups_data : np.ndarray where each column is the univariate time series of each group
                            of variables after the dimensionality reduction
        '''
        micro_groups = []
        micro_data = [] # List where each element is the ts data of a microgroup
        current_number_of_variables = 0
        for i, group in enumerate(self.groups):
            if groups_division_method == 'group_embedding':
                # Standarize data, so that the PCA algorithm works properly
                group_data = self.data[:, list(group)]
                group_data = (group_data - group_data.mean(axis=0)) / group_data.std(axis=0)
                # Extract the principal components of the group
                pca = PCA(n_components=explained_variance_threshold)
                group_data_pca = pca.fit_transform(group_data)
                
                # Append the microgroup variables indexes to the list    
                n_variables = group_data_pca.shape[1]
                current_number_of_variables = sum(arr.shape[1] for arr in micro_data)
                micro_groups.append( set(range(current_number_of_variables,
                                                current_number_of_variables + n_variables)) )
                
                # Append the microgroup data to the list
                micro_data.append(group_data_pca)
                
            elif groups_division_method == 'subgroups':
                def _prepare_subgroups(current_subgroup: set[int]) -> tuple[ list[set[int]], np.ndarray]:
                    '''
                    Recursive function that divides the group in 2 subgroups until the explained variance
                    of the first PC represents at least a "explained_variance_threshold" fraction of the total
                    '''
                    group_data = self.data[:, list(current_subgroup)]
                    pca = PCA(n_components=1)
                    group_data_pca = pca.fit_transform(group_data)
                    explained_variance = pca.explained_variance_ratio_[0]
                    
                    if explained_variance >= explained_variance_threshold:
                        # We have reached the desired explained variance; one single pc is enough
                        nonlocal current_number_of_variables
                        used_subgroup = [current_number_of_variables]
                        current_number_of_variables += 1
                        return used_subgroup, group_data_pca
                    else:
                        # Divide the half of the variables that have highest importance in PC1
                        ordered_nodes = np.argsort(pca.components_[0])
                        half = len(current_subgroup) // 2
                        first_half = ordered_nodes[:half]
                        second_half = ordered_nodes[half:]
                        first_subgroup, first_subgroup_data = _prepare_subgroups([current_subgroup[i] for i in first_half])
                        second_subgroup, second_subgroup_data = _prepare_subgroups([current_subgroup[i] for i in second_half])
                        return first_subgroup + second_subgroup, np.concatenate([first_subgroup_data, second_subgroup_data], axis=1)
                
                micro_group, group_data_pca = _prepare_subgroups(group)
                micro_groups.append( set(micro_group) )
                micro_data.append(group_data_pca)
            
            
            else:
                raise ValueError(f'Invalid groups division method: {groups_division_method}')
        
        micro_data = np.concatenate(micro_data, axis=1)
        
        if self.verbose > 0:
            print(f'Data dimensionality has been reduced to {micro_data.shape[1]} in order to perform microlevel causal discovery.')

        return micro_groups, micro_data
    
    
    def _convert_micro_to_group_parents(self, micro_parents: dict[int, list[int]]) -> dict[int, list[int]]:
        '''
        Convert the parents of each microgroup to the parents of each group of variables
        
        Args:
            micro_parents : dict[int, list[int]]. Dictionary with the parents of each microgroup.
        
        Returns:
            group_parents : dict[int, list[int]]. Dictionary with the parents of each group of variables.
        '''
        group_parents = {}
        for group_idx, group in enumerate(self.groups):
            group_parents[group_idx] = []
            for son_micro_idx in self.micro_groups[group_idx]:
                # A group is son of another group iff any microgroup of the son has a parent microgroup that is in the parent group
                for parent_micro_idx, lag in micro_parents[son_micro_idx]:
                    [parent_group_idx] = [idx for idx, micro_group in enumerate(self.micro_groups)
                                                if parent_micro_idx in micro_group]
                    # Add the parent group (with microgroup's lag) to the parents of the son group
                    group_parents[group_idx].append((parent_group_idx, lag))
            # Remove duplicates
            group_parents[group_idx] = list(set(group_parents[group_idx]))
        
        return group_parents
    