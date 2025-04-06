import numpy as np
from typing import Any

from group_causation.causal_discovery.causal_discovery_base import CausalDiscoveryBase
from group_causation.causal_discovery.causal_discovery_causalnex import DynotearsWrapper
from group_causation.causal_discovery.causal_discovery_tigramite import PCMCIWrapper, PCStableWrapper
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscoveryBase


class MicroLevelGroupCausalDiscovery(GroupCausalDiscoveryBase):
    '''
    Class that implements the dimension reduction algorithm for causal discovery on groups of variables.
    A causal discovery algorithm is applied to discover the causal relationships between all of the node-level
    variables, and then the whole graph is reduced to the group-level graph.
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        groups : list[set[int]] list with the sets that will compound each group of variables.
                    We will suppose that the groups are known beforehand.
                    The index of a group will be considered as its position in groups list.
        node_causal_discovery_alg : str indicating the algorithm that will be used to discover the causal
                    relationships between the variables of each group. options=['pcmci', 'pc-stable', 'dynotears']
        node_causal_discovery_params : dict with the parameters for the node causal discovery algorithm.
    '''
    def __init__(self, data: np.ndarray,
                    groups: list[set[int]],
                    node_causal_discovery_alg: str = 'pcmci',
                    node_causal_discovery_params: dict[str, Any] = None,
                    **kwargs):
        super().__init__(data, groups, **kwargs)
        
        self.node_causal_discovery_alg = node_causal_discovery_alg
        self.node_causal_discovery_params = node_causal_discovery_params if node_causal_discovery_params is not None else {}
        
    
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Extract the parents of each group of variables using the micro-level causal discovery algorithm
        to extract the micro-level causal DAG and convert it in a group-level graph.
        
        Returns
            Dictionary with the parents of each group of variables.
        '''
        self.causal_discovery_alg = self._getCausalDiscoveryAlgorithm()
        
        node_parents = self.causal_discovery_alg.extract_parents()

        group_parents = self._convert_node_to_group_parents(node_parents)
        
        return group_parents

    def _convert_node_to_group_parents(self, node_parents: dict[int, list[int]]) -> dict[int, list[int]]:
        '''
        Convert the parents of each node to the parents of each group of variables
        
        Args:
            node_parents : dict[int, list[int]]. Dictionary with the parents of each node.
        
        Returns:
            group_parents : dict[int, list[int]]. Dictionary with the parents of each group of variables.
        '''
        group_parents = {}
        for group_idx, group in enumerate(self.groups):
            group_parents[group_idx] = []
            for son_node_idx in group:
                # A group is son of another group iff any node has a parent node that is in the parent group
                for parent_node_idx in node_parents[son_node_idx]:
                    [parent_group_idx] = [idx for idx, group in enumerate(self.groups) if parent_node_idx[0] in group]
                    # Add the parent group (with node's lag) to the parents of the son group
                    group_parents[group_idx].append((parent_group_idx, parent_node_idx[1]))
            # Remove duplicates
            group_parents[group_idx] = list(set(group_parents[group_idx]))
        
        return group_parents
    
    def _getCausalDiscoveryAlgorithm(self) -> CausalDiscoveryBase:
        '''
        Get the causal discovery algorithm that will be used to discover the causal relationships
        between the variables of each group.
        
        Returns:
            causal_discovery_alg : function that will be used to discover the causal relationships
        '''
        if self.node_causal_discovery_alg == 'pcmci':
            return PCMCIWrapper(data=self.data, **self.node_causal_discovery_params)
        elif self.node_causal_discovery_alg == 'pc-stable':
            return PCStableWrapper(data=self.data, **self.node_causal_discovery_params)
        elif self.node_causal_discovery_alg == 'dynotears':
            return DynotearsWrapper(data=self.data, **self.node_causal_discovery_params)
        else:
            raise ValueError(f'Invalid node causal discovery algorithm: {self.node_causal_discovery_alg}')