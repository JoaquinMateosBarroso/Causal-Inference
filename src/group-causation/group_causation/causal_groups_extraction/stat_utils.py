from typing import Callable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from group_causation.group_causal_discovery import GroupCausalDiscovery, DimensionReductionGroupCausalDiscovery

from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

def get_pc1_explained_variance(data: np.ndarray) -> float:
    '''
    Get the explained variance of the first principal component of the data.
    '''
    if data.shape[1] == 0:
        return 1
    pca = PCA(n_components=1)
    pca.fit(data)
    
    return pca.explained_variance_ratio_[0]

def get_average_pc1_explained_variance(data: np.ndarray, groups: list[set[int]]) -> float:
    '''
    Get the average explained variance of the first principal component of the data
    for each group.
    '''
    explained_variances = [get_pc1_explained_variance(data[:, list(group)]) for group in groups]
    
    return np.mean(explained_variances)

def get_explainability_score(data: np.ndarray, groups: list[set[int]]) -> float:
    '''
    Get a score that represents how well the data can be explained by the groups.
    '''
    explained_variance = get_average_pc1_explained_variance(data, groups)
    inverse_n_groups = 1 - len(groups) / data.shape[1]
    
    geometric_mean = (explained_variance * inverse_n_groups) ** (1/2)
    
    return geometric_mean

def get_normalized_mutual_information(pred_groups: list[set[int]], gt_groups: list[set[int]]):
    '''
    Get the normalized mutual information between two sets of groups.
    '''
    n_nodes = len([*pred_groups])
    
    # Adapt group format to the labels one required by sklearn
    pred_labels = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j, group in enumerate(pred_groups):
            if i in group:
                pred_labels[i] = j
                break
    
    gt_labels = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j, group in enumerate(gt_groups):
            if i in group:
                gt_labels[i] = j
                break
    
    return normalized_mutual_info_score(gt_labels, pred_labels)

def get_bic(data: np.ndarray, groups: list[set[int]], discovery_model: GroupCausalDiscovery=None) -> float:
    '''
    Get the BIC score of the causal model inferred by the groups. A Dynamic Bayesian Network with linear Gaussian CPDs
    is assumed to calculate the BIC score.
    
    Parameters:
        data: The data used to infer the causal model.
        groups: The groups used to infer the causal model.
        discovery_model: The model used to infer the causal model. If None, a default model will be used.
    
    Returns:
        The BIC score of the inferred causal model.
    '''
    if discovery_model is None:
        discovery_model = DimensionReductionGroupCausalDiscovery(data, groups)
    
    # Extract parents and convert to pgmpy format
    parents = discovery_model.extract_parents()
    
    def compute_log_likelihood(data):
        log_likelihood = 0.0
        for t in range(data.shape[0]):  # Iterate over each time step (row)
            observation = {node: data[t, node] for node in range(data.shape[1])}  # Convert row to dict
            prob = inference.forward_inference([observation])
            log_likelihood += np.log(prob)
        return log_likelihood

    def count_parameters(dbn):
        k = 0
        for node in dbn.nodes:
            node_card = dbn.cardinality[node]
            parent_card = 1
            for parent in dbn.parents.get(node, []):
                parent_card *= dbn.cardinality[parent]
            # (node_card - 1) free parameters for each configuration of parents
            k += (node_card - 1) * parent_card
        return k

    def compute_bic(dbn, data):
        n = len(data)
        if n == 0:
            raise ValueError("Data must contain at least one sample.")
        
        logL = compute_log_likelihood(dbn, data)
        k = count_parameters(dbn)
        
        # Compute BIC: -2*log_likelihood + k*log(n)
        bic = -2 * logL + k * np.log(n)
        return bic

    model = LinearGaussianBayesianNetwork()
    
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    
    
    return bic

def get_scores_getter(data: np.ndarray, scores: list[str]) -> Callable:
    '''
    Generate a score getter function that receives a set of groups and returns a score to maximize.
    '''
    scores_getters = {
        'average_variance_explained': get_pc1_explained_variance,
        'explainability_score': get_explainability_score,
        'bic': get_bic,
    }
    
    return lambda groups: [scores_getters[score](data, groups) for score in scores]
