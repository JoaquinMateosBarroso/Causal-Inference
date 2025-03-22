import numpy as np
from sklearn.decomposition import PCA


def get_pc1_explained_variance(data: np.ndarray) -> float:
    '''
    Get the explained variance of the first principal component of the data.
    '''
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

def get_variance_explainability_score(data: np.ndarray, groups: list[set[int]]) -> float:
    '''
    Get a score that represents how well the data can be explained by the groups.
    '''
    explained_variance = get_average_pc1_explained_variance(data, groups)
    inverse_n_groups = 1 / len(groups)
    
    geometric_mean = (explained_variance * inverse_n_groups) ** (1/2)
    
    return geometric_mean