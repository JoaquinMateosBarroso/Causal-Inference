# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib notebook
## use `%matplotlib notebook` for interactive figures
plt.style.use('ggplot')

from tigramite import data_processing as pp



'''
    EVALUATION METRICS
'''
def get_precision(ground_truth_parents: dict, predicted_parents: dict):
    # Precision = TP / (TP + FP)
    true_positives = 0
    for effect, causes in predicted_parents.items():
        true_positives += len([cause for cause in causes if cause in ground_truth_parents.get(effect, [])])
    
    predicted_positives = sum([len(causes) for causes in predicted_parents.values()])
    
    return true_positives / predicted_positives if predicted_positives != 0 else 0

def get_recall(ground_truth_parents: dict, predicted_parents: dict):
    # Recall = TP / (TP + FN)
    true_positives = 0
    for effect, causes in predicted_parents.items():
        true_positives += len([cause for cause in causes if cause in ground_truth_parents.get(effect, [])])
        
    ground_truth_positives = sum([len(causes) for causes in ground_truth_parents.values()])
    
    return true_positives / ground_truth_positives if ground_truth_positives != 0 else 0

def get_f1(ground_truth_parents: dict, predicted_parents: dict):
    precision = get_precision(ground_truth_parents, predicted_parents)
    recall = get_recall(ground_truth_parents, predicted_parents)
    
    return 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

def get_false_positive_ration(ground_truth_parents: dict, predicted_parents: dict):
    # FPR = FP / (FP + TN)
    false_positives = 0
    for effect, causes in predicted_parents.items():
        false_positives += len([cause for cause in causes if cause not in ground_truth_parents.get(effect, [])])
    
    true_negatives = 0
    for effect, causes in ground_truth_parents.items():
        true_negatives += len([cause for cause in causes if cause not in predicted_parents.get(effect, [])])
    
    return false_positives / (false_positives + true_negatives) if false_positives != 0 else 0

def get_shd(graph1: dict, graph2: dict):
    """Calculate the Structural Hamming Distance between two graphs."""
    def dict_to_adjacency_matrix(graph_dict, nodes):
        """Convert a graph dictionary to an adjacency matrix."""
        size = len(nodes)
        index = {node: i for i, node in enumerate(nodes)}
        adj_matrix = np.zeros((size, size), dtype=int)
        for child, parents in graph_dict.items():
            for parent in parents:
                adj_matrix[index[parent], index[child]] = 1
        return adj_matrix
    
    nodes = set(graph1.keys()).union(set(graph2.keys()))
    for parents in graph1.values():
        nodes.update(parents)
    for parents in graph2.values():
        nodes.update(parents)
    nodes = list(nodes)
    
    adj_matrix1 = dict_to_adjacency_matrix(graph1, nodes)
    adj_matrix2 = dict_to_adjacency_matrix(graph2, nodes)
    
    # Calculate the number of differing edges
    diff = np.abs(adj_matrix1 - adj_matrix2)
    shd = np.sum(diff)
    
    return shd

