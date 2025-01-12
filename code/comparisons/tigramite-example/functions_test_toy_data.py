# Imports
import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib notebook
## use `%matplotlib notebook` for interactive figures
plt.style.use('ggplot')

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from tigramite.independence_tests.parcorr import ParCorr
import time


# %%
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




# %%
def test_toy_data(name, parents_extractor, verbose=0):
    # Load causal process dictionary
    with open(f'toy_data/causal_process{name}.txt') as f:
        line = f.readline()
        causal_process = eval(line)
    # Load example data
    df = pd.read_csv(f'toy_data/data{name}.csv', header=None)
    dataframe = pp.DataFrame(df.values, var_names=df.columns)
    dataframe.values[0].shape
    
    
    parents, time = parents_extractor(dataframe)
    
    precision = get_precision(parents, causal_process)
    recall = get_recall(parents, causal_process)
    f1 = get_f1(parents, causal_process)
    if verbose>0:
        # Compare results
        print(f'Predicted parents: {parents}')
        print(f'Causal process: {causal_process}')
        
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'time': time}
    

def extract_parents_pcmci(dataframe):
    '''
    Returns the parents dict and the time that took to run the algorithm
    '''
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0
    )

    tau_max = 3
    pc_alpha = None # Optimize in a list
    
    start_time = time.time()
    results = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    end_time = time.time()
    execution_time = end_time - start_time

    parents = pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
    return parents, execution_time

def extract_parents_pcmciplus(dataframe):
    '''
    Returns the parents dict and the time that took to run the algorithm
    '''
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0
    )

    tau_max = 3
    pc_alpha = None # Default value
    
    
    start_time = time.time()
    results = pcmci.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    end_time = time.time()
    execution_time = end_time - start_time

    
    
    parents = pcmci.return_parents_dict(graph=results['graph'], val_matrix=results['val_matrix'])
    return parents, execution_time

def extract_parents_lpcmci(dataframe) -> tuple[dict, float]:
    '''
    Returns the parents dict and the time that took to run the algorithm
    '''
    parcorr = ParCorr(significance='analytic')
    lpcmci = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0
    )
    
    tau_max = 3
    pc_alpha = 0.05 # Default value
    
    start_time = time.time()
    lpcmci.run_lpcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Return parents dict manually
    parents = dict()
    for j in range(lpcmci.N):
        for ((i, lag_i), link) in lpcmci.graph_dict[j].items():
            if len(link) > 0 and (lag_i < 0 or i < j):
                parents[j] = parents.get(j, []) + [(i, lag_i)]
    
    return parents, execution_time


if __name__ == '__main__':
    print(test_toy_data(1, extract_parents_lpcmci))