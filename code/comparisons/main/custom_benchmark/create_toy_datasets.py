# %%
import json
import numpy as np
import pandas as pd
from tigramite.toymodels.structural_causal_processes import generate_structural_causal_process, structural_causal_process


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
        
    def save(self, name, dataset_folder):
        # Save the time series data to a csv file
        df = pd.DataFrame(self.time_series)
        df.to_csv(f'{dataset_folder}/{name}_data.csv', index=False, header=True)
        # Save parents to a json file
        with open(f'{dataset_folder}/{name}_parents.json', 'w') as f:
            json.dump(self.parents_dict, f)
    
    def generate_toy_data(self, name, T=100, N=10, L=10,
                      max_lag=3, dependency_funcs=['nonlinear'],
                      datasets_folder = None, **kw_generation_args) \
                            -> tuple[np.ndarray, dict[int, list[int]]]:
        """
        Generate a toy dataset with a causal process and time series data
        Parameters:
            name : Name of the dataset
            T : Number of time points
            N : Number of variables
            L : Number of cross-links between two different variables.
            max_lag : Maximum lag of the causal process
            dependency_funcs : List of dependency functions (in {'linear', 'nonlinear'}, or a function f:R->R)
            dataset_folder : Name of the folder where datasets and parents will be saved. By default they are not saved.
        Returns:
            time_series : np.ndarray with shape (n_samples, n_variables)
            parents_dict: dictionary whose keys are each node, and values are the lists of parents.
        """
        # Generate random causal process
        causal_process, noise = generate_structural_causal_process(N=N,
                                                            L=L,
                                                            max_lag=max_lag,
                                                            dependency_funcs=dependency_funcs,
                                                            **kw_generation_args)

        self.parents_dict = get_parents_dict(causal_process)

        # Generate time series data from the causal process
        self.time_series = structural_causal_process(causal_process, T=T, noises=noise)[0]
        
        if datasets_folder is not None:
            self.save(name, datasets_folder)
                
        return self

if __name__ == '__main__':
    CausalDataset().generate_toy_data('1')


