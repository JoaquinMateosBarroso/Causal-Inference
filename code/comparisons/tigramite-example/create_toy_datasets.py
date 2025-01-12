# %%
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

def generate_toy_data(name, T=100, N=10, L=10, max_lag=3, dependency_funcs=['nonlinear']):
    """
    Generate a toy dataset with a causal process and time series data
    :param name: Name of the dataset
    :param T: Number of time points
    :param N: Number of variables
    :param L: Number of cross-links between two different variables.
    :param max_lag: Maximum lag of the causal process
    :param dependency_funcs: List of dependency functions (in {'linear', 'nonlinear'})
    """
    # Generate random causal process
    causal_process = generate_structural_causal_process(N=N,
                                                        L=L,
                                                        max_lag=max_lag,
                                                        dependency_funcs=dependency_funcs,
                                                        seed=0)[0]
    actual_dict = get_parents_dict(causal_process)

    # Generate time series data from the causal process
    time_series = structural_causal_process(causal_process, T=T)[0]
    
    # Save the causal process to a txt file
    with open(f'toy_data/causal_process{name}.txt', 'w') as f:
        f.write(actual_dict.__str__())
    
    # Save the time series data to a csv file
    df = pd.DataFrame(time_series)
    df.to_csv(f'toy_data/data{name}.csv', index=False, header=False)


if __name__ == '__main__':
    generate_toy_data(1)


