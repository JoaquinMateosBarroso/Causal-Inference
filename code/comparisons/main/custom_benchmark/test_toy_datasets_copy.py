# %%
from modified_pcmci.create_toy_datasets import generate_toy_data
from functions_test_toy_data import test_toy_data, extract_parents_pcmci, extract_parents_pcmciplus, extract_parents_lpcmci

import pandas as pd

# %%
from itertools import product

def create_and_test_toy_data():
    # values = {
    #     'max_lag': [3],
    #     'dependency_funcs': [['linear', 'nonlinear']],
    #     'L': [5, 10, 15, 20],
    #     'T': [100, 250, 500, 1000],
    #     'N': [5, 10, 25, 50],
    # }
    values = {
        'max_lag': [3],
        'dependency_funcs': [['nonlinear']],
        'L': [5, 10],
        'T': [100, 250],
        'N': [5, 10],
    }
    

    algorithms = {
        'pcmci': extract_parents_pcmci,
        'pcmciplus': extract_parents_pcmciplus,
        'lpcmci': extract_parents_lpcmci
    }
    
    # A list whose items are the dictionaries of results of the different executions
    results = {
        'pcmci': [],
        'pcmciplus': [],
        'lpcmci': [],
    }
    
    combinations = list(product(*values.values()))
    total = len(combinations)
    current = 1
    
    for iteration, combination in enumerate(combinations):
        params = dict(zip(values.keys(), combination))
        generate_toy_data(iteration, **params)
        
        for name, algorithm in algorithms.items():
            try:
                algorithm_results = test_toy_data(iteration, algorithm)
            except ValueError as e:
                print(f'Error in iteration {iteration} with algorithm {name}: {e}')
                continue
            algorithm_results.update(params) # Include the parameters in the information for results
            results[name].append(algorithm_results)
        
        print(f'{current}/{total} combinations tested')
        current += 1
    
    
    # Save the results in a csv file
    for name in algorithms.keys():
        df = pd.DataFrame(results[name])
        df = df[['T', 'N', 'L', 'dependency_funcs', 'max_lag', 'time', 'f1', 'precision', 'recall']]
        df.to_csv(f'results_{name}.csv', index=False)
        
create_and_test_toy_data()


