import numpy as np
import pandas as pd

def create_causal_data(n_independent_instances:int, k_multipliers: list[int], 
                       sigma_independent_variables: float, sigma_dependent_variables) -> pd.DataFrame:
    '''We create a simple dataset where the single feature is correlated with
        those s and i·s time steps ahead of them, forall i in range(k_multipliers).
        
        Independent instances:
            X(t) := N(0, sigma_independent_variables)
        Dependent instances:
            X(t) := k_i*X(t-i) + N(0, sigma_dependent_variables) for k_i in k_multipliers
            or, equivalently:
            X(t+i) = k_i*X(t) + N(0, sigma_dependent_variables) for k_i in k_multipliers
    '''
    independent_instances = np.random.normal(0, sigma_independent_variables, n_independent_instances)
    
    dependent_instances = np.array([k*independent_instances[t] + np.random.normal(0, sigma_dependent_variables)
                                      for k in k_multipliers for t in range(n_independent_instances)])
    
    time_series = np.concatenate([independent_instances, dependent_instances])
    
    return time_series
    
    

if __name__ == '__main__':
    data = create_causal_data(n_independent_instances=3,  k_multipliers=[1, 2, 3], 
                       sigma_independent_variables=1, sigma_dependent_variables=10**-2)
    print(data)
    
    pd.DataFrame(data, columns=['x(t)']).to_csv('causal_related_time_series.csv', index_label='t')