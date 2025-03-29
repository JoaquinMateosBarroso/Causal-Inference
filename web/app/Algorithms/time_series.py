from fastapi import UploadFile
from matplotlib import pyplot as plt
import pandas as pd


from group_causation.causal_discovery_algorithms import CausalDiscoveryBase
from group_causation.causal_discovery_algorithms import PCMCIModifiedWrapper, PCMCIWrapper, LPCMCIWrapper, PCStableWrapper
from group_causation.causal_discovery_algorithms import GrangerWrapper, VARLINGAMWrapper
from group_causation.causal_discovery_algorithms import DynotearsWrapper
from group_causation.benchmark import plot_ts_graph

# Ignore FutureWarnings, due to versions of libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from group_causation.functions_test_data import changing_N_variables, changing_preselection_alpha, static_parameters
import io
import base64

algorithms = {
    'pcmci': PCMCIWrapper,
    'dynotears': DynotearsWrapper,
    'granger': GrangerWrapper,
    'varlingam': VARLINGAMWrapper,
    'pc-stable': PCStableWrapper,
    'lpcmci': LPCMCIWrapper,
    
    # This works bad with big datasets
    # 'pcmci-modified': PCMCIModifiedWrapper,
}
algorithms_parameters = {
    # pc_alpha to None performs a search for the best alpha in tigramite algorithms
    'pcmci':     {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05, 'cond_ind_test': 'parcorr'},
    'lpcmci': {'pc_alpha': 0.01, 'min_lag': 1, 'max_lag': 3},
    'pc-stable': {'min_lag': 0, 'max_lag': 5, 'pc_alpha': 0.05, 'max_combinations': 100, 'max_conds_dim': 5},
    'granger':   {'min_lag': 0, 'max_lag': 5, 'cv': 5, },
    'varlingam': {'min_lag': 0, 'max_lag': 5},
    'dynotears': {              'max_lag': 5, 'max_iter': 1000, 'lambda_w': 0.05, 'lambda_a': 0.05},
    
    # 'pcmci-modified': {'pc_alpha': 0.05, 'min_lag': 1, 'max_lag': 5, 'max_combinations': 1,
    #                     'max_summarized_crosslinks_density': 0.2, 'preselection_alpha': 0.05},
}

data_generation_options = {
    'min_lag': 0,
    'max_lag': 5,
    'contemp_fraction': 1, # Fraction of contemporaneous links; between 0 and 1
    'crosslinks_density': 0.5, # Portion of links that won't be in the kind of X_{t-1}->X_t; between 0 and 1
    'T': 2000, # Number of time points in the dataset
    'N_vars': 5, # Number of variables in the dataset
    'confounders_density': 0.2, # Portion of dataset that will be overgenerated as confounders; between 0 and inf
    # These parameters are used in generate_structural_causal_process:
    'dependency_coeffs': [-0.3, 0.3], # default: [-0.5, 0.5]
    'auto_coeffs': [0.7], # default: [0.5, 0.7]
    'noise_dists': ['gaussian'], # deafult: ['gaussian']
    'noise_sigmas': [0.3], # default: [0.5, 2]
    
    'dependency_funcs': ['linear', 'negative-exponential', 'sin', 'cos', 'step'],
}

benchmark_options = {
    'changing_N_variables': (changing_N_variables, 
                                    {'list_N_variables': [5]}),
    
    'changing_preselection_alpha': (changing_preselection_alpha,
                                    {'list_preselection_alpha': [0.01, 0.05, 0.1, 0.2]}),
    'static': (static_parameters, {}),
}
chosen_option = 'static'


def runCausalDiscoveryFromTimeSeries(algorithm: str, parameters: dict, datasetFile: UploadFile)->dict:
    """
    Run the causal discovery from time series algorithm.
    
    Args:
        algorithm (str): The algorithm to run.
        datasetFile (UploadFile): The file containing the dataset.
        
    Returns:
        dict: The result of the algorithm.
    """
    if algorithm not in algorithms:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    else:
        df = pd.read_csv(datasetFile.file)
        
        algorithm_wrapper = algorithms[algorithm]
        algorithm: CausalDiscoveryBase = algorithm_wrapper(data=df.values, **parameters)
        
        parents = algorithm.extract_parents()
        plot_ts_graph(parents)
        
        def get_image():
            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Encode the image in base64
            graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            # Clear the plot to avoid overlapping in future plots
            plt.close()
            
            return graph_image
        
        graph_image = get_image()
        
        return {'graph_image': f"data:image/png;base64,{graph_image}"}

