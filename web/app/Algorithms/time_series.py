import asyncio
import os
import shutil
from fastapi import UploadFile
from matplotlib import pyplot as plt
import pandas as pd


from group_causation.causal_discovery import CausalDiscoveryBase
from group_causation.causal_discovery import PCMCIWrapper, LPCMCIWrapper, PCStableWrapper
from group_causation.causal_discovery import GrangerWrapper, VARLINGAMWrapper
from group_causation.causal_discovery import DynotearsWrapper
from group_causation.benchmark import plot_ts_graph
from group_causation.benchmark import BenchmarkCausalDiscovery

from fastapi.responses import JSONResponse

# Ignore FutureWarnings, due to versions of libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from group_causation.utils import static_parameters
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
    'contemp_fraction': 0.2, # Fraction of contemporaneous links; between 0 and 1
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
        
        graph_image = get_image()
        
        return {'graph_image': f"data:image/png;base64,{graph_image}"}


async def generateDataset(dataset_parameters: dict, n_datasets: int, aux_folder_name: str) -> pd.DataFrame:
    """
    Generate a dataset based on the given parameters.
    
    Args:
        aux_folder_name (str): The name of the auxiliary folder where the dataset will be created.
        dataset_parameters (dict): The parameters for generating the dataset.
        n_datasets (int): The number of datasets to generate.
        
    Returns:
        pd.DataFrame: The generated dataset.
    """
    benchmark = BenchmarkCausalDiscovery()

    # Modify parameters that need to be change from frontend
    try:
        dataset_parameters['auto_coeffs'] = [float(i) for i in dataset_parameters['auto_coeffs'].split(',')]
        dataset_parameters['dependency_coeffs'] = [float(i) for i in dataset_parameters['dependency_coeffs'].split(',')]
        dataset_parameters['dependency_funcs'] = dataset_parameters['dependency_funcs'].split(',')
        dataset_parameters['noise_dists'] = dataset_parameters['noise_dists'].split(',')
        dataset_parameters['noise_sigmas'] = [float(i) for i in dataset_parameters['noise_sigmas'].split(',')]
    except Exception as e:
        raise ValueError(f'Error in dataset parameters: {e}')
    
    datasets_folder = f'toy_data/{aux_folder_name}'
    
    # Delete previous toy data
    if os.path.exists(datasets_folder):
        for filename in os.listdir(datasets_folder):
            os.remove(f'{datasets_folder}/{filename}')
    else:
        os.makedirs(datasets_folder)

    # Generate the datasets asynchronously
    await asyncio.to_thread( benchmark.generate_datasets, 0, n_datasets, datasets_folder, dataset_parameters )
    await asyncio.to_thread( benchmark.plot_ts_datasets, datasets_folder )
    
    # Get all necessary files
    files_data = []
    for filename in os.listdir(datasets_folder):
        with open(os.path.join(datasets_folder, filename), "rb") as f:
            files_data.append({"filename": filename, 
                               "content": base64.b64encode(f.read()).decode("utf-8")})

    # Clean the datasets folder
    shutil.rmtree(datasets_folder)
    
    return JSONResponse(content={"files": files_data})

def get_image():
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image in base64
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Clear the plot to avoid overlapping in future plots
    plt.close()
    
    return graph_image