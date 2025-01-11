from fastapi import UploadFile
from app.Algorithms.tigramiteAlgorithms import tigramite_algorithms, runTigramiteCausalDiscovery
import pandas as pd


causal_discovery_from_time_series_algorithms = tigramite_algorithms

causal_discovery_from_time_series_metrics = ["F1", "AUROC", "SHD"]

def runCausalDiscoveryFromTimeSeries(algorithm: str, datasetFIle: UploadFile)->dict:
    """
    Run the causal discovery from time series algorithm.
    
    Args:
        algorithm (str): The algorithm to run.
        datasetFile (UploadFile): The file containing the dataset.
        
    Returns:
        dict: The result of the algorithm.
    """
    if algorithm not in causal_discovery_from_time_series_algorithms:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    
    dataset = pd.read_csv(datasetFIle.file)
    
    if algorithm in tigramite_algorithms:
        graph_image = runTigramiteCausalDiscovery(algorithm, dataset)
        
        return {'graph_image': f"data:image/png;base64,{graph_image}"}
    
        
  