from app.Algorithms.tigramiteAlgorithms import tigramite_algorithms, runTigramiteCausalDiscovery



causal_discovery_from_time_series_algorithms = tigramite_algorithms

causal_discovery_from_time_series_metrics = ["F1", "AUROC", "SHD"]

def runCausalDiscoveryFromTimeSeries(algorithm: str):
    """
    Run the causal discovery from time series algorithm.
    
    Args:
        algorithm (str): The algorithm to run.
        
    Returns:
        dict: The result of the algorithm.
    """
    if algorithm not in causal_discovery_from_time_series_algorithms:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    
    if algorithm in tigramite_algorithms:
        return runTigramiteCausalDiscovery(algorithm)