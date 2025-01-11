import pandas as pd

from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from tigramite.independence_tests.parcorr import ParCorr

from tigramite import data_processing as pp

import matplotlib.pyplot as plt

import io
import base64

from tigramite import plotting as tp

tigramite_algorithms = ['PCMCI', 'PCMCIplus', 'LPCMCI']
def runTigramiteCausalDiscovery(algorithmName: str,
                                   data: pd.DataFrame,
                                   max_lag: int = 3,
                                   cond_ind_test = ParCorr(),
                                   **kwargs)->dict:
    """
    Run the specified Tigramite algorithm.
    
    Args:
        algorithmName (str): The name of the algorithm to run.
        data (pd.DataFrame): The data to run the algorithm on.
        max_lag (int): The maximum lag to consider.
        cond_ind_test: The conditional independence test to use.
        **kwargs: Additional arguments to pass to the algorithm.
        
    Returns:
        dict: The result of the algorithm.
    """
    tigramite_dataframe = pp.DataFrame(data.values, var_names=data.columns)
    if algorithmName == 'PCMCI':
        pcmci = PCMCI(dataframe=tigramite_dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_max=max_lag)
    elif algorithmName == 'PCMCIplus':
        rpcmci = PCMCI(dataframe=tigramite_dataframe, cond_ind_test=cond_ind_test)
        results = rpcmci.run_pcmciplus(tau_max=max_lag)
    elif algorithmName == 'LPCMCI':
        lpcmci = LPCMCI(dataframe=tigramite_dataframe, cond_ind_test=cond_ind_test)
        results = lpcmci.run_lpcmci(tau_max=max_lag, tau_min=1)
    else:
        raise ValueError(f'Unknown algorithm: {algorithmName}')
    
    graph_image = getGraphImageFromResults(results, data.columns)
    
    return graph_image


def getGraphImageFromResults(results: dict, var_names: list[str]):
    # Generate the plot
    fig, _ = plt.subplots(figsize=(8, 8))
    tp.plot_time_series_graph(
        figsize=(8, 8),
        node_size=0.05,
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='MCI',
    )
    
    # Save the plot to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # Encode the image as Base64
    graph_image = base64.b64encode(buf.read()).decode("utf-8")
    
    return graph_image
        
