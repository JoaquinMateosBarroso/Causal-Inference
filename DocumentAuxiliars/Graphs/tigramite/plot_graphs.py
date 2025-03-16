import numpy as np
import matplotlib.pyplot as plt
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.plotting import plot_graph





basic_fully_connected_graph = np.ones((6, 6))

def plot_my_graph(graph):
    plot_graph(graph=graph,
            var_names=[f'$X^{i}$' for i in range(1, graph.shape[0]+1)],
            figsize=(5, 5))


if __name__ == '__main__':
    plot_my_graph(basic_fully_connected_graph)
    plt.savefig('basic_fully_connected_graph.pdf')

