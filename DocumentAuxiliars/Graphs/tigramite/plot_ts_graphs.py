import numpy as np
import matplotlib.pyplot as plt
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.plotting import plot_time_series_graph


ts_basic_window_graph = np.array([
                  [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]],])

ts_basic_extended_summary_graph = np.array([
                  [[0, 1], [1, 1], [0, 0]],
                  [[0, 0], [0, 1], [0, 1]],
                  [[0, 0], [0, 0], [0, 1]],])


ts_basic_summary_graph = np.array([
                  [[1], [1], [0]],
                  [[0], [1], [1]],
                  [[0], [0], [1]],])

ts_basic_fully_connected_graph = np.ones((2, 2, 4))

def plot_graph(graph):
    plot_time_series_graph(graph=graph,
            var_names=[f'$X^{i}$' for i in range(1, graph.shape[0]+1)],
            figsize=(5, 5))


if __name__ == '__main__':
    plot_graph(ts_basic_fully_connected_graph)
    plt.savefig('figs/' + 'ts_basic_fully_connected_graph.pdf')

