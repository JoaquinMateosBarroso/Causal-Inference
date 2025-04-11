import numpy as np
import matplotlib.pyplot as plt
# from tigramite.plotting import plot_time_series_graph
from tigramite_plotting_modified import plot_time_series_graph

plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

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

def plot_graph(graph, summary=False):
    plot_time_series_graph(graph=graph, summary=summary,
            var_names=[f'$X^{i}$' for i in range(1, graph.shape[0]+1)],
            figsize=(5, 5))


if __name__ == '__main__':
    plot_graph(ts_basic_fully_connected_graph, summary=False)
    plt.savefig('figs/' + 'ts_fully_connected_graph_more_lag.pdf')

