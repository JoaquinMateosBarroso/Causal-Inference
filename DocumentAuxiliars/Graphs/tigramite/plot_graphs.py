import numpy as np
import matplotlib.pyplot as plt
from tigramite.plotting import plot_graph

plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'




basic_fully_connected_graph = np.ones((6, 6))

basic_random_dag = np.array([
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0]
])

def plot_my_graph(graph):
    plot_graph(graph=graph,
            var_names=[f'$X^{i}$' for i in range(1, graph.shape[0]+1)],
            figsize=(5, 5))


if __name__ == '__main__':
    plot_my_graph(basic_random_dag)
    plt.savefig('figs/' + 'basic_random_dag.pdf')

