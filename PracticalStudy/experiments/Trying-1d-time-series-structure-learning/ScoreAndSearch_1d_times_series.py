import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore
from itertools import combinations
from create_causal_data import create_causal_data


class ScoreAndSearch_1d_times_series:
    def __init__(self, maximum_parents=4, verbose=False):
        self.maximum_parents = maximum_parents
        self.verbose = verbose

    def score(self, graph):
        """
        This function returns the BIC score of the time series with given graph.
        
        :return: BIC score of the time series with given graph
        """
        model = BayesianNetwork(graph)
        
        return self._bic.score(model)
        
    
    def search(self, ts_df):
        """
        This function searches for the optimum graph structure, in terms of BIC score, 
        for the given set of time series.
        
        :param ts_df: The time series dataframe where each row is a time series
        :return: The most similar time series in the list
        """
        self._bic = BicScore(ts_df)
        # pgmpy takes graphs as lists of edges, where an edge is a pair of nodes
        graph = []
        for t in range(ts_df.shape[1]):
            parents = self._search_parents_for_time(t)
            graph += [(parent, t) for parent in parents]
            print(f'After time {t} the graph is {graph}')
        
        return graph
    
    
    def _search_parents_for_time(self, t) -> list[tuple[int, int]]:
        """
        This function searches for the parents of a given node in the graph.
        
        :param t: X(t) is the node for which we are searching parents
        :param ts_df: The time series dataframe where each row is a time series
        :param graph: The graph where we are adding the parents of the node
        """
        previous_nodes = list(range(t))
        optimal_parents = []
        optimal_bic = self._bic.local_score(t, optimal_parents)
        
        # Try with all possible size of parents set
        for r in range(1, min(t+1, self.maximum_parents+1)):
            # Try with all possible combinations of parents
            for parents in combinations(previous_nodes, r):
                score = self._bic.local_score(t, parents)
                if self.verbose:
                    print('Parents', parents, '. Score->', score)
                if score > optimal_bic:
                    optimal_bic = score
                    optimal_parents = parents

        return optimal_parents


if __name__ == '__main__':
    n_ts = 100
    data = [create_causal_data(n_independent_instances=3,  k_multipliers=[1, 2, 3], 
                       sigma_independent_variables=1, sigma_dependent_variables=10**-2)
        for _ in range(n_ts)]

    data = pd.DataFrame(data).rename(index={i: f"ts_{i}" for i in range(n_ts)})
    
    score_and_search = ScoreAndSearch_1d_times_series(maximum_parents=3)
    score_and_search.search(ts_df=data)