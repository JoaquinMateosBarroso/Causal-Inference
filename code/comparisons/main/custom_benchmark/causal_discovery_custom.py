import numpy as np
import networkx as nx
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import scipy.stats as stats








def check_stationarity(X, significance=0.05):
    """
    Check covariance stationarity for each series in X using the Augmented Dickey-Fuller test.
    Returns a list of booleans indicating whether each series is stationary.
    """
    d = X.shape[1]
    stationary = []
    for i in range(d):
        result = adfuller(X[:, i])
        p_value = result[1]
        stationary.append(p_value < significance)
    return stationary

def create_lag_matrix(X, lags):
    """
    Construct a lagged design matrix from the time series data.
    For each time t (starting at t=lags), the row contains the lagged values
    [X[t-1], X[t-2], ..., X[t-lags]] flattened into a single vector.
    """
    T, d = X.shape
    rows = []
    for t in range(lags, T):
        row = []
        for lag in range(1, lags + 1):
            row.extend(X[t - lag])
        rows.append(row)
    return np.array(rows)

def multivariate_granger_causality(X, max_lag=5, alpha=0.025):
    """
    Implements the multivariate Granger causality algorithm.
    
    [Geweke, 1982; Chen et al., 2004; Barrett et al.,2010]
    
    Parameters:
      X      : np.array of shape (T, d) containing the time series data.
      maxlag : Maximum lag to consider for lag order selection.
      alpha  : Significance level for the F-test.
      
    Returns:
      G      : A networkx.DiGraph where an edge p -> q indicates that series p
               Granger-causes series q.
    """
    # Check stationarity for each series
    stationarity = check_stationarity(X)
    if not all(stationarity):
        print("Warning: Some series may not be stationary.")

    # Use VAR to select optimal lag (using AIC)
    var_model = VAR(X)
    order_selection = var_model.select_order(maxlags=max_lag)
    # Use the AIC-selected order; if not found, default to lag 1.
    optimal_lag = order_selection.selected_orders.get('aic', max_lag)
    if optimal_lag < 1:
        optimal_lag = 1
    print("Optimal lag selected:", optimal_lag)

    # Construct the lagged predictor matrix (all variables, all lags)
    X_lag = create_lag_matrix(X, optimal_lag)
    n_obs = X_lag.shape[0]

    # Initialize an empty directed graph with nodes for each series
    G = nx.DiGraph()
    d = X.shape[1]
    G.add_nodes_from(range(d))

    # Loop over each target variable q
    for q in range(d):
        # Response: values of series q, starting from time optimal_lag
        y = X[optimal_lag:, q]

        # Full model: use all lagged predictors (with constant)
        X_full = sm.add_constant(X_lag)
        full_model = sm.OLS(y, X_full).fit()
        RSS_full = np.sum(full_model.resid ** 2)
        df_full = X_full.shape[1]  # number of parameters in full model

        # For each candidate predictor variable p (exclude self)
        for p in range(d):
            if p == q:
                continue

            # Identify columns corresponding to variable p across all lags.
            # The design matrix is organized as:
            # [lag1_var0, lag1_var1, ..., lag1_var(d-1),
            #  lag2_var0, ..., lag_optimalLag_var(d-1)]
            cols_to_remove = [lag * d + p for lag in range(optimal_lag)]
            # Build restricted design matrix (drop all lags of variable p)
            X_restricted = np.delete(X_lag, cols_to_remove, axis=1)
            X_restricted = sm.add_constant(X_restricted)
            restricted_model = sm.OLS(y, X_restricted).fit()
            RSS_restricted = np.sum(restricted_model.resid ** 2)
            df_restricted = X_restricted.shape[1]

            # Degrees of freedom difference is number of parameters dropped
            df_diff = df_full - df_restricted

            # Compute the F-statistic:
            F_stat = ((RSS_restricted - RSS_full) / df_diff) / (RSS_full / (n_obs - df_full))
            p_value = 1 - stats.f.cdf(F_stat, df_diff, n_obs - df_full)

            # If p-value is significant, add edge from p to q
            if p_value < alpha:
                G.add_edge(p, q)

    return G


# Example usage:
if __name__ == "__main__":
    # Generate synthetic data: 200 time points, 4 variables.
    np.random.seed(42)
    T = 200
    d = 3
    X = np.random.randn(T, d)
    X_lagged = np.roll(X, 2, axis=0) + np.random.randn(T, d) * 0.1
    X = np.append(X, X_lagged, axis=1)
    print("Time series data shape:", X.shape)
    print(X)

    # Run the multivariate Granger causality procedure
    G = multivariate_granger_causality(X, max_lag=5, alpha=0.05)
    print("Edges in the Granger causality graph (p -> q):")
    print(list(G.edges()))
