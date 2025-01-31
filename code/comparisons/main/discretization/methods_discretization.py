import numpy as np
import pandas as pd
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation

def equal_width_binning(time_series: np.ndarray, bins: int = 10, use_labels: bool = False) -> np.ndarray:
    """
    Discretize a 1D time series into bins of equal width.

    Args:
        time_series (np.ndarray): 1D array of numerical data.
        bins (int, optional): Number of bins to use. Defaults to 10.
        use_labels (bool, optional):
            If True, returns discrete labels (integers) for each bin.
            If False, returns the bin midpoints corresponding to each value.
            Defaults to False.

    Returns:
        np.ndarray:
            - If use_labels is True, an array of integer labels indicating the bin index.
            - If use_labels is False, an array of bin midpoint values.
    """
    if use_labels:
        # Return integer labels using pandas cut
        labels = range(bins)
        return pd.cut(time_series, bins=bins, labels=labels).astype(int)
    else:
        # Calculate bin edges and assign midpoints
        bin_edges = np.linspace(time_series.min(), time_series.max(), bins + 1)
        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_indices = np.digitize(time_series, bin_edges, right=True) - 1
        # Clip the bin indices to avoid out-of-bounds indexing
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        return midpoints[bin_indices]


def equal_frequency_binning(time_series: np.ndarray, bins: int = 10, use_labels: bool = False) -> np.ndarray:
    """
    Discretize a 1D time series into bins so that each bin has (approximately)
    the same number of data points.

    Args:
        time_series (np.ndarray): 1D array of numerical data.
        bins (int, optional): Number of bins (quantiles) to use. Defaults to 10.
        use_labels (bool, optional):
            If True, returns discrete labels (integers) for each bin.
            If False, returns the bin midpoints corresponding to each value.
            Defaults to False.

    Returns:
        np.ndarray:
            - If use_labels is True, an array of integer labels indicating the bin index.
            - If use_labels is False, an array of bin midpoint values.
    """
    # Calculate the quantiles
    quantiles = np.quantile(time_series, np.linspace(0, 1, bins + 1))

    if use_labels:
        labels = range(bins)
        # Use qcut to create equally populated bins
        return pd.qcut(time_series, q=bins, labels=labels).astype(int)
    else:
        # Assign midpoints based on quantile edges
        midpoints = (quantiles[:-1] + quantiles[1:]) / 2
        bin_indices = np.digitize(time_series, quantiles, right=True) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        return midpoints[bin_indices]


def paa_method(time_series: np.ndarray, window_size: int = 10, expand: bool = True) -> np.ndarray:
    """
    Apply Piecewise Aggregate Approximation (PAA) to a 1D time series.

    Args:
        time_series (np.ndarray): 1D array of numerical data.
        window_size (int, optional): Size of each segment used by PAA. Defaults to 10.
        expand (bool, optional):
            If True, the transformed series is repeated to match the original length
            (truncating excess if the original length is not a multiple of window_size).
            If False, returns only the reduced PAA result. Defaults to True.

    Returns:
        np.ndarray:
            - If expand is True, a 1D array approximating the original length.
            - If expand is False, a 1D array of length (len(time_series) / window_size)
              or the next integer truncation thereof, depending on the PAA implementation.
    """
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    transformed = paa.transform(time_series.reshape(1, -1))[0]

    if expand:
        # Repeat each PAA segment value to match original approximate length
        expanded = np.repeat(transformed, window_size)
        # Truncate if it exceeds the original length
        expanded = expanded[: len(time_series)]
        return expanded

    return transformed


def sax_method( time_series: np.ndarray, n_bins: int = 5, window_size: int = 10, strategy: str = 'quantile',
                expand: bool = True, use_labels: bool = False) -> np.ndarray:
    """
    Apply Symbolic Aggregate approXimation (SAX) to a 1D time series.

    Steps:
    1. Perform Piecewise Aggregate Approximation (PAA).
    2. Convert PAA-reduced time series to symbolic form using SAX.

    Args:
        time_series (np.ndarray): 1D array of numerical data.
        n_bins (int, optional): Number of discrete symbols to use for SAX. Defaults to 5.
        window_size (int, optional): Size of each segment used by PAA. Defaults to 10.
        strategy (str, optional): Method for determining SAX breakpoints, e.g. 'quantile'. Defaults to 'quantile'.
        expand (bool, optional):
            If True, repeats either the symbols (if use_labels=True) or the representative means
            (if use_labels=False) to match the original series length. Truncates excess if needed.
            If False, returns only the reduced output. Defaults to True.
        use_labels (bool, optional):
            If True, returns symbolic (string) labels from SAX.
            If False, returns the numeric means corresponding to each symbol. Defaults to False.

    Returns:
        np.ndarray:
            - If use_labels is True and expand is True, a 1D array of symbols repeated to match original length.
            - If use_labels is True and expand is False, a 1D array of symbols (length of reduced PAA).
            - If use_labels is False and expand is True, a 1D array of numeric approximations repeated.
            - If use_labels is False and expand is False, a 1D array of numeric approximations (length of reduced PAA).
    """
    # 1. Piecewise Aggregate Approximation (PAA)
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    x_paa = paa.transform(time_series.reshape(1, -1))[0]

    # 2. Symbolic Aggregate Approximation (SAX)
    sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)
    x_sax_symbols = sax.transform(x_paa.reshape(1, -1))[0]

    if use_labels:
        # Return symbolic labels
        if expand:
            expanded_symbols = np.repeat(x_sax_symbols, window_size)
            # Truncate to original length
            expanded_symbols = expanded_symbols[: len(time_series)]
            return expanded_symbols
        return x_sax_symbols
    else:
        # Convert SAX symbols to their corresponding mean values in the PAA-reduced series
        unique_symbols, inverse_indices = np.unique(x_sax_symbols, return_inverse=True)
        # For each symbol, compute the mean of the PAA values that map to that symbol
        symbol_means = np.array([x_paa[inverse_indices == i].mean()
                                 for i in range(len(unique_symbols))])

        if expand:
            # Map each original symbol index to its mean, then expand
            expanded_means = symbol_means[inverse_indices]
            expanded_means = np.repeat(expanded_means, window_size)
            expanded_means = expanded_means[: len(time_series)]
            return expanded_means

        # Return only the mapped means for the reduced SAX series
        return symbol_means[inverse_indices]


def pla_segmentation(time_series: np.ndarray, window_size: int, threshold: float = 0.01, expand: bool = True,
                     use_labels: bool = False) -> np.ndarray:
    """
    Perform Piecewise Linear Approximation (PLA) on a 1D time series in fixed-sized segments.

    Each segment is approximated by a linear fit (first-degree polynomial).
    If use_labels is True, returns slope-based labels for each segment:
        - 1 for upward trend (slope > threshold)
        - -1 for downward trend (slope < -threshold)
        - 0 for flat (abs(slope) <= threshold)

    Args:
        time_series (np.ndarray): 1D array of numerical data.
        window_size (int): Size of each segment for linear fitting.
        threshold (float, optional): Threshold for categorizing slopes as up/down/flat. Defaults to 0.01.
        expand (bool, optional):
            If True, repeats the final labels or fitted values to the original length.
            The last window might be ignored if its length < 2. Defaults to True.
        use_labels (bool, optional):
            If True, returns a 1D array of slope-based labels.
            If False, returns a 1D array with the piecewise linear reconstructed time series.
            Defaults to False.

    Returns:
        np.ndarray:
            - If use_labels is True, an array of slope-based integer labels (+1, -1, 0).
            - If use_labels is False, an array of the same length as time_series, containing
              the piecewise linear approximation. The last few points may remain 0 if
              they form a segment with length < 2 (skipped).
    """
    n = len(time_series)
    reconstructed = np.zeros_like(time_series)
    labels_per_segment = []

    # Process segments of length window_size
    for start_idx in range(0, n, window_size):
        segment = time_series[start_idx: start_idx + window_size]
        if len(segment) < 2:
            # Ignore segments too small for a linear fit
            continue

        # Linear fit (degree 1)
        x = np.arange(len(segment))
        y = segment
        coef = np.polyfit(x, y, 1)
        slope = coef[0]

        # Determine label based on slope
        if slope > threshold:
            label = 1  # Upward
        elif slope < -threshold:
            label = -1  # Downward
        else:
            label = 0  # Flat
        labels_per_segment.append(label)

        # Reconstruct the linear fit for this segment
        y_fit = np.polyval(coef, x)
        reconstructed[start_idx: start_idx + len(segment)] = y_fit

    if use_labels:
        if expand:
            # Repeat each segment's label window_size times
            expanded_labels = np.repeat(labels_per_segment, window_size)
            # Truncate to original length
            return expanded_labels[:n]
        else:
            # Return one label per segment
            return np.array(labels_per_segment, dtype=int)

    # If not using labels, return the reconstructed series
    return reconstructed

