import numpy as np

# TODO: Adapt this function to our class structure
def trace_method(X,Y):
    ##implementation of the trace method of (Janzing et al, 2010, Telling cause from effect based on high- dimensional observations).
    # Returns delta values as defined in the paper.
    Xt = X.values.T
    Yt = Y.values.T
    CovX = np.cov(Xt)
    CovY = np.cov(Yt)
    XandY = np.concatenate((Xt, Yt), axis=0)
    sizeX = CovX.shape[0]
    sizeY = CovY.shape[0]
    CovXY = np.cov(XandY)[:sizeX, sizeX:]
    CovYX = CovXY.T
    pinvX = np.linalg.pinv(CovX)
    pinvY = np.linalg.pinv(CovY)
    A1 = np.matmul(CovYX, pinvX)
    A2 = np.matmul(CovXY, pinvY)
    rank1 = np.trace(np.matmul(CovX, pinvX))
    rank2 = np.trace(np.matmul(CovY, pinvY))
    delta1 = np.log(np.trace(np.matmul(np.matmul(A1, CovX), A1.T)) / sizeX) - np.log(
        np.trace(np.matmul(A1, A1.T)) / rank1) - np.log(np.trace(CovX) / sizeX)
    delta2 = np.log(np.trace(np.matmul(np.matmul(A2, CovY), A2.T)) / sizeY) - np.log(
        np.trace(np.matmul(A2, A2.T)) / rank2) - np.log(np.trace(CovY) / sizeY)
    return delta1, delta2


def identify_causal_direction_trace_method(X,Y,eps=0.1):
    ##function that executes trace method.
    # OUTPUT: same format as identify_causal_direction
    test_results = trace_method(X,Y)
    X_criterion = test_results[0]
    Y_criterion = test_results[1]
    if  abs(Y_criterion) > (1.0+eps)*abs(X_criterion):
        comparison = 'X is the cause of Y'
    elif abs(X_criterion) > (1.0+eps)*abs(Y_criterion):
        comparison = 'Y is the cause of X'
    else:
        comparison = 'Causal direction can not be inferred'
    return comparison, X_criterion, Y_criterion