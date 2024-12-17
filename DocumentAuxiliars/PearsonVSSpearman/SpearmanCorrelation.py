from PearsonCorrelation import pearsonCorrelation

def spearmanCorrelation(x, y):
    R_x = [sorted(x).index(i) for i in x]
    R_y = [sorted(y).index(i) for i in y]
    return pearsonCorrelation(R_x, R_y)