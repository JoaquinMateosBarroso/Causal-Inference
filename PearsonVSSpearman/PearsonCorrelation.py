def pearsonCorrelation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    std_x = (sum([(i - mean_x)**2 for i in x]) / n)**0.5
    std_y = (sum([(i - mean_y)**2 for i in y]) / n)**0.5
    return sum([(x - mean_x) * (y - mean_y) for x, y in zip(x,y)]) / (n * std_x * std_y)
