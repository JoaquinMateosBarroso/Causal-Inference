import numpy as np
from PearsonCorrelation import pearsonCorrelation
from SpearmanCorrelation import spearmanCorrelation
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import spearmanr


np.random.seed(43)

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False


x = np.linspace(-.99, .99, 200)

y = np.arctanh(x) + np.random.normal(0, 0.1, 200)

if __name__ == '__main__':
    print('Pearson correlation ->', pearsonCorrelation(x, y))
    print('Spearman correlation ->', spearmanCorrelation(x, y))

    plt.scatter(x, y, s=5)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()  