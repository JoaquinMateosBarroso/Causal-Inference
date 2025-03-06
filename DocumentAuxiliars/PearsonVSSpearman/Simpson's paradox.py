import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

from PearsonCorrelation import pearsonCorrelation
from SpearmanCorrelation import spearmanCorrelation

np.random.seed(43)

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False



x1 = np.linspace(0, 20, 100)
x2 = np.linspace(15, 35, 100)
x3 = np.linspace(30, 50, 100)

y1 = 2*(-x1 + 25) + np.random.normal(0, 2, 100)
y2 = 2*(-x2 + 50) + np.random.normal(0, 2, 100)
y3 = 2*(-x3 + 75) + np.random.normal(0, 2, 100)

x = np.concatenate([x1, x2, x3])
y = np.concatenate([y1, y2, y3])
y_reg = linregress(x, y)


if __name__ == '__main__':
    print('Pearson correlation ->', pearsonCorrelation(x, y))
    print('Spearman correlation ->', spearmanCorrelation(x, y))
    
    
    plt.scatter(x1, y1, color='r', s=5, label='Group 1')
    plt.plot(x1, linregress(x1, y1).intercept + linregress(x1, y1).slope*x1, color='r')

    plt.scatter(x2, y2, color='g', s=5, label='Group 2')
    plt.plot(x2, linregress(x2, y2).intercept + linregress(x2, y2).slope*x2, color='g')

    plt.scatter(x3, y3, color='b', s=5, label='Group 3')
    plt.plot(x3, linregress(x3, y3).intercept + linregress(x3, y3).slope*x3, color='b')

    plt.plot(x, y_reg.intercept + y_reg.slope*x, color='black', label='Linear regression (all data)')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()

    plt.savefig('simpsons-paradox.pdf')