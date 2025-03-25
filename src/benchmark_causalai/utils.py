import matplotlib.pyplot as plt
from causalai.benchmark.time_series.continuous import BenchmarkContinuousTimeSeries
import numpy as np

def save_benchmark_results(benchmark: BenchmarkContinuousTimeSeries, folder: str):
    plt=benchmark.plot('f1_score', xaxis_mode=1)
    plt.savefig(f'{folder}f1_score.pdf')
    plt.show()
    plt.clf()

    plt=benchmark.plot('precision', xaxis_mode=1)
    plt.savefig(f'{folder}precision.pdf')
    plt.show()
    plt.clf()

    plt=benchmark.plot('recall', xaxis_mode=1)
    plt.savefig(f'{folder}recall.pdf')
    plt.show()
    plt.clf()

    plt=benchmark.plot('time_taken', xaxis_mode=1)
    plt.savefig(f'{folder}time_taken.pdf')
    plt.show()
    plt.clf()


def save_score_result(score: dict[int, dict[str, np.array]],
                      algorithms: list[str],
                      folder: str, name_y: str, name_x: str):
    ''' 
        Save the score results in a pdf file, where the x axis is the name_x and the y axis is the name_y
        
        :param score: the score is a dictionary whose keys are the x values, and the values are dictionaries whose keys are the algorithms
            and values are the scores of the algorithms at that x
        :type score: dict[int, dict[str, np.array]]
        :param algorithms: the algorithms that will be plotted
        :type algorithms: list[str]
        :param folder: the folder where the pdf file will be saved
        :type folder: str
        :param name_y: the name of the y axis
        :type name_y: str
        :param name_x: the name of the x axis
        :type name_x: str
        :return: None
    '''
    x = list(score.keys())
    
    for algorithm in algorithms:
        y = [score[x_value][algorithm] for x_value in x]
        plt.plot(x, y, '-o', label=algorithm)
    
    plt.legend()
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    
    plt.savefig(f'{folder}{name_y}.pdf')
    plt.clf()
    