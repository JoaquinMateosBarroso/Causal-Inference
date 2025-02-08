import matplotlib.pyplot as plt

def save_results(benchmark, folder: str):
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