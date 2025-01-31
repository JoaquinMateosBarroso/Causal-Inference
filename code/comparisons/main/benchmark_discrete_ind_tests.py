from causalai.benchmark.time_series.continuous import BenchmarkContinuousTimeSeries
from causalai.benchmark.time_series.discrete import BenchmarkDiscreteTimeSeries

from causalai.models.time_series.pc import PCSingle, PC
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.models.common.CI_tests.kci import KCI
from causalai.data.data_generator import DataGenerator, GenerateRandomTimeseriesSEM
from causalai.models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests

from causalai.models.time_series.granger import Granger
from causalai.models.time_series.var_lingam import VARLINGAM

from functools import partial
import numpy as np

from algo_tigramite import Extractor_LPCMCI, Extractor_PCMCI, Extractor_FullCI, Extractor_DiscretizedPC

folder = 'discrete_ind_tests_results/'

n_bins = 10

algo_dict = {
            'PCMCI-CMIsymb_fixed_thres': partial(Extractor_DiscretizedPC, 
                                                cond_ind_test='CMIsymb_fixed_thres',
                                                n_bins=n_bins),
            'PCMCI-CMIsymb_analytic': partial(Extractor_DiscretizedPC,
                                                cond_ind_test='CMIsymb_analytic',
                                                n_bins=n_bins),
            'PCMCI-Gsquared': partial(Extractor_DiscretizedPC,
                                                cond_ind_test='Gsquared',
                                                n_bins=n_bins),
            }

kargs_dict = {
            'PCMCI-CMIsymb_fixed_thres': {'tau_max': 3, 'pc_alpha': 0.01},
            'PCMCI-CMIsymb_analytic': {'tau_max': 3, 'pc_alpha': 0.01},
            'PCMCI-Gsquared': {'tau_max': 3, 'pc_alpha': 0.01}
            }

b = BenchmarkContinuousTimeSeries(algo_dict=algo_dict, kargs_dict=kargs_dict,
                             num_exp=3, custom_metric_dict=None)


# Obtain the times taken for each algorithm, at each number of variables
times_per_vars = dict()
for num_vars in [10]:
    b.benchmark_sample_complexity(T_list=[100, 500, 2000], num_vars=num_vars, graph_density=0.2,\
                                data_max_lag=3,
                                fn = lambda x:np.log(abs(x)) + np.sin(x), # Non-linearity
                                coef=0.1, noise_fn=np.random.randn)
    
    b.aggregate_results('time_taken')
    
    times_per_vars[num_vars] = {algo: np.mean(results) for algo, results in zip(algo_dict.keys(), b.results_mean)}

    with open(f'{folder}results_{num_vars}vars.txt', 'w') as f:
        f.write(str(b.results_full))

    print(f'Finished {num_vars} variables')
        
with open(f'{folder}times_per_vars.txt', 'w') as f:
    f.write(str(times_per_vars))

plt=b.plot('f1_score', xaxis_mode=1)
plt.savefig(f'{folder}f1_score.pdf')
plt.show()
plt.clf()

plt=b.plot('precision', xaxis_mode=1)
plt.savefig(f'{folder}precision.pdf')
plt.show()
plt.clf()

plt=b.plot('recall', xaxis_mode=1)
plt.savefig(f'{folder}recall.pdf')
plt.show()
plt.clf()

plt=b.plot('time_taken', xaxis_mode=1)
plt.savefig(f'{folder}time_taken.pdf')
plt.show()
plt.clf()
