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

from algo_tigramite import Extractor_LPCMCI, Extractor_PCMCI, Extractor_FullCI



algo_dict = {
            'PCMCI': partial(Extractor_PCMCI),
            'LPCMCI': partial(Extractor_LPCMCI),
            'FullCI': partial(Extractor_FullCI),
            'PC-PartialCorr':partial(PC, CI_test=PartialCorrelation(), use_multiprocessing=False,
                                      prior_knowledge=None),
            'Granger':partial(Granger, use_multiprocessing=False, prior_knowledge=None),
            'VARLINGAM':partial(VARLINGAM, use_multiprocessing=True, prior_knowledge=None)}

kargs_dict = {
            'PCMCI': {'tau_max': 3, 'pc_alpha': 0.01},
            'LPCMCI': {'tau_max': 3, 'pc_alpha': 0.01},
            'FullCI': {'tau_max': 3, 'pc_alpha': 0.01},
            'PC-PartialCorr': {'max_condition_set_size': 4, 'pvalue_thres': 0.01, 'max_lag': 3},
            'Granger': {'pvalue_thres': 0.01, 'max_lag': 3},
            'VARLINGAM': {'pvalue_thres': 0.01, 'max_lag': 3}}

b = BenchmarkContinuousTimeSeries(algo_dict=algo_dict, kargs_dict=kargs_dict,
                             num_exp=10, custom_metric_dict=None)


# Obtain the times taken for each algorithm, at each number of variables
times_per_vars = dict()
for num_vars in [5, 10, 20, 20]:
    b.benchmark_sample_complexity(T_list=[200, 800, 2000], num_vars=num_vars, graph_density=0.2,\
                                data_max_lag=3,
                                fn = lambda x:np.log(abs(x)) + np.sin(x), # Non-linearity
                                coef=0.1, noise_fn=np.random.randn)
    
    b.aggregate_results('time_taken')
    
    times_per_vars[num_vars] = {algo: np.mean(results) for algo, results in zip(algo_dict.keys(), b.results_mean)}


with open('results_50vars.txt', 'w') as f:
    f.write(str(b.results_full))
    
with open('times_per_vars.txt', 'w') as f:
    f.write(str(times_per_vars))

plt=b.plot('f1_score', xaxis_mode=1)
plt.savefig('f1_score_50vars.pdf')
plt.show()

plt=b.plot('time_taken', xaxis_mode=1)
plt.savefig('time_taken_50vars.pdf')
plt.show()
