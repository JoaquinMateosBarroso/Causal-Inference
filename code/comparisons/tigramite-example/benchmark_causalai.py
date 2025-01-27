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

from algo_tigramite import Extractor_PCMCI






algo_dict = {
            'PCMCI': partial(Extractor_PCMCI),
            'PC-PartialCorr':partial(PC, CI_test=PartialCorrelation(), use_multiprocessing=False,
                                      prior_knowledge=None),
            'Granger':partial(Granger, use_multiprocessing=False, prior_knowledge=None),
           'VARLINGAM':partial(VARLINGAM, use_multiprocessing=True, prior_knowledge=None)}

kargs_dict = {
            'PCMCI': {'tau_max': 3, 'pc_alpha': None},
            'PC-PartialCorr': {'max_condition_set_size': 4, 'pvalue_thres': 0.01, 'max_lag': 3},
            'Granger': {'pvalue_thres': 0.01, 'max_lag': 3},
           'VARLINGAM': {'pvalue_thres': 0.01, 'max_lag': 3}}

b = BenchmarkContinuousTimeSeries(algo_dict=algo_dict, kargs_dict=kargs_dict,
                             num_exp=10, custom_metric_dict=None)
b.benchmark_sample_complexity(T_list=[100, 500, 1000], num_vars=5, graph_density=0.1,\
                           data_max_lag=3,
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn) # default arguments in the library
# note that the first argument for all the benchmarking methods is always the list of values of the variant

plt=b.plot('f1_score', xaxis_mode=1)
# plt.savefig('myfig.pdf')
plt.show()

plt=b.plot('time_taken', xaxis_mode=1)
plt.show()