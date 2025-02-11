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
from utils import save_benchmark_results, save_score_result

folder = 'results_max_lag/'
nonlinear_fn = lambda x: x + 5. * x**2 * np.exp(-x**2 / 20.)

algo_dict = {
            'PCMCI': partial(Extractor_PCMCI),
            # 'LPCMCI': partial(Extractor_LPCMCI),
            'FullCI': partial(Extractor_FullCI),
            # 'DiscretizedPC': partial(Extractor_DiscretizedPC), # Takes too much time
            'PC-PartialCorr':partial(PC, CI_test=PartialCorrelation(), use_multiprocessing=False,
                                      prior_knowledge=None),
            'Granger':partial(Granger, use_multiprocessing=False, prior_knowledge=None),
            'VARLINGAM':partial(VARLINGAM, use_multiprocessing=True, prior_knowledge=None)}



num_vars = 10
# Obtain the scores obtained for each algorithm, at each max_lag
times_per_max_lag = {algorithm: dict() for algorithm in algo_dict.keys()}
f1_scores = {algorithm: dict() for algorithm in algo_dict.keys()}
precision_scores = {algorithm: dict() for algorithm in algo_dict.keys()}
recall_scores = {algorithm: dict() for algorithm in algo_dict.keys()}


for max_lag in [5, 10, 25, 50, 100]:
    kargs_dict = {
            'PCMCI': {'tau_max': max_lag, 'pc_alpha': 0.01},
            # 'LPCMCI': {'tau_max': max_lag, 'pc_alpha': 0.01},
            'FullCI': {'tau_max': max_lag, 'pc_alpha': 0.01},
            # 'DiscretizedPC': {'tau_max': max_lag, 'pc_alpha': 0.01, 'n_symbs': 10}, # Takes too much time
            'PC-PartialCorr': {'max_condition_set_size': 4, 'pvalue_thres': 0.01, 'max_lag': max_lag},
            'Granger': {'pvalue_thres': 0.01, 'max_lag': max_lag},
            'VARLINGAM': {'pvalue_thres': 0.01, 'max_lag': max_lag}
            }
    
    b = BenchmarkContinuousTimeSeries(algo_dict=algo_dict, kargs_dict=kargs_dict,
                                num_exp=2, custom_metric_dict=None)
    b.benchmark_sample_complexity(T_list=[200], num_vars=num_vars, graph_density=0.2,\
                                data_max_lag=max_lag,
                                fn = lambda x: nonlinear_fn(x), # Non-linearity
                                coef=0.1, noise_fn=np.random.randn)
    
    b.aggregate_results('time_taken')
    for algo, results in zip(algo_dict.keys(), b.results_mean):
        times_per_max_lag[algo][max_lag] = results
    b.aggregate_results('f1_score')
    for algo, results in zip(algo_dict.keys(), b.results_mean):
        f1_scores[algo][max_lag] = results
    b.aggregate_results('precision')
    for algo, results in zip(algo_dict.keys(), b.results_mean):
        precision_scores[algo][max_lag] = results
    b.aggregate_results('recall')
    for algo, results in zip(algo_dict.keys(), b.results_mean):
        recall_scores[algo][max_lag] = results
    print(f'Finished {max_lag} max_lag')
    
    print(f'{f1_scores=}')


    # Save results in each iteration; they are rewritten in each iteration, but it is useful to see the progress
    save_benchmark_results(benchmark=b, folder=folder)

    with open(f'{folder}times_per_vars.txt', 'w') as f:
        f.write(str(max_lag))
        
        
    algorithms = list(algo_dict.keys())

    save_score_result(score=max_lag, algorithms=algorithms, folder=folder, name_y='time_taken', name_x='num_vars')
    save_score_result(score=f1_scores, algorithms=algorithms, folder=folder, name_y='f1_score', name_x='num_vars')
    save_score_result(score=precision_scores, algorithms=algorithms, folder=folder, name_y='precision', name_x='num_vars')
    save_score_result(score=recall_scores, algorithms=algorithms, folder=folder, name_y='recall', name_x='num_vars')