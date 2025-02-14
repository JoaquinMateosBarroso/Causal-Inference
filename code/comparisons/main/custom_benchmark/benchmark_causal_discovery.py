import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from create_toy_datasets import CausalDataset
from functions_test_data import get_f1, get_precision, get_recall, get_shd
from causal_discovery_base import CausalDiscoveryBase
from typing import Any, Iterator
from tigramite import plotting as tp
from tqdm import tqdm

# For printings
BLUE = '\033[34m'
GREEN = '\033[32m'
RESET = '\033[0m'

class BenchmarkCausalDiscovery:
    def __init__(self):
        self.verbose = 0
        self.results = None
        
    def benchmark_causal_discovery(self, 
                                algorithms: dict[str, type[CausalDiscoveryBase]],
                                parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]],
                                datasets: list[np.ndarray] = None,
                                datasets_folder: str = None,
                                results_folder: str = None,
                                scores: list[ str ] = ['f1', 'precision', 'recall', 'time', 'memory'],
                                n_executions: int = 3,
                                verbose: int = 0) \
                                        -> dict[str, list[ dict[str, Any] ]]:
        '''
        Function to execute a series of algorithms for causal discovery over time series datasets,
            using a series of parameters for algorithms and options in the creation of the datasets.
        Parameters:
            algorithms : dict[str, CausalDiscoveryBase]
                A dictionary where keys are the names of the algorithms and values are instances of the algorithms to be tested.
            algorithms_parameters : dict[str, Any]
                A dictionary where keys are the names of the algorithms and values are parameters for each algorithm.
            options : dict[str, list], optional
                A dictionary where keys are the names of the options and values are lists of possible values for each option.
            datasets : list[np.ndarray], optional
                A list of numpy arrays representing the datasets to be used in the benchmark.
            verbose : int, optional
                The level of comments that is going to be printed.
            datasets_folder : str, optional
                The name of the folder in which datasets will be saved. If not specified, datasets are not saved.
        Returns:
            results: dict[str, list[ dict[str, Any] ]]
                A dictionary where keys are the names of the algorithms and values are 
                    lists with dictionaries containing the results of the benchmark for each algorithm.
        '''
        self.verbose = verbose
        self.results_folder = results_folder
        self.algorithms = algorithms
        
        # A list whose items are the lists of dictionaries of results and parameters of the different executions
        self.results = {alg: list() for alg in algorithms.keys()}
        
        if datasets is None:
            self._benchmark_dataset_with_toy_data(algorithms, parameters_iterator, n_executions, datasets_folder)
        

        return self.results                        
    
    def save_results(self):    
        if self.results_folder is not None:
            # If the folder does not exist, create it
            if not os.path.exists(self.results_folder):
                os.makedirs(self.results_folder)
            # Save the results in a csv file
            for name in self.algorithms.keys():
                df = pd.DataFrame(self.results[name])
                df.to_csv(f'{self.results_folder}/results_{name}.csv', index=False)
        else:
            # If it does exist, delete previous results
            for filename in os.listdir(self.results_folder):
                if filename.endswith('.csv'):
                    os.remove(f'{self.results_folder}/{filename}')
    
    def _benchmark_dataset_with_toy_data(self, algorithms: dict[str, type[CausalDiscoveryBase]],
                                         parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]],
                                         n_executions: int,
                                         datasets_folder: str,\
                                        )  -> dict[str, list[ dict[str, Any] ]]:
        
        for iteration, current_parameters in enumerate(parameters_iterator):
            algorithms_parameters, data_option = current_parameters
            # Generate the datasets, with graph structure and time series
            causal_datasets = [CausalDataset() for _ in range(n_executions)]
            for causal_dataset in causal_datasets: 
                causal_dataset.generate_toy_data(iteration, datasets_folder=datasets_folder, **data_option)
            
            if self.verbose > 0:
                print('\n' + '-'*50)
                print(BLUE, 'Executing the datasets with option:', data_option, RESET)
            
            current_results = self.test_algorithms(causal_datasets, algorithms,
                                                   algorithms_parameters)
            
            for name, algorithm_result in current_results.items():
                algorithm_result.update(data_option) # Include the parameters in the information for results
                # Include current result in the list of result
                self.results[name].append(algorithm_result)
            
            self.save_results()
            if self.verbose > 0:
                print(f'{iteration+1} combinations executed')
            
        return self.results
    
    
    def test_algorithms(self, causal_datasets: list[CausalDataset],
                            algorithms: dict[str, type[CausalDiscoveryBase]],
                            algorithms_parameters: dict[str, type[CausalDiscoveryBase]],
                            ) -> dict[str, dict[str, Any]]:
        '''
        Execute the given algorithms and return the results
        '''
        result = dict() # keys are score names and values are the score values
        for name, algorithm in algorithms.items():
            algorithm_result = self.test_particular_algorithm(algorithm_name=name,
                                    causal_datasets=causal_datasets, causalDiscovery=algorithm,
                                    algorithm_parameters=algorithms_parameters[name])
            result[name] = (algorithm_result)

        return result
    
    def test_particular_algorithm(self, algorithm_name: str,
                                causal_datasets: list[CausalDataset],
                                causalDiscovery: type[CausalDiscoveryBase],
                                algorithm_parameters: dict[str, Any]) -> dict[str, Any]:
        '''
        Execute the given algorithm n_executions times and return the average and std of the results
        '''
        if len(causal_datasets) <= 0:
            return []
        
        if self.verbose > 0:
            print(GREEN, 'Executing algorithm', algorithm_name, RESET)
        # Execute the algorithm n_executions times
        results_per_exection = []
        for causal_dataset in tqdm(  causal_datasets ): # tqdm is used to show a progress bar
            results_per_exection.append( self.base_test_particular_algorithm(causal_dataset, causalDiscovery, algorithm_parameters) )
        
        # Calculate the average and std of the results
        aggregated_result = dict()
        for score in results_per_exection[0].keys():
            values = [result[score] for result in results_per_exection]
            aggregated_result[score + '_avg'] = np.mean(values)
            aggregated_result[score + '_std'] = np.std(values)
            
        if self.verbose > 1:
            print(f'{aggregated_result=}')
            
        return aggregated_result
        
    
    def base_test_particular_algorithm(self, causal_dataset: CausalDataset,
                      causalDiscovery: type[CausalDiscoveryBase],
                      algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
        '''
        Execute the algorithm one single time and calculate the necessary scores.
        
        Parameters:
            causal_dataset : CausalDataset with the time series and the parents
            causalDiscovery : class of the algorithm to be executed
            algorithm_parameters : dictionary with the parameters for the algorithm
        Returns:
            result : dictionary with the scores of the algorithm
        '''
        time_series = causal_dataset.time_series
        actual_parents = causal_dataset.parents_dict
        
        algorithm = causalDiscovery(data=time_series, **algorithm_parameters)
        try:
            predicted_parents, time, memory = algorithm.extract_parents_time_and_memory()
        except Exception as e:
            print(f'Error in algorithm {causalDiscovery.__name__}: {e}')
            predicted_parents = {}
            time = np.nan
            memory = np.nan
        
        finally:
            result = {'time': time, 'memory': memory}
            
            result['precision'] = get_precision(predicted_parents, actual_parents)
            result['recall'] = get_recall(predicted_parents, actual_parents)
            result['f1'] = get_f1(predicted_parents, actual_parents)
            result['shd'] = get_shd(predicted_parents, actual_parents)
        
            return result
    
    def plot_ts_datasets(self, folder_name):
        files = os.listdir(folder_name)
        data_files = filter(lambda x: x.endswith('.csv'), files)
        for filename in data_files:
            data_name = filename.split('_')[0]
            with open(f'{folder_name}/{data_name}_parents.json', 'r') as f:
                parents_dict = json.load(f)
            
            # Plot the time series dataset
            self.__plot_ts_dataset(f'{folder_name}/{filename}', parents_dict)
            plt.savefig(f'{folder_name}/{data_name}_plot.pdf')
            plt.clf()
            
            # Plot the graph structure
            self.__plot_ts_graph(parents_dict)
            plt.savefig(f'{folder_name}/{data_name}_graph.pdf')
            plt.clf()
        
    def __plot_ts_dataset(self, dataset_name, parents_dict):
        '''
        Function to plot the time series dataset and the causal graph
        '''
        dataset = pd.read_csv(dataset_name)
        time_series = dataset.values
        n_variables = time_series.shape[1]
        
        # Plot the time series
        fig, axs = plt.subplots(n_variables, 1, figsize=(10, 3*n_variables))
        for i, variable_name in enumerate(dataset.columns):
            axs[i].plot(time_series[:, i])
            axs[i].set_title(f'Variable {i}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(f'$X^{{{i}}}$')
            
            
            # Include the parents in the title
            parents = parents_dict.get(variable_name, [])
            parents_str = ', '.join([f'$X^{{{p[0]}}}_{{{p[1]}}}$' for p in parents])
            axs[i].set_title(f'$X^{{{i}}}_t$ - Parents: {parents_str}')
        
        plt.subplots_adjust(hspace=0.5)
    
    def __plot_ts_graph(self, parents_dict):
        '''
        Function to plot the graph structure of the time series
        '''
        max_lag = max([max([-tau for _, tau in parents_dict[i]]) for i in parents_dict])
        graph = np.array([[[(1 if (j, tau) in parents_dict[i] else 0 for j in parents_dict)]\
                            for i in parents_dict] for tau in range(1, max_lag+1)])
        print(f'{max_lag=}')
        print(f'{graph.shape=}')
        
        tp.plot_time_series_graph(
            graph=graph,
            var_names=list(parents_dict.keys()),
            link_colorbar_label='cross-MCI (edges)'
        )
    
    def plot_moving_results(self, results_folder, scores=['shd', 'f1', 'precision', 'recall', 'time', 'memory'],
                            x_axis='max_lag'):
        '''
        Function to plot the results of the benchmark in when a parameter is varied
        '''
        files = os.listdir(results_folder)
        results_files = filter(lambda x: x.startswith('results_') and x.endswith('.csv'), files)
        get_algo_name = lambda filename: filename.split('_')[1].split('.')[0]
        results_dataframes = {get_algo_name(filename): pd.read_csv(f'{results_folder}/{filename}')\
                                for filename in results_files}
        
        for score in scores:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            for algorithm_name, df_results in results_dataframes.items():
                x = df_results[x_axis]
                y = df_results[score + '_avg']
                std = df_results[score + '_std']
                ax.errorbar(x, y, yerr=std, label=algorithm_name,
                             fmt='.-', linewidth=1, capsize=3)
                ax.grid()
            ax.set_xlabel(x_axis)
            ax.set_ylabel(score)
            ax.legend()
            
            plt.savefig(f'{results_folder}/plot_{score}.pdf')
            plt.close(fig)

    def plot_particular_result(self, results_folder, scores=['shd', 'f1', 'precision', 'recall', 'time', 'memory']):
        '''
        Function to plot the result of the benchmark in a particular configuration (in case 
            the csv files has more than one configuration, the first one is shown)
        '''
        files = os.listdir(results_folder)
        results_files = filter(lambda x: x.startswith('results_') and x.endswith('.csv'), files)
        get_algo_name = lambda filename: filename.split('_')[1].split('.')[0]
        results_dataframes = {get_algo_name(filename): pd.read_csv(f'{results_folder}/{filename}')\
                                for filename in results_files}
        
        for score in scores:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            for iteration, df_results in enumerate(results_dataframes.values()):
                result = df_results.iloc[0, :]
                x = iteration
                y = result[score + '_avg']
                std = result[score + '_std']
                ax.errorbar(x, y, yerr=std, fmt='.-', linewidth=1, capsize=3)
                ax.grid()
                
            algorithms_names = list(results_dataframes.keys())
            ax.set_xticks(range(len(algorithms_names)), algorithms_names)
            ax.set_ylabel(score)
            
            plt.savefig(f'{results_folder}/comparison_{score}.pdf')
            plt.close(fig)

