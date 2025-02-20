import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from create_toy_datasets import CausalDataset
from functions_test_data import get_f1, get_precision, get_recall, get_shd, window_to_summary_graph
from causal_discovery_algorithms.causal_discovery_base import CausalDiscoveryBase
from typing import Any, Iterator
from tigramite import plotting as tp
from tigramite.graphs import Graphs
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
        -----------
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
        --------
            results: dict[str, list[ dict[str, Any] ]]
                A dictionary where keys are the names of the algorithms and values are 
                    lists with dictionaries containing the results of the benchmark for each algorithm.
        '''
        self.verbose = verbose
        self.results_folder = results_folder
        self.algorithms = algorithms
        self.all_algorithms_parameters = {name: list() for name in algorithms.keys()}
        
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
                # Create a dataframe with the results and the parameters
                df = pd.concat([pd.DataFrame(self.results[name]),
                                pd.DataFrame(self.all_algorithms_parameters[name]) ],
                                axis=1)
                               
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
            current_algorithms_parameters, data_option = current_parameters
            # Delete previous toy data
            if os.path.exists(datasets_folder):
                for filename in os.listdir(datasets_folder):
                        os.remove(f'{datasets_folder}/{filename}')
            else:
                os.makedirs(datasets_folder)
            
            # Generate the datasets, with their graph structure and time series
            causal_datasets = [CausalDataset() for _ in range(n_executions)]
            for current_dataset_index, causal_dataset in enumerate(causal_datasets):
                dataset_index = iteration * n_executions + current_dataset_index
                causal_dataset.generate_toy_data(dataset_index, datasets_folder=datasets_folder, **data_option)
            
            if self.verbose > 0:
                print('\n' + '-'*50)
                print(BLUE, 'Executing the datasets with option:', data_option, RESET)
            
            # Generate and save results of all algorithms with current dataset options
            current_results = self.test_algorithms(causal_datasets, algorithms,
                                                   current_algorithms_parameters)
            
            for name, algorithm_result in current_results.items():
                algorithm_result.update(data_option) # Include the parameters in the information for results
                # Include current result in the list of result
                self.results[name].append(algorithm_result)
                self.all_algorithms_parameters[name].\
                            append(copy.deepcopy(current_algorithms_parameters[name]))
                
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
        actual_parents_summary = window_to_summary_graph(actual_parents)
        
        algorithm = causalDiscovery(data=time_series, **algorithm_parameters)
        try:
            predicted_parents, time, memory = algorithm.extract_parents_time_and_memory()
        except Exception as e:
            print(f'Error in algorithm {causalDiscovery.__name__}: {e}')
            print('Returning nan values for this algorithm')
            predicted_parents = {}
            time = np.nan
            memory = np.nan
        
        finally:
            result = {'time': time, 'memory': memory}
            
            result['precision'] = get_precision(actual_parents, predicted_parents)
            result['recall'] = get_recall(actual_parents, predicted_parents)
            result['f1'] = get_f1(actual_parents, predicted_parents)
            result['shd'] = get_shd(actual_parents, predicted_parents)
            
            # Obtain the same metrics in the summary graph
            predicted_parents_summary = window_to_summary_graph(predicted_parents)
            result['precision_summary'] = get_precision(actual_parents_summary, predicted_parents_summary)
            result['recall_summary'] = get_recall(actual_parents_summary, predicted_parents_summary)
            result['f1_summary'] = get_f1(actual_parents_summary, predicted_parents_summary)
            result['shd_summary'] = get_shd(actual_parents_summary, predicted_parents_summary)
            
            return result
    
    def plot_ts_datasets(self, folder_name):
        files = os.listdir(folder_name)
        data_files = filter(lambda x: x.endswith('.csv'), files)
        for filename in data_files:
            data_name = filename.split('_')[0]
            with open(f'{folder_name}/{data_name}_parents.txt', 'r') as f:
                parents_dict = eval(f.read())
            
            # Plot the time series dataset
            fig, axs = self._plot_ts_dataset(f'{folder_name}/{filename}', parents_dict)
            plt.savefig(f'{folder_name}/{data_name}_plot.pdf')
            plt.clf()
            plt.close(fig)
            
            # Plot the graph structure
            self._plot_ts_graph(parents_dict)
            plt.savefig(f'{folder_name}/{data_name}_graph.pdf')
            plt.clf()
            
        
    def _plot_ts_dataset(self, dataset_name, parents_dict):
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
            parents = parents_dict.get(int(variable_name), [])
            parents_str = ', '.join([f'$X^{{{p[0]}}}_{{{p[1]}}}$' for p in parents])
            axs[i].set_title(f'$X^{{{i}}}_t$ - Parents: {parents_str}')
        
        plt.subplots_adjust(hspace=0.5)
        
        return fig, axs
    
    def _plot_ts_graph(self, parents_dict):
        '''
        Function to plot the graph structure of the time series
        '''
        graph = Graphs.get_graph_from_dict(parents_dict)
        
        tp.plot_time_series_graph(
            graph=graph,
            var_names=list(parents_dict.keys()),
            link_colorbar_label='cross-MCI (edges)',
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

    def plot_particular_result(self, results_folder,
                                     output_folder=None,
                                     scores=['shd', 'f1', 'precision', 'recall', 'time', 'memory']):
        '''
        Function to plot the result of the benchmark in a particular configuration (in case 
            the csv files has more than one configuration, the first one is shown)
        '''
        if output_folder is None:
            output_folder = results_folder
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
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
                ax.boxplot([y], positions=[x], widths=0.6)
                ax.grid()
                if score in ['f1', 'precision', 'recall', 
                        'f1_summary', 'precision_summary', 'recall_summary']:
                    ax.set_ylim(0, 1)
                
            algorithms_names = list(results_dataframes.keys())
            ax.set_xticks(range(len(algorithms_names)), algorithms_names)
            ax.set_ylabel(score)
            
            plt.savefig(f'{output_folder}/comparison_{score}.pdf')
            plt.close(fig)

