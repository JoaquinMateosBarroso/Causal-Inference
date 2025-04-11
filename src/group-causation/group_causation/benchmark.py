from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from typing import Any, Iterator
from tqdm import tqdm

# Inner library imports
from group_causation.causal_groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase
from group_causation.causal_groups_extraction.stat_utils import get_average_pc1_explained_variance, get_normalized_mutual_information, get_explainability_score
from group_causation.create_toy_datasets import CausalDataset, plot_ts_graph
from group_causation.utils import get_FN, get_FP, get_TP, get_f1, get_precision, get_recall, get_shd, window_to_summary_graph
from group_causation.causal_discovery.causal_discovery_base import CausalDiscoveryBase
from group_causation.group_causal_discovery.direction_extraction.direction_extraction_base import DirectionExtractorBase
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscoveryBase

# For printings
BLUE = '\033[34m'
GREEN = '\033[32m'
RESET = '\033[0m'


class BenchmarkBase(ABC):
    '''Abstract class with functions that are useful to benchmark different kinds of algorithms'''
    def __init__(self):
        self.verbose = 0
        self.results = None
        self.results_folder = None
        self.algorithms = None
        self.all_algorithms_parameters = None
        
    @abstractmethod
    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                      causalDiscovery: type[CausalDiscoveryBase],
                      algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
        '''
        To be implemented by subclasses.
        '''
        pass


    @abstractmethod
    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                    causalDiscovery: type[CausalDiscoveryBase],
                    algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
        '''
        To be implemented by subclasses.
        '''
        pass
    
    @abstractmethod
    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        To be implemented by subclasses.
        '''
        pass
    
    @abstractmethod
    def load_datasets(self, datasets_folder):
        '''
        To be implemented by subclasses.
        '''
        pass
    
    def benchmark_causal_discovery(self, 
                                algorithms: dict[str, type[CausalDiscoveryBase]],
                                parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]],
                                generate_toy_data: bool = False,
                                datasets_folder: str = None,
                                results_folder: str = None,
                                n_executions: int = 3,
                                verbose: int = 0,
                                )        -> dict[str, list[ dict[str, Any] ]]:
        '''
        Function to execute a series of algorithms over time series datasets,
        using a series of parameters for algorithms and options in the creation of the datasets.
        
        Args:
            algorithms : dict[str, CausalDiscoveryBase]
                A dictionary where keys are the names of the algorithms and values are instances of the algorithms to be tested.
            parameters_iterator : Iterator[tuple[dict[str, Any], dict[str, Any]]]
                An iterator that returns a tuple with the parameters for the algorithms and the options for the datasets.
                Note: If generate_toy_data is False, iterations are used for algorithms parameters.
            generate_toy_data : bool, optional
                If True, the datasets are generated with the options in the parameters_iterator.
                If False, the datasets are taken from the datasets_folder.
            datasets_folder : str, optional
                The name of the folder in which datasets will be saved. If not specified, datasets are not saved.
            results_folder : str, optional
                The name of the folder in which results will be saved. If not specified, results are not saved.
            n_executions : int, optional
                The number of executions for each combination of parameters.
            verbose : int, optional
                The level of comments that is going to be printed.
            
        Returns:
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
        
        if generate_toy_data:
            self.benchmark_with_toy_data(algorithms, parameters_iterator, n_executions, datasets_folder)
        else:
            self.benchmark_with_given_data(algorithms, parameters_iterator, n_executions, datasets_folder)

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
                # Set dataset_iteration as the first column
                df.insert(0, 'dataset_iteration', df.pop('dataset_iteration'))
                               
                df.to_csv(f'{self.results_folder}/results_{name}.csv', index=False)
        else:
            # If it does exist, delete previous results
            for filename in os.listdir(self.results_folder):
                if filename.endswith('.csv'):
                    os.remove(f'{self.results_folder}/{filename}')
    
    def benchmark_with_toy_data(self, algorithms: dict[str, type[CausalDiscoveryBase]],
                                         parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]],
                                         n_executions: int,
                                         datasets_folder: str,
                                        )  -> dict[str, list[ dict[str, Any] ]]:
        # Delete previous toy data
        if os.path.exists(datasets_folder):
            for filename in os.listdir(datasets_folder):
                os.remove(f'{datasets_folder}/{filename}')
        else:
            os.makedirs(datasets_folder)

        for iteration, current_parameters in enumerate(parameters_iterator):
            current_algorithms_parameters, data_option = current_parameters
            causal_datasets = self.generate_datasets(iteration, n_executions, datasets_folder, data_option)
            
            if self.verbose > 0:
                print('\n' + '-'*50)
                print(BLUE, 'Executing the datasets with option:', data_option, RESET)
            
            # Generate and save results of all algorithms with current dataset options
            current_results = self.test_algorithms(causal_datasets, algorithms,
                                                   current_algorithms_parameters)
            
            for name, algorithm_results in current_results.items():
                for particular_result in algorithm_results:
                    particular_result.update(data_option) # Include the parameters in the information for results
                    particular_result['dataset_iteration'] = iteration
                
                    # Include current result in the list of result
                    self.results[name].append(particular_result)
                
                    self.all_algorithms_parameters[name].\
                                append(copy.deepcopy(current_algorithms_parameters[name]))
                
            self.save_results()
            if self.verbose > 0:
                print(f'{iteration+1} combinations executed')
            
        return self.results
    
    def benchmark_with_given_data(self, algorithms: dict[str, type[GroupCausalDiscoveryBase]],
                                        parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]],
                                        n_executions_per_data_param: int,
                                        datasets_folder: str,
                                        )  -> dict[str, list[ dict[str, Any] ]]:
        causal_datasets = self.load_datasets(datasets_folder)
        
        # Execute the algorithms with the given datasets
        for current_algorithms_parameters, data_option in parameters_iterator:
            if self.verbose > 0:
                print('\n' + '-'*50)
                print(BLUE, 'Datasets have been loaded.', RESET)
            
            # Generate and save results of all algorithms with given datasets
            current_results = self.test_algorithms(causal_datasets, algorithms,
                                                    current_algorithms_parameters)
            
            for name, algorithm_results in current_results.items():
                iteration = -1
                for particular_result in algorithm_results:
                    particular_result.update(data_option) # Include the parameters in the information for results
                    particular_result['dataset_iteration'] = (iteration:=iteration+1) // n_executions_per_data_param

                    # Include current result in the list of result
                    self.results[name].append(particular_result)
            
                    self.all_algorithms_parameters[name].\
                                append(copy.deepcopy(current_algorithms_parameters[name]))
            
            self.save_results()
            
        return self.results
    
    def test_algorithms(self, causal_datasets: list[CausalDataset],
                            algorithms: dict[str, type[CausalDiscoveryBase]],
                            algorithms_parameters: dict[str, type[CausalDiscoveryBase]],
                            ) -> dict[str, dict[str, list[Any]]]:
        '''
        Execute the given algorithms and return the results
        '''
        result = dict() # keys are score names and values are the score values
        for name, algorithm in algorithms.items():
            algorithm_results = self.test_particular_algorithm(algorithm_name=name,
                                    causal_datasets=causal_datasets, causalDiscovery=algorithm,
                                    algorithm_parameters=algorithms_parameters[name])
            result[name] = algorithm_results

        return result
    
    def test_particular_algorithm(self, algorithm_name: str,
                                causal_datasets: list[CausalDataset],
                                causalDiscovery: type[CausalDiscoveryBase],
                                algorithm_parameters: dict[str, Any]) -> dict[str, list[Any]]:
        '''
        Execute the given algorithm n_executions times and return the average and std of the results
        '''
        if len(causal_datasets) <= 0:
            return []
        
        if self.verbose > 0:
            print(GREEN, 'Executing algorithm', algorithm_name, RESET)
        # Execute the algorithm n_executions times
        results_per_execution = []
        for causal_dataset in tqdm(  causal_datasets ): # tqdm is used to show a progress bar
            results_per_execution.append( self.test_particular_algorithm_particular_dataset(causal_dataset, causalDiscovery, algorithm_parameters) )
            
        if self.verbose > 1:
            print(f'{results_per_execution=}')
            
        return results_per_execution
        

    
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
            plt.close('all')
            
            # Plot the graph structure
            plot_ts_graph(parents_dict)
            plt.savefig(f'{folder_name}/{data_name}_graph.pdf')
            
            # Plot the summary graph structure
            summary_parents = window_to_summary_graph(parents_dict)
            # Make the graph more beautiful by setting parents as past variables
            for son, parents in summary_parents.items():
                summary_parents[son] = [(p, -1) for p in parents]
            plot_ts_graph(summary_parents)
            plt.savefig(f'{folder_name}/{data_name}_summary_graph.pdf')
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
                datasets_groups = [df_results[df_results['dataset_iteration']==i] \
                                    for i in df_results['dataset_iteration'].unique()]
                x = [group[x_axis].mean() for group in datasets_groups]
                std = [group[score].std() for group in datasets_groups]
                y = [group[score].mean() for group in datasets_groups]
                ax.errorbar(x, y, yerr=std, label=algorithm_name,
                             fmt='.-', linewidth=1, capsize=3)
                ax.grid(axis='y')
            ax.set_xlabel(x_axis)
            ax.set_ylabel(score)
            ax.legend()
            
            plt.savefig(f'{results_folder}/plot_{score}.pdf')
            fig.clf(); plt.close('all') # Clear the figure and close it

    def plot_particular_result(self, results_folder,
                                     output_folder=None,
                                     scores=['shd', 'f1', 'precision', 'recall', 'time', 'memory'],
                                     dataset_iteration_to_plot=0):
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
            all_results = []
            for algorithm_name, df_results in results_dataframes.items():
                if dataset_iteration_to_plot == -1: # Get the results for the last dataset iteration
                    dataset_iteration_to_plot = df_results['dataset_iteration'].max()
                    
                current_dataset_results = df_results.loc[df_results['dataset_iteration'] == dataset_iteration_to_plot].copy()
                current_dataset_results['algorithm'] = algorithm_name
                all_results.append(current_dataset_results)
            
            all_results_df = pd.concat(all_results)
            sns.violinplot(x='algorithm', y=score, data=all_results_df,
                           hue='algorithm',ax=ax)
            ax.grid(axis='y')
            
            
            if score in ['f1', 'precision', 'recall', 
                    'f1_summary', 'precision_summary', 'recall_summary']:
                ax.set_ylim(0, 1)
            if score in ['shd', 'shd_summary']:
                ax.set_ylim(bottom=0, top=None)
                
            algorithms_names = list(results_dataframes.keys())
            ax.set_xticks(range(len(algorithms_names)), algorithms_names)
            ax.set_ylabel(score)
            
            plt.savefig(f'{output_folder}/comparison_{score}.pdf')
            plt.close('all')

class BenchmarkCausalDiscovery(BenchmarkBase):
    def __init__(self):
        self.verbose = 0
        self.results = None
    
    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        Function to generate the datasets for the benchmark
        
        Args:
            n_datasets : int The number of datasets to be generated
            datasets_folder : str The folder in which the datasets will be saved
            data_option : dict[str, Any] The options to generate the datasets
        
        Returns:
            causal_datasets : list[CausalDataset] The list with the datasets
        '''
        
        if self.verbose > 0:
            print('Generating datasets...')
        # Generate the datasets, with their graph structure and time series
        causal_datasets = [CausalDataset() for _ in range(n_datasets)]
        for current_dataset_index, causal_dataset in enumerate(causal_datasets):
            dataset_index = iteration * n_datasets + current_dataset_index
            causal_dataset.generate_toy_data(dataset_index, datasets_folder=datasets_folder, **data_option)
        
        return causal_datasets
    
    def load_datasets(self, datasets_folder):
        '''
        Function to load the datasets for the benchmark
        
        Args:
            datasets_folder : str The folder in which the datasets are saved
        '''
        causal_datasets = []
        # Obtain datasets from folders acording to their filename: {number}_data.csv
        if os.path.exists(datasets_folder):
            for filename in sorted(os.listdir(datasets_folder), key = lambda x: int(x.split('_')[0])):
                if filename.endswith('.csv'):
                    dataset = pd.read_csv(f'{datasets_folder}/{filename}')
                    parents_filename = f'{datasets_folder}/{filename.split("_")[0]}_parents.txt'
                    with open(parents_filename, 'r') as f:
                        parents_dict = eval(f.read())
                    
                    causal_datasets.append(CausalDataset(time_series=dataset.values,
                                                            parents_dict=parents_dict))
        else:
            raise ValueError(f'The dataset folder {datasets_folder} does not exist')
        
        return causal_datasets
    
    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                      causalDiscovery: type[CausalDiscoveryBase],
                      algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
        '''
        Execute the algorithm one single time and calculate the necessary scores.
        
        Args:
            causal_dataset : CausalDataset with the time series and the parents
            causalDiscovery : class of the algorithm to be executed
            algorithm_parameters : dictionary with the parameters for the algorithm
        Returns:
            result : dictionary with the scores of the algorithm
        '''
        
        algorithm = causalDiscovery(data=causal_dataset.time_series, **algorithm_parameters)
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
            actual_parents = causal_dataset.parents_dict
            actual_parents_summary = window_to_summary_graph(actual_parents)
            
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


class BenchmarkGroupCausalDiscovery(BenchmarkCausalDiscovery):        
    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        Function to generate the datasets for the benchmark
        
        Args:
            n_datasets : int The number of datasets to be generated
            datasets_folder : str The folder in which the datasets will be saved
            data_option : dict[str, Any] The options to generate the datasets
        '''
        if self.verbose > 0:
            print('Generating datasets...')
        return _generate_group_dataset(iteration, n_datasets, datasets_folder, data_option)
    
    def load_datasets(self, datasets_folder):
        '''
        Function to load the datasets for the benchmark
        
        Args:
            datasets_folder : str The folder in which the datasets are saved
        '''
        return _load_group_datasets(datasets_folder)
        
    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                      causalDiscovery: type[GroupCausalDiscoveryBase],
                      algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
        '''
        Execute the algorithm one single time and calculate the necessary scores.
        
        Args:
            causal_dataset : CausalDataset with the time series and the parents
            causalDiscovery : class of the algorithm to be executed
            algorithm_parameters : dictionary with the parameters for the algorithm
        Returns:
            result : dictionary with the scores of the algorithm
        '''
        algorithm = causalDiscovery(data=causal_dataset.time_series, groups=causal_dataset.groups,  **algorithm_parameters)
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
            actual_parents = causal_dataset.parents_dict
            actual_parents_summary = window_to_summary_graph(actual_parents)
            
            result['TP'] = get_TP(actual_parents, predicted_parents)
            result['FP'] = get_FP(actual_parents, predicted_parents)
            result['FN'] = get_FN(actual_parents, predicted_parents)
            result['precision'] = get_precision(actual_parents, predicted_parents)
            result['recall'] = get_recall(actual_parents, predicted_parents)
            result['f1'] = get_f1(actual_parents, predicted_parents)
            result['shd'] = get_shd(actual_parents, predicted_parents)
            
            # Obtain the same metrics in the summary graph
            predicted_parents_summary = window_to_summary_graph(predicted_parents)
            result['TP_summary'] = get_TP(actual_parents_summary, predicted_parents_summary)
            result['FP_summary'] = get_FP(actual_parents_summary, predicted_parents_summary)
            result['FN_summary'] = get_FN(actual_parents_summary, predicted_parents_summary)
            result['precision_summary'] = get_precision(actual_parents_summary, predicted_parents_summary)
            result['recall_summary'] = get_recall(actual_parents_summary, predicted_parents_summary)
            result['f1_summary'] = get_f1(actual_parents_summary, predicted_parents_summary)
            result['shd_summary'] = get_shd(actual_parents_summary, predicted_parents_summary)
            
            return result
        

class BenchmarkGroupsExtraction(BenchmarkBase):
    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        Function to generate the datasets for the benchmark
        
        Args:
            n_datasets : int The number of datasets to be generated
            datasets_folder : str The folder in which the datasets will be saved
            data_option : dict[str, Any] The options to generate the datasets
        '''
        if self.verbose > 0:
            print('Generating datasets...')
        return _generate_group_dataset(iteration, n_datasets, datasets_folder, data_option)
    
    def load_datasets(self, datasets_folder):
        '''
        Function to load the datasets for the benchmark
        
        Args:
            datasets_folder : str The folder in which the datasets are saved
        '''
        return _load_group_datasets(datasets_folder)

    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                      causalExtraction: type[CausalGroupsExtractorBase],
                      algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
        '''
        Execute the algorithm one single time and calculate the necessary scores.
        
        Args:
            causal_dataset : CausalDataset with the time series and the parents
            causalDiscovery : class of the algorithm to be executed
            algorithm_parameters : dictionary with the parameters for the algorithm
        Returns:
            result : dictionary with the scores of the algorithm
        '''
        algorithm = causalExtraction(data=causal_dataset.time_series, **algorithm_parameters)
        try:
            predicted_groups, time, memory = algorithm.extract_groups_time_and_memory()
        except Exception as e:
            print(f'Error in algorithm {causalExtraction.__name__}: {e.with_traceback()}')
            print('Returning nan values for this algorithm')
            predicted_groups = [set(range(causal_dataset.time_series.shape[1]))]
            time = np.nan
            memory = np.nan
        finally:
            result = {'time': time, 'memory': memory}
            actual_groups = causal_dataset.groups
            
            result['predicted_groups'] = predicted_groups
            result['actual_groups'] = actual_groups
            
            result['average_explained_variance'] = get_average_pc1_explained_variance(causal_dataset.time_series, predicted_groups)
            result['n_groups'] = len(predicted_groups)
            result['explainability_score'] = get_explainability_score(causal_dataset.time_series, predicted_groups)
            result['NMI'] = get_normalized_mutual_information(predicted_groups, actual_groups)
            
            return result
        


# INNER AUXILIAR FUNCTIONS
def _generate_group_dataset(iteration, n_datasets, datasets_folder, data_option):
    '''
    Function to generate the datasets for the benchmark
    
    Args:
        n_datasets : int The number of datasets to be generated
        datasets_folder : str The folder in which the datasets will be saved
        data_option : dict[str, Any] The options to generate the datasets
    '''
    causal_datasets = [CausalDataset() for _ in range(n_datasets)]
    for current_dataset_index, causal_dataset in enumerate(causal_datasets):
        dataset_index = iteration * n_datasets + current_dataset_index
        causal_dataset.generate_group_toy_data(dataset_index, datasets_folder=datasets_folder, **data_option)
    
    return causal_datasets

def _load_group_datasets(datasets_folder):
        '''
        Function to load the datasets for the benchmark
        
        Args:
            datasets_folder : str The folder in which the datasets are saved
        '''
        causal_datasets = []
        # Obtain datasets from folders acording to their filename: {number}_data.csv
        if os.path.exists(datasets_folder):
            for filename in sorted(os.listdir(datasets_folder), key = lambda x: int(x.split('_')[0])):
                if filename.endswith('.csv'):
                    dataset = pd.read_csv(f'{datasets_folder}/{filename}')
                    parents_filename = f'{datasets_folder}/{filename.split("_")[0]}_parents.txt'
                    with open(parents_filename, 'r') as f:
                        parents_dict = eval(f.read())
                    groups_filename = f'{datasets_folder}/{filename.split("_")[0]}_groups.txt'
                    with open(groups_filename, 'r') as f:
                        groups = eval(f.read())
                    
                    causal_datasets.append(CausalDataset(time_series=dataset.values,
                                                            parents_dict=parents_dict,
                                                            groups=groups))
        else:
            raise ValueError(f'The dataset folder {datasets_folder} does not exist')
        
        return causal_datasets
