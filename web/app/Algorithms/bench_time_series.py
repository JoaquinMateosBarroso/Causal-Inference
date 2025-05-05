import ast
import base64
import io
from typing import Any, Iterator, Union
import zipfile

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from matplotlib import pyplot as plt
import numpy as np

from app.Algorithms.time_series import ts_algorithms, ts_algorithms_parameters
from app.Algorithms.time_series import group_ts_algorithms, group_ts_algorithms_parameters
from app.Algorithms.time_series import data_generation_options, group_data_generation_options

from group_causation.benchmark import BenchmarkCausalDiscovery, BenchmarkGroupCausalDiscovery

import shutil
import os

# Ignore FutureWarnings, due to versions of libraries
import warnings

import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)

from group_causation.utils import changing_N_variables, changing_preselection_alpha, static_parameters


benchmark_options = {
    'changing_N_variables': (changing_N_variables, 
                                    {'list_N_variables': [5]}),
    
    'changing_preselection_alpha': (changing_preselection_alpha,
                                    {'list_preselection_alpha': [0.01, 0.05, 0.1, 0.2]}),
    'static': (static_parameters, {}),
}
chosen_option = 'static'


async def runTimeSeriesBenchmarkFromZip(algorithms_parameters: list[dict[str, Any]],
                                  datasetsFile: UploadFile,
                                  aux_folder_name,
                                  group_type: bool = False):
    plt.style.use('ggplot')
    if group_type:
        benchmark = BenchmarkGroupCausalDiscovery()
        algorithms = group_ts_algorithms
    else:
        benchmark = BenchmarkCausalDiscovery()
        algorithms = ts_algorithms

    # Convert the string representations of the parameters to their actual types
    complex_parameters = ['dimensionality_reduction_params', 'node_causal_discovery_params']
    for algorithm_parameters in algorithms_parameters.values():
        for parameter in complex_parameters:
            if parameter in algorithm_parameters:
                algorithm_parameters[parameter] = ast.literal_eval(algorithm_parameters[parameter])
    
    options_generator, options_kwargs = benchmark_options[chosen_option]
    parameters_iterator = options_generator(data_generation_options,
                                                algorithms_parameters,
                                                **options_kwargs)
    
    datasets_folder = f'toy_data/{aux_folder_name}'
    results_folder = f'results/{aux_folder_name}'
    
    # Delete previous toy data
    if os.path.exists(datasets_folder):
        for filename in os.listdir(datasets_folder):
            os.remove(f'{datasets_folder}/{filename}')
    else:
        os.makedirs(datasets_folder)
    
    # Unzip the datasets
    dataset_contents = await datasetsFile.read()
    with zipfile.ZipFile(io.BytesIO(dataset_contents), "r") as zip_ref:
        zip_ref.extractall(datasets_folder)
    
    # Delete previous results
    if os.path.exists(results_folder):
        for filename in os.listdir(results_folder):
            os.remove(f'{results_folder}/{filename}')
    else:
        os.makedirs(results_folder)
    
    chosen_algorithms = {f'{algorithm}': algorithms[algorithm] \
                                    for algorithm in algorithms_parameters.keys()}
    await run_in_threadpool(benchmark.benchmark_causal_discovery,
                                        algorithms=chosen_algorithms,
                                        parameters_iterator=parameters_iterator,
                                        datasets_folder=datasets_folder,
                                        results_folder=results_folder,
                                        verbose=1)
    
    files = os.listdir(results_folder)
    # Save results for whole graph scores
    benchmark.plot_particular_result(results_folder)
    # Save results for summary graph scores
    benchmark.plot_particular_result(results_folder, results_folder + '/summary',
                                     scores=[f'{score}_summary' for score in \
                                                    ['shd', 'f1', 'precision', 'recall']],
                                     dataset_iteration_to_plot=0)
    
    # Get all necessary files, including those in subfolders
    files_data = []
    for root, _, files in os.walk(results_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, results_folder)  # Preserve folder structure in filenames
            with open(file_path, "rb") as f:
                files_data.append({
                    "filename": relative_path,  # Use relative path instead of just filename
                    "content": base64.b64encode(f.read()).decode("utf-8")})
                
    # Clean folders
    shutil.rmtree(datasets_folder)
    shutil.rmtree(results_folder)

    return JSONResponse(content={"files": files_data})
