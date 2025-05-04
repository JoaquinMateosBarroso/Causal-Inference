import json
from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.Algorithms.time_series import ts_algorithms_parameters
from app.Algorithms.time_series import group_ts_algorithms_parameters
from app.Algorithms.time_series import data_generation_options, group_data_generation_options
from app.Algorithms.time_series import runCausalDiscoveryFromTimeSeries, runGroupCausalDiscoveryFromTimeSeries, generateDataset, generateGroupDataset

import uuid

from app.Algorithms.bench_time_series import runTimeSeriesBenchmarkFromZip

# Create and mount the FastAPI app
app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

templates = Jinja2Templates(directory="frontend/templates")
favicon_path = "frontend/static/favicon.ico"


@app.get("/")
async def readIndex(request: Request):
    return templates.TemplateResponse("index.jinja",
                    {"request": request})
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


'''
Functions for the Dataset Creation
'''
@app.get("/create-toy-data/")
async def read_create_toy_data(request: Request,):
    return templates.TemplateResponse("create-toy-data.jinja",
                                {'request': request,
                                 'dataset_creation_params': data_generation_options})

@app.put("/create-toy-data/")
async def execute_create_toy_data(request: Request, dataset_parameters_str: str = Form(...)):
    dataset_parameters = json.loads(dataset_parameters_str)
    n_datasets = dataset_parameters.pop('n_datasets')
    aux_folder_name = str(uuid.uuid4())
    
    # Call the function to create the toy data
    return await generateDataset(dataset_parameters, n_datasets, aux_folder_name)

@app.get("/create-group-toy-data/")
async def read_create_toy_data_group(request: Request,):
    return templates.TemplateResponse("create-group-toy-data.jinja",
                                {'request': request,
                                 'dataset_creation_params': group_data_generation_options})

@app.put("/create-group-toy-data/")
async def execute_create_toy_data_group(dataset_parameters_str: str = Form(...)):
    dataset_parameters = json.loads(dataset_parameters_str)
    n_datasets = dataset_parameters.pop('n_datasets')
    aux_folder_name = str(uuid.uuid4())
    
    # Call the function to create the toy data
    return await generateGroupDataset(dataset_parameters, n_datasets, aux_folder_name)


'''
Functions for the Causal Discovery from Time Series
'''
@app.get("/ts-causal-discovery/")
@app.get("/ts-causal-discovery/{algorithm}")
async def read_ts_causal_discovery(request: Request,
                                     chosen_algorithm: str='pcmci'):
    return templates.TemplateResponse("ts-causal-discovery.jinja",
                                {'request': request,
                                 'algs_params': ts_algorithms_parameters,
                                 'chosen_algorithm': chosen_algorithm})

@app.put("/ts-causal-discovery/{algorithm}")
async def run_ts_causal_discovery(algorithm: str,
                                    algorithm_parameters_str: str = Form(...),
                                    datasetFile: UploadFile = File(...)):
    algorithm_parameters = json.loads(algorithm_parameters_str)
    return await run_in_threadpool(runCausalDiscoveryFromTimeSeries,
                                   algorithm, algorithm_parameters, datasetFile)


'''
Functions for the Causal Discovery from Groups of Time Series
'''
@app.get("/group-ts-causal-discovery/")
@app.get("/group-ts-causal-discovery/{algorithm}")
async def read_group_ts_causal_discovery(request: Request):
    return templates.TemplateResponse("group-ts-causal-discovery.jinja",
                                {'request': request,
                                 'algs_params': group_ts_algorithms_parameters})

@app.put("/group-ts-causal-discovery/{algorithm}")
async def run_group_ts_causal_discovery(request: Request, 
                                    algorithm: str,
                                    algorithm_parameters_str: str = Form(...),
                                    datasetFile: UploadFile = File(...)):
    algorithm_parameters = json.loads(algorithm_parameters_str)
    return await run_in_threadpool(runGroupCausalDiscoveryFromTimeSeries,
                                    algorithm, algorithm_parameters,
                                    datasetFile, request.query_params)
    
'''
Functions for the Benchmarking of Causal Discovery from Time Series
'''
@app.get("/benchmark-ts-causal-discovery")
async def read_benchmark_causal_discovery(request: Request,
                                     chosen_algorithm: str='pcmci'):
    return templates.TemplateResponse("benchmark-ts-causal-discovery.jinja",
                                {'request': request,
                                 'algs_params': ts_algorithms_parameters,
                                 'chosen_algorithm': chosen_algorithm})

@app.put("/benchmark-ts-causal-discovery")
async def run_benchmark_causal_discovery(algorithms_parameters_str: str = Form(...),
                                        datasetFile: UploadFile = File(...)):
    algorithms_parameters = json.loads(algorithms_parameters_str)
    aux_folder_name = str(uuid.uuid4())

    return await run_in_threadpool(runTimeSeriesBenchmarkFromZip,
                             algorithms_parameters, datasetFile, aux_folder_name)
