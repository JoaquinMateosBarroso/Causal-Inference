import json
from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.Algorithms.time_series import algorithms_parameters as algs_params_cd_from_ts
from app.Algorithms.time_series import data_generation_options
from app.Algorithms.time_series import runCausalDiscoveryFromTimeSeries, generateDataset

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
async def read_ts_causal_discovery(request: Request,):
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
    


'''
Functions for the Causal Discovery from Time Series
'''
@app.get("/ts-causal-discovery/")
@app.get("/ts-causal-discovery/{algorithm}")
async def read_ts_causal_discovery(request: Request,
                                     chosen_algorithm: str='pcmci'):
    return templates.TemplateResponse("ts-causal-discovery.jinja",
                                {'request': request,
                                 'algs_params': algs_params_cd_from_ts,
                                 'chosen_algorithm': chosen_algorithm})

@app.put("/ts-causal-discovery/{algorithm}")
async def run_ts_causal_discovery(algorithm: str,
                                    algorithm_parameters_str: str = Form(...),
                                    datasetFile: UploadFile = File(...)):
    algorithm_parameters = json.loads(algorithm_parameters_str)
    return runCausalDiscoveryFromTimeSeries(algorithm, algorithm_parameters, datasetFile)



'''
Functions for the Benchmarking of Causal Discovery from Time Series
'''
@app.get("/benchmark-ts-causal-discovery")
async def read_benchmark_causal_discovery(request: Request,
                                     chosen_algorithm: str='pcmci'):
    return templates.TemplateResponse("benchmark-ts-causal-discovery.jinja",
                                {'request': request,
                                 'algs_params': algs_params_cd_from_ts,
                                 'chosen_algorithm': chosen_algorithm})

@app.put("/benchmark-ts-causal-discovery")
async def run_benchmark_causal_discovery(algorithms_parameters_str: str = Form(...),
                                        datasetFile: UploadFile = File(...)):
    algorithms_parameters = json.loads(algorithms_parameters_str)
    aux_folder_name = str(uuid.uuid4())

    return await runTimeSeriesBenchmarkFromZip(algorithms_parameters, datasetFile, aux_folder_name)
