import json
from fastapi import FastAPI, Form, Request, Response, UploadFile, File
from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi_sessions.frontends.primitive import CookieParameters
from fastapi_sessions.backends.implicit import InMemoryBackend
import uuid


from app.Algorithms.time_series import algorithms_parameters as algs_params_cd_from_ts, generateDataset
from app.Algorithms.time_series import data_generation_options
from app.Algorithms.time_series import runCausalDiscoveryFromTimeSeries


# Create and mount the FastAPI app
app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

templates = Jinja2Templates(directory="frontend/templates")
favicon_path = "frontend/static/favicon.ico"

# Prepare the Session IDs
cookie_params = CookieParameters(secure=False, httponly=True, samesite="lax")
session_backend = InMemoryBackend()

@app.get("/")
async def readIndex(request: Request):
    response = templates.TemplateResponse("index.jinja",
                    {"request": request,
                    "causal_discovery_base_algorithms": ["basic-pc"],})

    return response

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


@app.get("/causal-discovery-base/{algorithm}")
async def read_causal_discovery_base(algorithm: str,
                                     request: Request):
    response = templates.TemplateResponse("causal-discovery-base/index.jinja",
                                {"request": request,
                                 "algorithm": algorithm})
    return response

from app.parametersDefinition import Parameters_CausalDiscoveryBase
from app.Algorithms.basicPC import callBasicPC  
@app.put("/basic-pc")
async def executeBasicPC(defaultFeatures: str,
                endogeneousFeatures: str,
                exogeneousFeatures: str,
                datasetFile: UploadFile = File(...)):
    
    pcParameters = Parameters_CausalDiscoveryBase(
            # I use the list comprehension to avoid empty strings
            defaultFeatures=[feature for feature in defaultFeatures.split(",") if feature],
            endogeneousFeatures=[feature for feature in endogeneousFeatures.split(",") if feature],
            exogeneousFeatures=[feature for feature in exogeneousFeatures.split(",") if feature])
    
    response = await callBasicPC(datasetFile, pcParameters)
    return response


@app.get("/create-toy-data/")
async def read_ts_causal_discovery(request: Request,):
    return templates.TemplateResponse("create-toy-data.jinja",
                                {'request': request,
                                 'dataset_creation_params': data_generation_options})

@app.put("/create-toy-data/")
async def execute_create_toy_data(request: Request, dataset_parameters_str: str = Form(...)):
    dataset_parameters = json.loads(dataset_parameters_str)
    print(f'{dataset_parameters=}')
    sesion_name = request.session
    
    # Call the function to create the toy data
    generateDataset(dataset_parameters)
    
    return {"message": "Toy data created successfully!"}


@app.get("/ts-causal-discovery/")
@app.get("/ts-causal-discovery/{algorithm}")
async def read_ts_causal_discovery(request: Request,
                                     chosen_algorithm: str='pcmci'):
    return templates.TemplateResponse("ts-causal-discovery.jinja",
                                {'request': request,
                                 'algs_params': algs_params_cd_from_ts,
                                 'chosen_algorithm': chosen_algorithm})

@app.put("/ts-causal-discovery/{algorithm}")
async def execute_ts_causal_discovery(algorithm: str,
                                        algorithm_parameters_str: str = Form(...),
                                        datasetFile: UploadFile = File(...)):
    algorithm_parameters = json.loads(algorithm_parameters_str)
    graph_image = runCausalDiscoveryFromTimeSeries(algorithm, algorithm_parameters, datasetFile)
    
    return graph_image


@app.get("/benchmark-ts-causal-discovery")
@app.get("/benchmark-ts-causal-discovery/{algorrithm}")
async def read_benchmark_causal_discovery_base(request: Request,
                                     chosen_algorithm: str='pcmci'):
    return templates.TemplateResponse("ts-causal-discovery.jinja",
                                {'request': request,
                                 'algs_params': algs_params_cd_from_ts,
                                 'chosen_algorithm': chosen_algorithm})
