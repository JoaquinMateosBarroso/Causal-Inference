from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.Algorithms.timeSeries import causal_discovery_from_time_series_algorithms, runCausalDiscoveryFromTimeSeries

app = FastAPI()
favicon_path = "frontend/static/favicon.ico"

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/")
async def readIndex(request: Request):
    response = templates.TemplateResponse("index.html",
                    {"request": request,
                    "causal_discovery_base_algorithms": ["basic-pc"],
                    "causal_discovery_from_time_series_algorithms": causal_discovery_from_time_series_algorithms})
    return response

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


@app.get("/causal-discovery-base/{algorithm}")
async def read_causal_discovery_base(algorithm: str,
                                     request: Request):
    response = templates.TemplateResponse("causal-discovery-base/index.html",
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

@app.get("/ts-causal-discovery/")
@app.get("/ts-causal-discovery/{algorithm}")
async def read_causal_discovery_base(request: Request,
                                     algorithm: str='pcmci'):
    response = templates.TemplateResponse("ts-causal-discovery/index.html",
                                {"request": request,
                                 'algorithms_names': causal_discovery_from_time_series_algorithms,
                                 "algorithm": algorithm})    
    return response

@app.put("/ts-causal-discovery/{algorithm}")
async def executeCusalDiscovery_TimeSeries(algorithm: str,
                                           datasetFile: UploadFile = File(...)):
    graph_image = runCausalDiscoveryFromTimeSeries(algorithm, datasetFile)

    return graph_image
    return Response(content=graph_image, media_type="image/png")


@app.get("/benchmark-ts-causal-discovery")
async def read_causal_discovery_compare_ts(request: Request):
    response = templates.TemplateResponse("causal-discovery-compare-ts/index.html",
                                {"request": request,
                                 "algorithms": causal_discovery_from_time_series_algorithms})
    
    return response
