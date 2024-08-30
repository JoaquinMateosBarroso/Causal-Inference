from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
favicon_path = "frontend/static/favicon.ico"

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/")
async def readIndex(request: Request):
    response = templates.TemplateResponse("index.html", {"request": request})
    return response

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


@app.get("/basic-pc")
async def readBasicPCIndex(request: Request):
    response = FileResponse("frontend/static/basic-pc/index.html")
    return response

from app.parametersDefinition import BasicPC_parameters
from typing import List
from app.algorithms.basicPC import callBasicPC  
@app.put("/basic-pc")
async def executeBasicPC(request: Request,
                defaultFeatures: str,
                endogeneousFeatures: str,
                exogeneousFeatures: str,
                datasetFile: UploadFile = File(...)):
    
    
    pcParameters = BasicPC_parameters(
            # I use the list comprehension to avoid empty strings
            defaultFeatures=[feature for feature in defaultFeatures.split(",") if feature],
            endogeneousFeatures=[feature for feature in endogeneousFeatures.split(",") if feature],
            exogeneousFeatures=[feature for feature in exogeneousFeatures.split(",") if feature])
    
    
    
    response = await callBasicPC(datasetFile, pcParameters)
    return response
        


