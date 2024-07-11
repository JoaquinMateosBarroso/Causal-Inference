from fastapi import FastAPI, Request, UploadFile
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


from app.parametersDefinition import BasicPC_parameters
from app.algorithms.basicPC import callBasicPC
@app.get("/basic-pc")
async def readIndex(request: Request, 
                    dataFile: UploadFile | None = None, 
                    parameters: BasicPC_parameters | None = None):
    if dataFile is None: # No file uploaded
        response = FileResponse("frontend/static/basic-pc/index.html")
    else:
        response = callBasicPC(dataFile, parameters)
    return response


