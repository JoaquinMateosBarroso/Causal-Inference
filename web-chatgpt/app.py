from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import os
import shutil
import json
import io
import base64
import matplotlib.pyplot as plt

# === IMPORTS FROM THE ORIGINAL CODE ===
from benchmark_causal_discovery import BenchmarkCausalDiscovery
from causal_discovery_tigramite import PCMCIModifiedWrapper, PCMCIWrapper, LPCMCIWrapper, PCStableWrapper
from causal_discovery_algorithms.causal_discovery_causalai import GrangerWrapper, VARLINGAMWrapper
from causal_discovery_algorithms.causal_discovery_causalnex import DynotearsWrapper
from functions_test_data import changing_N_variables, changing_preselection_alpha

# === Global definitions as in the original script ===
algorithms = {
    'pcmci-modified': PCMCIModifiedWrapper,
    'pcmci': PCMCIWrapper,
    'dynotears': DynotearsWrapper,
    'granger': GrangerWrapper,
    'varlingam': VARLINGAMWrapper,
    'pc-stable': PCStableWrapper,
    
    'fullpcmci': PCMCIWrapper,
    'fastpcmci': PCMCIWrapper,
    'lpcmci': LPCMCIWrapper,
}

benchmark_options = {
    'changing_N_variables': changing_N_variables,
    'changing_preselection_alpha': changing_preselection_alpha,
}

def generate_parameters_iterator(data_gen_opts: Dict[str, Any],
                                 algo_params: Dict[str, Any],
                                 benchmark_key: str):
    """
    Generates an iterator over (data_generation_options, algorithms_parameters)
    based on the selected benchmark option.
    """
    benchmark_func = benchmark_options.get(benchmark_key)
    for data_generation_options, algorithms_parameters in benchmark_func(data_gen_opts, algo_params):
        yield data_generation_options, algorithms_parameters

# === Default Parameters (can be adjusted via the frontend) ===
default_algorithms_parameters = {
    'pcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'cond_ind_test': 'parcorr'},
    'granger': {'cv': 5, 'min_lag': 1, 'max_lag': 3},
    'varlingam': {'min_lag': 1, 'max_lag': 3},
    'dynotears': {'max_lag': 3, 'max_iter': 10000},
    'pc-stable': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
    'pcmci-modified': {'pc_alpha': 0.01, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 1, 'max_conds_dim': 5,
                       'max_crosslink_density': 0.5, 'preselection_alpha': 0.01},
    'fullpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 100, 'max_conds_dim': 5},
    'fastpcmci': {'pc_alpha': None, 'min_lag': 1, 'max_lag': 3, 'max_combinations': 1, 'max_conds_dim': 3},
    'lpcmci': {'pc_alpha': 0.05, 'min_lag': 1, 'max_lag': 3},
}

default_data_generation_options = {
    'max_lag': 5,
    'crosslinks_density': 0.75,
    'T': 500,
    'N': 20,
    'dependency_coeffs': [-0.4, 0.4],
    'auto_coeffs': [0.7],
    'noise_dists': ['gaussian'],
    'noise_sigmas': [0.2],
    # Note: For functions it is easier to pass their string representations,
    # then later (if needed) eval them in a controlled environment.
    'dependency_funcs': [
        "lambda x: 0.5*x",
        "lambda x: np.exp(-abs(x)) - 1 + np.tanh(x)",
        "lambda x: np.sin(x)",
        "lambda x: np.cos(x)",
        "lambda x: 1 if x > 0 else 0"
    ],
}

# === Pydantic model for the benchmark input parameters ===
class BenchmarkParameters(BaseModel):
    selected_benchmark: str
    algorithms_parameters: Optional[Dict[str, Any]] = None
    data_generation_options: Optional[Dict[str, Any]] = None
    n_executions: Optional[int] = 3
    scores: Optional[List[str]] = ["f1", "precision", "recall", "time", "memory"]
    verbose: Optional[int] = 1

# === Initialize FastAPI, templates and static file serving ===
app = FastAPI()
templates = Jinja2Templates(directory="/home/joaquin/Documents/Asignaturas/TFG UCO/Introduction-to-Causal-Inference/web-chatgpt/frontend/templates/")
app.mount("/static", StaticFiles(directory="/home/joaquin/Documents/Asignaturas/TFG UCO/Introduction-to-Causal-Inference/web-chatgpt/static"), name="static")

# === Utility function to capture a matplotlib plot as Base64 string ===
def plot_to_base64(plot_func, *args, **kwargs):
    fig = plt.figure()
    plot_func(*args, **kwargs)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# === GET endpoint for the frontend page ===
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_algorithms_parameters": json.dumps(default_algorithms_parameters, indent=4),
        "default_data_generation_options": json.dumps(default_data_generation_options, indent=4),
        "benchmark_options": list(benchmark_options.keys())
    })

# === POST endpoint to run the benchmark and return the plots ===
@app.post("/run_benchmark")
async def run_benchmark(params: BenchmarkParameters):
    # Use user-provided parameters or fall back to defaults.
    algo_params = params.algorithms_parameters or default_algorithms_parameters
    data_gen_opts = params.data_generation_options or default_data_generation_options
    selected_benchmark = params.selected_benchmark

    # Create a BenchmarkCausalDiscovery instance
    benchmark = BenchmarkCausalDiscovery()
    datasets_folder = "toy_data"
    results_folder = "results"
    
    # Clean previous results if they exist
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder, exist_ok=True)
    
    # Run the benchmark using the provided parameters.
    parameters_iterator = generate_parameters_iterator(data_gen_opts, algo_params, selected_benchmark)
    results = benchmark.benchmark_causal_discovery(
        algorithms=algorithms,
        parameters_iterator=parameters_iterator,
        datasets_folder=datasets_folder,
        results_folder=results_folder,
        n_executions=params.n_executions,
        scores=params.scores,
        verbose=params.verbose
    )
    
    # Generate plots and convert them to Base64 strings.
    ts_plot = plot_to_base64(benchmark.plot_ts_datasets, datasets_folder)
    moving_plot = plot_to_base64(benchmark.plot_moving_results, results_folder, x_axis='preselection_alpha')
    particular_plot = plot_to_base64(benchmark.plot_particular_result, results_folder)
    
    # Optionally, copy the toy_data folder inside the results folder.
    destination_folder = os.path.join(results_folder, datasets_folder)
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(datasets_folder, destination_folder)
    
    return JSONResponse({
        "ts_datasets": ts_plot,
        "moving_results": moving_plot,
        "particular_result": particular_plot
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
