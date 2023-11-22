from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch
from .. import MODELS_DIR
from ..models.model import SimpleCNNReducedStride10 

from pydantic import BaseModel

model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title="Team Beans",
    description="This API let you make predictions on beans pics.",
    version="0.1",
)

def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "MODELS_DIR": results["MODELS_DIR"],
            "model":results["model"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.on_event("startup")
def _load_models():

    # Initialize the pytorch model
    model=SimpleCNNReducedStride10()

    model_paths = [
        filename
        for filename in MODELS_DIR.iterdir()
        if filename.suffix == ".pt" and filename.stem.startswith("trained_model")
    ]

    for path in model_paths:
        with open(path, "rb") as file:
            model.load_state_dict(torch.load(file))
            model.eval()
    
    # add model and other preprocess tools too app state
    app.package = {
        #"scaler": load(CONFIG['SCALAR_PATH']),  # joblib.load
        "model": model
    }

@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Teams Beans classifier! Please, read the `/docs`!"},
        "MODELS_DIR":{"MODELS_DIR":MODELS_DIR},
        "model":{"model":app.package},
    }
    return response



# @app.get("/models", tags=["Prediction"])
# @construct_response