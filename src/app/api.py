from http import HTTPStatus
from typing import List
from PIL import Image
from io import BytesIO
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from torchvision import transforms
import torch
from src import MODELS_DIR
from src.models.model import SimpleCNNReducedStride10
from pydantic import BaseModel, ValidationError


model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title="Team Beans",
    description="This API let you make predictions on beans pics.",
    version="0.1",
)

class BeansImageInfo(BaseModel):
    filename: str
    content_type: str
    width: int
    height: int


@app.on_event("startup")
def _load_models_and_constants():

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

    predict_dict = {
        0: "angular_leaf_spot",
        1: "bean_rust",
        2: "healthy"
    }

    allowed_content_types = ["image/jpeg", "image/jpg", "image/png"]

    # add model and other preprocess tools too app state
    app.package = {
        "model": model,
        "pred_dict": predict_dict,
        'allowed_content_types': allowed_content_types
    }

@app.get("/", tags=["General"])  # path operation decorator
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Teams Beans classifier! Please, read the `/docs`!"},
        "MODELS_DIR":{"MODELS_DIR":MODELS_DIR},
        "model":{"model":app.package['model']},
    }
    return response



# curl -X POST -H "Content-Type: multipart/form-data" -H "Accept: application/json"
# -F "beans_img=@/Users/wwoszczek/Desktop/test_beans_img/0.jpg"
# http://localhost:8000/make_prediction

# API call only for 500x500 pictures
@app.post("/make_prediction_strict", tags=["Prediction"])
async def _index(beans_img: UploadFile = File(...)):
    try:
        img_info = BeansImageInfo(
            filename=beans_img.filename,
            content_type=beans_img.content_type,
            width=0,
            height=0)

        if img_info.content_type not in app.package['allowed_content_types']:
            raise HTTPException(status_code=415, detail="Unsupported media type. Please upload JPG or PNG file.")

        img_bytes = await beans_img.read()
        img_raw = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_info.width, img_info.height = img_raw.size

    except ValidationError as e:
        raise HTTPException(status_code=415, detail=f"Invalid request: {e}")


    if img_info.height != 500 or img_info.width != 500:
        raise HTTPException(status_code=400, detail="Image dimensions are not equal to 500x500. "
                                                    "Please upload appropriate file.")

    model = app.package['model']

    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize RGB channels
    ])

    image = preprocess(img_raw).unsqueeze(0)

    output = torch.exp(model(image))

    pred = torch.argmax(output, 1).item()
    label = app.package['pred_dict'][pred]

    return {
        "probs": output.tolist(),
        "prediction": label
    }

# API calls for images of any size
@app.post("/make_prediction", tags=["Prediction"])
async def _index(beans_img: UploadFile = File(...)):
    try:
        img_info = BeansImageInfo(
            filename=beans_img.filename,
            content_type=beans_img.content_type,
            width=0,
            height=0)

        if img_info.content_type not in app.package['allowed_content_types']:
            raise HTTPException(status_code=415, detail="Unsupported media type. Please upload JPG or PNG file.")

        img_bytes = await beans_img.read()
        img_raw = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_info.width, img_info.height = img_raw.size

    except ValidationError as e:
        raise HTTPException(status_code=415, detail=f"Invalid request: {e}")


    model = app.package['model']

    preprocess = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize RGB channels
    ])

    image = preprocess(img_raw).unsqueeze(0)

    output = torch.exp(model(image))

    pred = torch.argmax(output, 1).item()
    label = app.package['pred_dict'][pred]

    return {
        "probs": output.tolist(),
        "prediction": label
    }




