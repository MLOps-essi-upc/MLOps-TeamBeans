import pytest
from fastapi.testclient import TestClient
import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app.api import app
from pathlib import Path

ROOT_DIR = Path(Path(__file__).resolve().parent)


def test_make_prediction_endpoint():
    with TestClient(app) as client:
        # Replace 'path/to/your/image.jpg' with the actual path to your test image
        image_path = ROOT_DIR / 'test_image_1.jpg'

        # Open the image file in binary mode
        with open(image_path, 'rb') as img_file:
            # Create a file-like object from the image file
            img_file_object = io.BytesIO(img_file.read())

        files = {'beans_img': ('test_image_1.jpg', img_file_object, 'image/jpeg')}
        response = client.post("/make_prediction", files=files)

        assert response.status_code == 200
        assert set(response.json().keys()) == {'prediction', 'probs'}
