import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
from mlops.api import app

def test_health_check():
    """M24: Testing functionality of health endpoint"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_prediction():
    """M24: Testing integration of model inference with a dummy image"""
    with TestClient(app) as client:
        # 1. Create a dummy image in memory
        file_name = "test_image.jpg"
        image = Image.new("RGB", (224, 224), color="red")
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # 2. Post to the /predict endpoint
        response = client.post(
            "/predict",
            files={"file": (file_name, img_byte_arr, "image/jpeg")}
        )

        # 3. Assertions
        assert response.status_code == 200
        data = response.json()
        assert "emotion" in data
        assert "confidence" in data
        assert isinstance(data["emotion"], str)
        assert 0.0 <= data["confidence"] <= 1.0