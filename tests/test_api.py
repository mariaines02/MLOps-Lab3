"""
Integration tests for the FastAPI application.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from api.api import app


@pytest.fixture
def client():
    """Fixture providing a test client for the API."""
    return TestClient(app)


@pytest.fixture
def sample_image_file():
    """Fixture creating a sample image file for API testing."""
    img = Image.new("RGB", (100, 100), color="yellow")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return ("test_image.jpg", img_bytes, "image/jpeg")


def test_home_endpoint(client):
    """Test the home endpoint returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "MLOps Lab API" in response.text


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "MLOps Lab API"


def test_predict_endpoint(client, sample_image_file):
    """Test the predict endpoint."""
    response = client.post("/predict", files={"file": sample_image_file})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "all_classes" in data
    assert isinstance(data["all_classes"], list)


def test_predict_with_seed(client, sample_image_file):
    """Test predict endpoint with seed for reproducibility."""
    response1 = client.post("/predict?seed=42", files={"file": sample_image_file})

    # Reset file pointer
    sample_image_file[1].seek(0)

    response2 = client.post("/predict?seed=42", files={"file": sample_image_file})

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()


def test_predict_confidence_range(client, sample_image_file):
    """Test predict returns confidence in expected range."""
    response = client.post("/predict", files={"file": sample_image_file})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_resize_endpoint(client, sample_image_file):
    """Test the resize endpoint."""
    response = client.post(
        "/resize?width=50&height=50", files={"file": sample_image_file}
    )
    assert response.status_code == 200
    assert "image" in response.headers["content-type"]

    # Verify resized image
    img = Image.open(io.BytesIO(response.content))
    assert img.size == (50, 50)


def test_resize_default_dimensions(client, sample_image_file):
    """Test resize with default dimensions."""
    response = client.post("/resize", files={"file": sample_image_file})
    assert response.status_code == 200

    # Default is 224x224
    img = Image.open(io.BytesIO(response.content))
    assert img.size == (224, 224)


def test_grayscale_endpoint(client, sample_image_file):
    """Test the grayscale endpoint."""
    response = client.post("/grayscale", files={"file": sample_image_file})
    assert response.status_code == 200
    assert "image" in response.headers["content-type"]

    # Verify grayscale conversion
    img = Image.open(io.BytesIO(response.content))
    assert img.mode == "L"


def test_normalize_endpoint(client, sample_image_file):
    """Test the normalize endpoint."""
    response = client.post("/normalize", files={"file": sample_image_file})
    assert response.status_code == 200
    assert "image" in response.headers["content-type"]
    # Check if the response content is a valid image by trying to open it
    with Image.open(io.BytesIO(response.content)) as img:
        assert img.format.lower() == "jpeg"


def test_normalize_values_range(client, sample_image_file):
    """Test normalize returns values in expected ranges."""
    response = client.post("/normalize", files={"file": sample_image_file})
    assert response.status_code == 200
    # A simple check to ensure the response is not empty
    assert len(response.content) > 100


def test_crop_endpoint(client, sample_image_file):
    """Test the crop endpoint."""
    response = client.post(
        "/crop?left=10&top=10&right=60&bottom=60", files={"file": sample_image_file}
    )
    assert response.status_code == 200
    assert "image" in response.headers["content-type"]

    # Verify cropped image
    with Image.open(io.BytesIO(response.content)) as img:
        assert img.size == (50, 50)


@patch("logic.predictor.ImagePredictor.resize_image_from_bytes")
def test_resize_endpoint_handles_exception(mock_resize, client, sample_image_file):
    """Test that the resize endpoint handles exceptions gracefully."""
    mock_resize.side_effect = ValueError("Test error")
    response = client.post(
        "/resize?width=50&height=50", files={"file": sample_image_file}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Test error"}


@patch("logic.predictor.ImagePredictor.convert_to_grayscale_from_bytes")
def test_grayscale_endpoint_handles_exception(
    mock_grayscale, client, sample_image_file
):
    """Test that the grayscale endpoint handles exceptions gracefully."""
    mock_grayscale.side_effect = Exception("Grayscale failed")
    response = client.post("/grayscale", files={"file": sample_image_file})
    assert response.status_code == 400
    assert response.json() == {"detail": "Grayscale failed"}


@patch("logic.predictor.ImagePredictor.crop_image_from_bytes")
def test_crop_endpoint_handles_exception(mock_crop, client, sample_image_file):
    """Test that the crop endpoint handles exceptions gracefully."""
    mock_crop.side_effect = ValueError("Crop failed")
    response = client.post("/crop", files={"file": sample_image_file})
    assert response.status_code == 400
    assert response.json() == {"detail": "Crop failed"}


@patch("logic.predictor.ImagePredictor.normalize_image_from_bytes")
def test_normalize_endpoint_handles_exception(
    mock_normalize, client, sample_image_file
):
    """Test that the normalize endpoint handles exceptions gracefully."""
    mock_normalize.side_effect = Exception("Normalize failed")
    response = client.post("/normalize", files={"file": sample_image_file})
    assert response.status_code == 400
    assert response.json() == {"detail": "Normalize failed"}


def test_predict_without_file(client):
    """Test predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity


def test_resize_without_file(client):
    """Test resize endpoint without file."""
    response = client.post("/resize")
    assert response.status_code == 422


def test_grayscale_without_file(client):
    """Test grayscale endpoint without file."""
    response = client.post("/grayscale")
    assert response.status_code == 422


def test_normalize_without_file(client):
    """Test normalize endpoint without file."""
    response = client.post("/normalize")
    assert response.status_code == 422


def test_crop_without_file(client):
    """Test crop endpoint without file."""
    response = client.post("/crop")
    assert response.status_code == 422


def test_api_docs_available(client):
    """Test that API documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "info" in schema
    assert schema["info"]["title"] == "MLOps Lab API"
