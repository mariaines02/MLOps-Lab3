"""
Integration testing with the API

"""
import io
import json
import pytest
from api.api import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)


def test_home_endpoint(client):
    """Verify that the endpoint / returns the right message."""
    response = client.get("/")
    assert response.status_code == 200


def test_calculate_add(client):
    """Verify that the endpoint /calculate performs the sum correctly."""
    response = client.post(
        "/calculate",
        data={"op": "add", "a": 5, "b": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == 8

def test_calculate_subtract(client):
    """Verify that the endpoint /calculate performs the subtract correctly."""
    response = client.post(
        "/calculate",
        data={"op": "subtract", "a": 5, "b": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == 2

def test_calculate_multiply(client):
    """Verify that the endpoint /calculate performs the multiply correctly."""
    response = client.post(
        "/calculate",
        data={"op": "multiply", "a": 5, "b": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == 15

def test_calculate_divide(client):
    """Verify that the endpoint /calculate performs the divide correctly."""
    response = client.post(
        "/calculate",
        data={"op": "divide", "a": 6, "b": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == 2

def test_calculate_divide_by_zero(client):
    """Verify that the endpoint /calculate manages correctly the division by zero."""
    response = client.post("/calculate", data={"op": "divide", "a": 5, "b": 0})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Zero division not allowed"

def test_calculate_power(client):
    """Verify that the endpoint /calculate performs the power correctly."""
    response = client.post(
        "/calculate",
        data={"op": "power", "a": 2, "b": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == 8

def test_calculate_invalid_operation(client):
    """Verify that the endpoint /calculate manages correctly unvalid operations."""
    response = client.post(
        "/calculate", data={"op": "invalid_op", "a": 5, "b": 3}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Unvalid operation"


def test_calculate_invalid_parameters(client):
    """Verify that the endpoint /calculate manages correctly unvalid parameters."""
    response = client.post("/calculate", data={"op": "add", "a": "five", "b": 3})
    assert (
        response.status_code == 422 # FastAPI returns 422 for validation errors
    )  
    data = response.json()
    assert "detail" in data


def test_calculate_missing_parameters(client):
    """Verify that the endpoint /calculate manages correctly missing parameters."""
    response = client.post("/calculate", data={"op": "add", "a": 5})  # 'b' missed
    assert (
        response.status_code == 422 # FastAPI returns 422 for validation errors
    )
    data = response.json()
    assert "detail" in data
